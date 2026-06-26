#!/usr/bin/env python3
"""Cache raw calibration and test logits from trained BreakHis checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.patient_parser import collect_metadata, resolve_breakhis_root
from src.data.transforms import build_eval_transform
from src.models.architectures import build_model
from src.training.completion import RunExpectation, classify_training_run
from src.training.train import (
    load_training_config,
    magnification_tag,
    normalize_magnification,
    resolve_device,
)
from src.utils.io import ensure_parent_dir
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed

CHECKPOINT_RUN_RE = re.compile(r"^seed(?P<seed>\d+)_fold(?P<fold>\d+)$")


@dataclass(frozen=True, slots=True)
class CacheTarget:
    architecture: str
    magnification: str
    seed: int
    fold: int
    checkpoint_path: Path

    @property
    def output_filename(self) -> str:
        return f"seed{self.seed}_fold{self.fold}.npz"

    @property
    def run_label(self) -> str:
        return f"{self.architecture}/{self.magnification}/seed{self.seed}_fold{self.fold}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-path", type=Path, default=Path("data/folds/folds.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/logits"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/cache_logits.log"))
    return parser.parse_args()


def parse_checkpoint_path(checkpoint_path: Path, checkpoint_root: Path) -> CacheTarget:
    relative = checkpoint_path.resolve().relative_to(checkpoint_root.resolve())
    if relative.name != "best.pt" or len(relative.parts) != 4:
        raise ValueError(
            "Checkpoint path must match {arch}/{mag}/seed{s}_fold{f}/best.pt under the checkpoint root: "
            f"{checkpoint_path}"
        )

    architecture, magnification, run_dir, _filename = relative.parts
    match = CHECKPOINT_RUN_RE.match(run_dir)
    if match is None:
        raise ValueError(f"Checkpoint run directory does not match seed/fold naming: {run_dir}")

    return CacheTarget(
        architecture=architecture,
        magnification=magnification,
        seed=int(match.group("seed")),
        fold=int(match.group("fold")),
        checkpoint_path=checkpoint_path.resolve(),
    )


def parse_only_filter(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for segment in value.split(","):
        if "=" not in segment:
            raise ValueError(f"Invalid --only segment '{segment}'. Expected key=value.")
        key, raw_value = segment.split("=", maxsplit=1)
        pairs[key.strip()] = raw_value.strip()

    required = {"arch", "mag", "seed", "fold"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"--only is missing required keys: {sorted(missing)}")
    return pairs


def target_matches_only_filter(target: CacheTarget, only_filter: dict[str, str] | None) -> bool:
    if only_filter is None:
        return True

    requested_mag = magnification_tag(normalize_magnification(only_filter["mag"]))
    return (
        target.architecture == only_filter["arch"]
        and target.magnification == requested_mag
        and target.seed == int(only_filter["seed"])
        and target.fold == int(only_filter["fold"])
    )


def discover_targets(checkpoint_root: Path, only: str | None = None) -> list[CacheTarget]:
    only_filter = parse_only_filter(only) if only is not None else None
    targets: list[CacheTarget] = []

    for checkpoint_path in sorted(checkpoint_root.glob("*/*/seed*_fold*/best.pt")):
        target = parse_checkpoint_path(checkpoint_path, checkpoint_root)
        if target_matches_only_filter(target, only_filter):
            targets.append(target)

    if only_filter is not None and not targets:
        raise FileNotFoundError(
            "No checkpoint matched --only="
            f"{only!r} under checkpoint root {checkpoint_root.resolve()}"
        )

    return targets


def load_fold_split(folds_path: Path, seed: int, fold: int) -> dict[str, list[str]]:
    folds_data = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    return folds_data[str(seed)][str(fold)]


def build_eval_dataloader(
    metadata_df,
    patient_ids: list[str],
    magnification: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = BreakHisDataset(
        metadata_df=metadata_df,
        patient_ids=patient_ids,
        magnification=normalize_magnification(magnification),
        transform=build_eval_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_checkpoint_weights(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def collect_split_outputs(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    logits_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    patient_ids: list[str] = []

    with torch.no_grad():
        for images, labels, batch_patient_ids in dataloader:
            images = images.to(device)
            logits = model(images).detach().cpu().numpy()
            logits_batches.append(logits)
            labels_batches.append(labels.numpy())
            patient_ids.extend(list(batch_patient_ids))

    if logits_batches:
        logits = np.concatenate(logits_batches, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(labels_batches, axis=0).astype(np.int64, copy=False)
    else:
        logits = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)

    return logits, labels, np.asarray(patient_ids, dtype=str)


def output_path_for_target(output_root: Path, target: CacheTarget) -> Path:
    return output_root / target.architecture / target.magnification / target.output_filename


def cache_target_logits(
    target: CacheTarget,
    metadata_df,
    folds_path: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Path:
    split = load_fold_split(folds_path, seed=target.seed, fold=target.fold)
    cal_loader = build_eval_dataloader(
        metadata_df=metadata_df,
        patient_ids=split["cal"],
        magnification=target.magnification,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = build_eval_dataloader(
        metadata_df=metadata_df,
        patient_ids=split["test"],
        magnification=target.magnification,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(target.architecture, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(load_checkpoint_weights(target.checkpoint_path))
    model.eval()

    logits_cal, labels_cal, patient_ids_cal = collect_split_outputs(model, cal_loader, device)
    logits_test, labels_test, patient_ids_test = collect_split_outputs(model, test_loader, device)

    output_path = output_path_for_target(output_root, target)
    ensure_parent_dir(output_path)
    np.savez_compressed(
        output_path,
        logits_cal=logits_cal,
        labels_cal=labels_cal,
        patient_ids_cal=patient_ids_cal,
        logits_test=logits_test,
        labels_test=labels_test,
        patient_ids_test=patient_ids_test,
    )
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(log_path=args.log_file)
    set_global_seed(args.seed)

    device = resolve_device()
    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    breakhis_root = resolve_breakhis_root(args.data_root)
    metadata_df = collect_metadata(breakhis_root)
    targets = discover_targets(checkpoint_root, only=args.only)
    training_config = load_training_config(args.training_config)
    for target in targets:
        classification = classify_training_run(
            target.checkpoint_path.parent,
            RunExpectation(
                architecture=target.architecture,
                magnification=normalize_magnification(target.magnification),
                seed=target.seed,
                fold=target.fold,
                epochs=int(training_config["epochs"]),
                patience=int(training_config["early_stopping_patience"]),
            ),
        )
        if not classification.is_complete:
            raise RuntimeError(
                f"Refusing to cache logits for {target.run_label}: "
                f"{classification.status} ({classification.reason})"
            )

    total_targets = len(targets)
    cached_count = 0
    skipped_count = 0
    logging.info(
        "Caching logits from %s checkpoints | checkpoint_root=%s | output_root=%s | device=%s",
        total_targets,
        checkpoint_root,
        output_root,
        device,
    )

    for index, target in enumerate(targets, start=1):
        output_path = output_path_for_target(output_root, target)
        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            logging.info("[%s/%s] skipped %s | output exists", index, total_targets, target.run_label)
            continue

        start_time = time.perf_counter()
        cache_target_logits(
            target=target,
            metadata_df=metadata_df,
            folds_path=args.folds_path,
            output_root=output_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        cached_count += 1
        wall_seconds = time.perf_counter() - start_time
        logging.info(
            "[%s/%s] cached %s | wall=%s",
            index,
            total_targets,
            target.run_label,
            f"{wall_seconds:.2f}s",
        )

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logging.info(
        "Finished logit caching | cached=%s | skipped=%s | total=%s",
        cached_count,
        skipped_count,
        total_targets,
    )


if __name__ == "__main__":
    main()
