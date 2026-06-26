"""Train one architecture/magnification/seed/fold cell."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.patient_parser import collect_metadata, resolve_breakhis_root
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.architectures import build_model
from src.training.loops import evaluate, train_one_epoch
from src.training.metrics import compute_metric_dict
from src.utils.io import ensure_parent_dir, save_json
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed


@dataclass(slots=True)
class TrainingConfig:
    architecture: str
    magnification: int
    seed: int
    fold: int
    data_root: Path = Path("data/raw")
    folds_path: Path = Path("data/folds/folds.json")
    output_root: Path = Path("outputs/checkpoints")
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    patience: int = 5
    pretrained: bool = True
    num_workers: int = 0
    run_kind: str = "full"


def magnification_tag(magnification: int | str) -> str:
    return f"{normalize_magnification(magnification)}X"


def normalize_magnification(magnification: int | str) -> int:
    return int(str(magnification).upper().removesuffix("X"))


def resolve_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def format_seconds(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def checkpoint_dir_for_run(
    output_root: Path,
    architecture: str,
    magnification: int | str,
    seed: int,
    fold: int,
) -> Path:
    return output_root / architecture / magnification_tag(magnification) / f"seed{seed}_fold{fold}"


def checkpoint_path_for_run(
    output_root: Path,
    architecture: str,
    magnification: int | str,
    seed: int,
    fold: int,
) -> Path:
    return checkpoint_dir_for_run(output_root, architecture, magnification, seed, fold) / "best.pt"


def history_path_for_run(
    output_root: Path,
    architecture: str,
    magnification: int | str,
    seed: int,
    fold: int,
) -> Path:
    return checkpoint_dir_for_run(output_root, architecture, magnification, seed, fold) / "history.json"


def summary_path_for_run(
    output_root: Path,
    architecture: str,
    magnification: int | str,
    seed: int,
    fold: int,
) -> Path:
    return checkpoint_dir_for_run(output_root, architecture, magnification, seed, fold) / "summary.json"


def load_fold_assignments(folds_path: Path, seed: int, fold: int) -> dict[str, list[str]]:
    folds_data = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    return folds_data[str(seed)][str(fold)]


def build_class_weights(
    metadata_df: pd.DataFrame,
    patient_ids: list[str],
    magnification: int | str,
) -> torch.Tensor:
    mag_value = normalize_magnification(magnification)
    filtered = metadata_df.loc[
        metadata_df["patient_id"].isin(patient_ids) & (metadata_df["magnification"] == mag_value)
    ]
    counts = filtered["label"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError(
            "Each training split must include both classes for weighted loss. "
            f"Observed counts: {counts.to_dict()}"
        )
    weights = torch.tensor([1.0 / counts.loc[0], 1.0 / counts.loc[1]], dtype=torch.float32)
    return weights


def build_dataloader(
    dataset: BreakHisDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        pin_memory=False,
    )


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    config: TrainingConfig,
    epoch: int,
    best_val_patient_loss: float,
    device: torch.device,
) -> None:
    ensure_parent_dir(checkpoint_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_patient_loss": best_val_patient_loss,
            "config": {
                **asdict(config),
                "data_root": str(config.data_root),
                "folds_path": str(config.folds_path),
                "output_root": str(config.output_root),
            },
            "device": str(device),
        },
        checkpoint_path,
    )


def run_training(
    config: TrainingConfig,
    *,
    metadata_df: pd.DataFrame | None = None,
    folds_data: dict[str, Any] | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Train one experimental cell and persist the checkpoint plus JSON logs."""
    set_global_seed(config.seed)

    device = resolve_device()
    mag_value = normalize_magnification(config.magnification)
    checkpoint_dir = checkpoint_dir_for_run(
        config.output_root, config.architecture, mag_value, config.seed, config.fold
    )
    checkpoint_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "history.json"
    summary_path = checkpoint_dir / "summary.json"
    ensure_parent_dir(checkpoint_path)
    existing_artifacts = [
        path for path in (checkpoint_path, history_path, summary_path) if path.exists()
    ]
    if existing_artifacts and not overwrite_existing:
        names = ", ".join(path.name for path in existing_artifacts)
        raise FileExistsError(
            f"Run directory {checkpoint_dir} already contains {names}. "
            "Back it up and pass overwrite_existing=True only for an intentional rerun."
        )

    running_summary = {
        "completion_status": "running",
        "run_kind": config.run_kind,
        "architecture": config.architecture,
        "magnification": mag_value,
        "seed": config.seed,
        "fold": config.fold,
        "device": str(device),
        "python_version": platform.python_version(),
        "epochs_requested": config.epochs,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
    }
    save_json(running_summary, summary_path)

    if metadata_df is None:
        metadata_df = collect_metadata(resolve_breakhis_root(config.data_root))
    if folds_data is None:
        folds_data = json.loads(Path(config.folds_path).read_text(encoding="utf-8"))

    split = folds_data[str(config.seed)][str(config.fold)]
    train_patient_ids = split["train"]
    cal_patient_ids = split["cal"]

    train_dataset = BreakHisDataset(
        metadata_df=metadata_df,
        patient_ids=train_patient_ids,
        magnification=mag_value,
        transform=build_train_transform(),
    )
    cal_dataset = BreakHisDataset(
        metadata_df=metadata_df,
        patient_ids=cal_patient_ids,
        magnification=mag_value,
        transform=build_eval_transform(),
    )
    if len(train_dataset) == 0 or len(cal_dataset) == 0:
        raise ValueError(
            f"Encountered an empty dataset for architecture={config.architecture}, "
            f"magnification={mag_value}, seed={config.seed}, fold={config.fold}."
        )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    cal_loader = build_dataloader(
        cal_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    class_weights = build_class_weights(metadata_df, train_patient_ids, mag_value).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(config.architecture, num_classes=2, pretrained=config.pretrained).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_patient_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = evaluate(model, cal_loader, criterion, device)
        scheduler.step()

        patient_metrics = compute_metric_dict(val_stats["patient_labels"], val_stats["patient_probs"])
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["image_loss"]),
            "val_image_loss": float(val_stats["image_loss"]),
            "val_patient_loss": float(val_stats["patient_loss"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(epoch_seconds),
            "num_train_images": int(train_stats["num_samples"]),
            "num_val_images": int(val_stats["num_samples"]),
            "num_val_patients": int(val_stats["num_patients"]),
            **{f"val_{name}": value for name, value in patient_metrics.items()},
        }
        history.append(epoch_record)
        save_json(history, history_path)

        logging.info(
            "Epoch %s/%s | arch=%s mag=%s seed=%s fold=%s | train_loss=%.4f | "
            "val_image_loss=%.4f | val_patient_loss=%.4f | val_bal_acc=%.4f | wall=%.2fs",
            epoch,
            config.epochs,
            config.architecture,
            magnification_tag(mag_value),
            config.seed,
            config.fold,
            epoch_record["train_loss"],
            epoch_record["val_image_loss"],
            epoch_record["val_patient_loss"],
            epoch_record["val_balanced_accuracy"],
            epoch_record["epoch_seconds"],
        )

        if epoch_record["val_patient_loss"] < best_val_patient_loss - 1e-8:
            best_val_patient_loss = epoch_record["val_patient_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_val_patient_loss=best_val_patient_loss,
                device=device,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logging.info(
                    "Early stopping triggered at epoch %s after %s epochs without improvement.",
                    epoch,
                    config.patience,
                )
                break

    total_runtime_seconds = time.perf_counter() - start_time
    best_epoch_record = history[best_epoch - 1] if best_epoch else None
    summary = {
        "completion_status": "complete",
        "run_kind": config.run_kind,
        "architecture": config.architecture,
        "magnification": mag_value,
        "seed": config.seed,
        "fold": config.fold,
        "device": str(device),
        "python_version": platform.python_version(),
        "epochs_requested": config.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_image_loss": (
            float(best_epoch_record["val_image_loss"]) if best_epoch_record is not None else None
        ),
        "best_val_patient_loss": float(best_val_patient_loss),
        "runtime_seconds": float(total_runtime_seconds),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "stopped_early": len(history) < config.epochs,
    }
    save_json(summary, summary_path)
    return summary


def load_training_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--magnification", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-path", type=Path, default=Path("data/folds/folds.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-existing-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = TrainingConfig(
        architecture=args.architecture,
        magnification=normalize_magnification(args.magnification),
        seed=args.seed,
        fold=args.fold,
        data_root=args.data_root,
        folds_path=args.folds_path,
        output_root=args.output_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
    )
    summary = run_training(config, overwrite_existing=args.overwrite_existing_run)
    logging.info(
        "Run complete | arch=%s mag=%s seed=%s fold=%s | total=%s | best_epoch=%s",
        config.architecture,
        magnification_tag(config.magnification),
        config.seed,
        config.fold,
        format_seconds(summary["runtime_seconds"]),
        summary["best_epoch"],
    )


if __name__ == "__main__":
    main()
