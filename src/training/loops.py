"""Training and evaluation loops for BreakHis models."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def _positive_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def aggregate_patient_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    patient_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Average image-level logits within each patient for patient-level evaluation."""
    if logits.shape[0] != labels.shape[0] or logits.shape[0] != len(patient_ids):
        raise ValueError("logits, labels, and patient_ids must have matching lengths.")

    grouped_logits: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
    grouped_labels: OrderedDict[str, int] = OrderedDict()
    for row_idx, patient_id in enumerate(patient_ids):
        grouped_logits.setdefault(patient_id, []).append(logits[row_idx])
        label_value = int(labels[row_idx].item())
        if patient_id in grouped_labels and grouped_labels[patient_id] != label_value:
            raise ValueError(f"Patient '{patient_id}' has inconsistent labels.")
        grouped_labels[patient_id] = label_value

    aggregated_logits = torch.stack(
        [torch.stack(values, dim=0).mean(dim=0) for values in grouped_logits.values()],
        dim=0,
    )
    aggregated_labels = torch.tensor(list(grouped_labels.values()), dtype=torch.long)
    return aggregated_logits, aggregated_labels, list(grouped_logits.keys())


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run a single optimization epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels, _patient_ids in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "num_samples": float(total_samples),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a model and return image-level plus patient-level outputs."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    logits_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    patient_ids: list[str] = []

    for images, labels, batch_patient_ids in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        logits_batches.append(logits.detach().cpu())
        labels_batches.append(labels.detach().cpu())
        patient_ids.extend(list(batch_patient_ids))

    logits_tensor = torch.cat(logits_batches, dim=0)
    labels_tensor = torch.cat(labels_batches, dim=0)
    probs_tensor = _positive_class_probabilities(logits_tensor)

    patient_logits, patient_labels, patient_id_order = aggregate_patient_logits(
        logits_tensor,
        labels_tensor,
        patient_ids,
    )
    patient_probs = _positive_class_probabilities(patient_logits)

    weight = criterion.weight.detach().cpu() if getattr(criterion, "weight", None) is not None else None
    patient_criterion = nn.CrossEntropyLoss(weight=weight)
    patient_loss = float(patient_criterion(patient_logits, patient_labels).item())
    image_loss = total_loss / max(total_samples, 1)

    return {
        "loss": image_loss,
        "image_loss": image_loss,
        "patient_loss": patient_loss,
        "logits": logits_tensor.numpy(),
        "labels": labels_tensor.numpy(),
        "probs": probs_tensor.numpy(),
        "patient_logits": patient_logits.numpy(),
        "patient_labels": patient_labels.numpy(),
        "patient_probs": patient_probs.numpy(),
        "patient_ids": np.asarray(patient_id_order, dtype=object),
        "num_samples": int(total_samples),
        "num_patients": len(patient_id_order),
    }
