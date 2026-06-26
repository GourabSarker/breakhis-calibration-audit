from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training.loops import aggregate_patient_logits, evaluate, train_one_epoch


class TinyPatientDataset(Dataset):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.patient_ids = ["patient_a", "patient_a", "patient_b", "patient_b"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.patient_ids[index]


def test_aggregate_patient_logits_averages_by_patient() -> None:
    logits = torch.tensor(
        [[2.0, 0.0], [4.0, 0.0], [0.0, 1.0], [0.0, 3.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    patient_ids = ["a", "a", "b", "b"]

    patient_logits, patient_labels, ordered_ids = aggregate_patient_logits(logits, labels, patient_ids)

    assert ordered_ids == ["a", "b"]
    assert torch.allclose(patient_logits, torch.tensor([[3.0, 0.0], [0.0, 2.0]]))
    assert torch.equal(patient_labels, torch.tensor([0, 1]))


def test_train_one_epoch_updates_model_parameters() -> None:
    dataset = TinyPatientDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    before = [param.detach().clone() for param in model.parameters()]
    stats = train_one_epoch(model, dataloader, optimizer, criterion, torch.device("cpu"))
    after = [param.detach().clone() for param in model.parameters()]

    assert stats["loss"] > 0.0
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_evaluate_returns_patient_level_outputs() -> None:
    dataset = TinyPatientDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()

    stats = evaluate(model, dataloader, criterion, torch.device("cpu"))

    assert stats["num_samples"] == 4
    assert stats["num_patients"] == 2
    assert "image_loss" in stats
    assert len(stats["probs"]) == 4
    assert len(stats["patient_probs"]) == 2
    assert set(stats["patient_ids"].tolist()) == {"patient_a", "patient_b"}
