from __future__ import annotations

import pytest
import torch

from src.models.architectures import build_model
from src.models.custom_cnn import CustomCNN


@pytest.mark.parametrize("architecture", ["custom_cnn", "resnet50", "efficientnet_b0"])
def test_build_model_outputs_two_logits(architecture: str) -> None:
    model = build_model(architecture, num_classes=2, pretrained=False)
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (2, 2)


def test_custom_cnn_exposes_expected_linear_layers() -> None:
    model = CustomCNN(num_classes=2)
    linear_layers = [layer for layer in model.classifier if isinstance(layer, torch.nn.Linear)]
    assert linear_layers[0].in_features == 32
    assert linear_layers[0].out_features == 128
    assert linear_layers[1].out_features == 2


def test_build_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model("unknown_model", pretrained=False)
