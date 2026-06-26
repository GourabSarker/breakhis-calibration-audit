"""Architecture factory for BreakHis experiments."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

from src.models.custom_cnn import CustomCNN


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a supported architecture with a two-logit output head."""
    normalized_name = name.lower()

    if normalized_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    if normalized_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if normalized_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported architecture '{name}'. Expected one of: custom_cnn, resnet50, efficientnet_b0."
    )
