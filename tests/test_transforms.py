from __future__ import annotations

from PIL import Image
from torchvision import transforms

from src.data.transforms import build_eval_transform, build_train_transform


def test_train_transform_contains_expected_augmentations() -> None:
    transform = build_train_transform()
    transform_types = [type(step) for step in transform.transforms]

    assert transforms.Resize in transform_types
    assert transforms.RandomHorizontalFlip in transform_types
    assert transforms.RandomVerticalFlip in transform_types
    assert transforms.RandomRotation in transform_types
    assert transforms.ColorJitter in transform_types
    assert transforms.ToTensor in transform_types
    assert transforms.Normalize in transform_types


def test_eval_transform_has_no_augmentation_and_outputs_tensor() -> None:
    transform = build_eval_transform()
    transform_types = [type(step) for step in transform.transforms]

    assert transform_types == [transforms.Resize, transforms.ToTensor, transforms.Normalize]

    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
