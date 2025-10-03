import albumentations as A
import cv2
import pytest

from manga_ocr_dev.training.augmentations import (
    build_augmentations,
    build_transforms,
)


def test_build_transforms_empty():
    """Test that building transforms with an empty list returns an empty list."""
    assert build_transforms([]) == []


def test_build_transforms_simple():
    """Test building a simple list of transforms."""
    aug_list = [{"name": "HorizontalFlip", "params": {"p": 1}}]
    transforms = build_transforms(aug_list)
    assert len(transforms) == 1
    assert isinstance(transforms[0], A.HorizontalFlip)
    assert transforms[0].p == 1


def test_build_transforms_with_cv2_constant():
    """Test building transforms with a cv2 constant."""
    aug_list = [
        {
            "name": "ShiftScaleRotate",
            "params": {"border_mode": "cv2.BORDER_CONSTANT"},
        }
    ]
    transforms = build_transforms(aug_list)
    assert len(transforms) == 1
    assert isinstance(transforms[0], A.ShiftScaleRotate)
    assert transforms[0].border_mode == cv2.BORDER_CONSTANT


def test_build_transforms_nested():
    """Test building nested transforms."""
    aug_list = [
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "HorizontalFlip", "params": {"p": 1}},
                    {"name": "VerticalFlip", "params": {"p": 1}},
                ]
            },
        }
    ]
    transforms = build_transforms(aug_list)
    assert len(transforms) == 1
    assert isinstance(transforms[0], A.OneOf)
    assert len(transforms[0].transforms) == 2
    assert isinstance(transforms[0].transforms[0], A.HorizontalFlip)
    assert isinstance(transforms[0].transforms[1], A.VerticalFlip)


def test_build_augmentations_none():
    """Test that building augmentations with None returns None."""
    assert build_augmentations(None) is None


def test_build_augmentations_empty():
    """Test that building augmentations with an empty list returns None."""
    assert build_augmentations([]) is None


def test_build_augmentations_simple():
    """Test building a simple augmentation pipeline."""
    aug_list = [{"name": "HorizontalFlip", "params": {"p": 1}}]
    pipeline = build_augmentations(aug_list)
    assert isinstance(pipeline, A.Compose)
    assert len(pipeline.transforms) == 1
    assert isinstance(pipeline.transforms[0], A.HorizontalFlip)