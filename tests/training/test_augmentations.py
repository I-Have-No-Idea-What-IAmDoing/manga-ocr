"""Tests for the data augmentation pipeline builder.

This module contains unit tests for the functions in `augmentations.py`, which
are responsible for dynamically constructing `albumentations` data augmentation
pipelines from a configuration structure. The tests cover simple transforms,
nested transforms, and the handling of OpenCV constants.
"""

import albumentations as A
import cv2
import pytest

from manga_ocr_dev.training.augmentations import (
    build_augmentations,
    build_transforms,
)


def test_build_transforms_returns_empty_list_for_empty_config():
    """Tests that `build_transforms` returns an empty list for an empty config."""
    assert build_transforms([]) == []


def test_build_transforms_creates_simple_transform_list():
    """Tests that `build_transforms` correctly creates a simple list of transforms."""
    aug_list = [{"name": "HorizontalFlip", "params": {"p": 1}}]
    transforms = build_transforms(aug_list)
    assert len(transforms) == 1
    assert isinstance(transforms[0], A.HorizontalFlip)
    assert transforms[0].p == 1


def test_build_transforms_parses_cv2_constants():
    """Tests that `build_transforms` correctly parses and resolves `cv2` constants."""
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


def test_build_transforms_handles_nested_transforms():
    """Tests that `build_transforms` correctly handles nested transform definitions."""
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


def test_build_augmentations_returns_none_for_none_config():
    """Tests that `build_augmentations` returns `None` for a `None` config."""
    assert build_augmentations(None) is None


def test_build_augmentations_returns_none_for_empty_list():
    """Tests that `build_augmentations` returns `None` for an empty config list."""
    assert build_augmentations([]) is None


def test_build_augmentations_creates_simple_compose_pipeline():
    """Tests that `build_augmentations` correctly creates a simple `Compose` pipeline."""
    aug_list = [{"name": "HorizontalFlip", "params": {"p": 1}}]
    pipeline = build_augmentations(aug_list)
    assert isinstance(pipeline, A.Compose)
    assert len(pipeline.transforms) == 1
    assert isinstance(pipeline.transforms[0], A.HorizontalFlip)