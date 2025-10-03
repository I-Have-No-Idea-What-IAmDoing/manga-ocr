"""Factory functions for building data augmentation pipelines.

This script provides helper functions to dynamically construct data augmentation
pipelines using `albumentations`. It parses a configuration structure, typically
defined in a YAML file, and translates it into a composable `albumentations`
pipeline. This allows for flexible and easily configurable data augmentation
without changing the training code.
"""

from typing import Any, Dict, List, Optional

import albumentations as A
import cv2


def build_transforms(aug_list: List[Dict[str, Any]]) -> List[A.BasicTransform]:
    """Recursively builds a list of Albumentations transforms from a config.

    This function parses a list of dictionaries, where each dictionary defines
    an augmentation, and constructs a corresponding list of Albumentations
    transform objects. It supports nested transforms (e.g., for `A.OneOf`) and
    can resolve OpenCV constants provided as strings (e.g., "cv2.BORDER_CONSTANT").

    Args:
        aug_list: A list of augmentation configurations. Each item is a
            dictionary with a 'name' key for the transform class and an
            optional 'params' key for its parameters.

    Returns:
        A list of instantiated Albumentations transform objects.
    """
    transforms = []
    for aug in aug_list:
        name = aug["name"]
        params = aug.get("params", {}).copy()

        # Handle nested transforms for OneOf, etc.
        if "transforms" in params:
            nested_transforms = build_transforms(params.pop("transforms"))
        else:
            nested_transforms = []

        # Special handling for cv2 constants
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("cv2."):
                params[key] = getattr(cv2, value.split(".")[-1])

        transform_class = getattr(A, name)

        if nested_transforms:
            # Assumes the nested transforms are the first positional argument
            instance = transform_class(nested_transforms, **params)
        else:
            instance = transform_class(**params)
        transforms.append(instance)

    return transforms


def build_augmentations(
    aug_config: Optional[List[Dict[str, Any]]]
) -> Optional[A.Compose]:
    """Builds an Albumentations composition from a configuration list.

    This function serves as the main entry point for creating an augmentation
    pipeline from a configuration file. It takes a list of augmentation
    definitions and uses `build_transforms` to construct the full pipeline,
    which is then wrapped in an `A.Compose` object.

    Args:
        aug_config: A list of augmentation configurations, where each item
            is a dictionary with 'name' and 'params'.

    Returns:
        An Albumentations `Compose` object representing the full augmentation
        pipeline, or `None` if the input configuration is empty or None.
    """
    if not aug_config:
        return None

    transforms = build_transforms(aug_config)
    return A.Compose(transforms)