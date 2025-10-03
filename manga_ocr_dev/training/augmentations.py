from typing import List, Dict, Any, Optional
import albumentations as A
import cv2

def build_transforms(aug_list: List[Dict[str, Any]]) -> List[A.BasicTransform]:
    """Builds a list of Albumentations transforms from a config list."""
    transforms = []
    for aug in aug_list:
        name = aug['name']
        params = aug.get('params', {}).copy()

        # Handle nested transforms for OneOf, etc.
        if 'transforms' in params:
            nested_transforms = build_transforms(params.pop('transforms'))
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


def build_augmentations(aug_config: Optional[List[Dict[str, Any]]]) -> Optional[A.Compose]:
    """Builds an Albumentations composition from a configuration list.

    Args:
        aug_config: A list of augmentation configurations, where each item
            is a dictionary with 'name' and 'params'.

    Returns:
        An Albumentations Compose object, or None if the config is empty.
    """
    if not aug_config:
        return None

    transforms = build_transforms(aug_config)
    return A.Compose(transforms)