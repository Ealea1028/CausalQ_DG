"""Cityscapes 19-class label-space utilities."""

from __future__ import annotations

import numpy as np


IGNORE_INDEX = 255
NUM_CLASSES = 19

LABEL_ID_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

TRAIN_ID_PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


def label_ids_to_train_ids(label: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes label IDs to train IDs without mutating the input."""
    if label.ndim != 2:
        raise ValueError(f"Expected a 2-D label mask, got shape {label.shape}")
    converted = np.full(label.shape, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        converted[label == label_id] = train_id
    return converted


def invalid_train_ids(label: np.ndarray) -> set[int]:
    """Return IDs outside the 19-class train-ID space and ignore index."""
    allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    return {int(value) for value in np.unique(label)} - allowed


def colorize_train_ids(label: np.ndarray) -> np.ndarray:
    """Convert a train-ID mask to an RGB visualization."""
    invalid = invalid_train_ids(label)
    if invalid:
        raise ValueError(f"Mask contains invalid train IDs: {sorted(invalid)}")
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for train_id, color in enumerate(TRAIN_ID_PALETTE):
        rgb[label == train_id] = color
    return rgb

