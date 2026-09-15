"""Dataset adapters use the Cityscapes 19-class label protocol."""

from .segmentation import (
    PairedSegmentationDataset,
    SegmentationPair,
    TrainTransform,
    cityscapes_dataset,
    gta5_dataset,
)

__all__ = [
    "PairedSegmentationDataset",
    "SegmentationPair",
    "TrainTransform",
    "cityscapes_dataset",
    "gta5_dataset",
]
