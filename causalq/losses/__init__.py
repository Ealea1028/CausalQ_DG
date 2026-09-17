"""Training losses are introduced with their controlled experiments."""

from .segmentation import segmentation_cross_entropy
from .prediction_consistency import prediction_consistency_kl

__all__ = ["prediction_consistency_kl", "segmentation_cross_entropy"]
