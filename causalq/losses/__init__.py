"""Training losses are introduced with their controlled experiments."""

from .segmentation import segmentation_cross_entropy

__all__ = ["segmentation_cross_entropy"]
