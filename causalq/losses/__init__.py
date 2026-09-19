"""Training losses are introduced with their controlled experiments."""

from .causal_query_effect import causal_query_effect_loss
from .segmentation import segmentation_cross_entropy
from .prediction_consistency import prediction_consistency_kl
from .query_diversity import query_diversity_loss

__all__ = [
    "causal_query_effect_loss",
    "prediction_consistency_kl",
    "query_diversity_loss",
    "segmentation_cross_entropy",
]
