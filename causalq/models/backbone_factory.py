"""Explicit backbone selection; legacy DINOv3 loader remains unchanged."""
from .dinov2_wrapper import DINOv2Backbone
from .dinov3_wrapper import DINOv3Backbone


def build_backbone(name, *, weights, **kwargs):
    if name == 'dinov2_vitl14_hpi_p16':
        return DINOv2Backbone.from_pretrained(weights, **kwargs)
    return DINOv3Backbone.from_pretrained(name, weights=weights, **kwargs)
