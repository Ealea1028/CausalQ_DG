"""Phase35 model construction and two-view objective."""
import torch

from causalq.losses.hpi_localization import patch_localization_loss
from causalq.losses.source_ce import deterministic_source_ce
from causalq.models.hpi_anchor_only import freeze_query_residuals
from causalq.models.hpi_query_route import QueryFeatureRouter, RoutedQuerySegmentor
from causalq.models.query_segmentor import QuerySegmentor
from causalq.protocols.hpi_dinov2_20261031 import build_head, initialization_identity


def build_ablation_head(backbone, variant, expected_initial_sha, **kwargs):
    if variant not in ('B1', 'B2', 'B3'):
        raise ValueError('Phase35 supports B1, B2 and B3')
    model = build_head(backbone, head_type=QuerySegmentor if variant == 'B1' else RoutedQuerySegmentor, **kwargs)
    if expected_initial_sha is not None and initialization_identity(model)['initial_head_sha256'] != expected_initial_sha:
        raise ValueError('Phase35 common initialization differs from Phase34')
    freeze_query_residuals(model)
    if variant in ('B2', 'B3'):
        # Both variants start with the same router without advancing the head RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(20261031).get_state())
            model.query_router = QueryFeatureRouter(backbone.hidden_size)
    return model


def ablation_backward(head, original, photo, label, output_size, *, amp=False):
    ce_losses = []
    localization = []
    for features in (original, photo):
        with torch.autocast(label.device.type, dtype=torch.bfloat16, enabled=amp):
            if head.name == 'B3':
                output, route = head.model.forward_routed(features, output_size)
                loc = .05 * patch_localization_loss(route.match_logits, label, features.patch_map.shape[-2:])
            else:
                output = head.model.forward_features(features, output_size)
                loc = None
            ce = deterministic_source_ce(output.logits, label)
        (ce if loc is None else ce + loc).backward()
        ce_losses.append(ce.detach())
        localization.append(ce.new_zeros(()) if loc is None else loc.detach())
    return torch.stack((*ce_losses, localization[0] + localization[1]))
