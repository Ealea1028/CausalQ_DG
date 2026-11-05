"""B1 retains the two-slot readout with one trainable anchor per class."""
import torch


def freeze_query_residuals(model):
    if model.interaction != 'static' or model.queries_per_class != 2:
        raise ValueError('Anchor-only requires the static two-slot readout')
    residual = model.query_bank.query_residuals
    with torch.no_grad():
        residual.zero_()
    residual.requires_grad_(False)
    model.residual_status = 'structurally_absent'
    model.effective_independent_queries_per_class = 1
    return model


def verify_frozen_residuals(model):
    residual = model.query_bank.query_residuals
    if residual.requires_grad or torch.count_nonzero(residual):
        raise ValueError('Expected a frozen zero residual in the compact head state')
