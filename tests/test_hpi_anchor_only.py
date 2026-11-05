import torch
from causalq.models.hpi_anchor_only import freeze_query_residuals, verify_frozen_residuals
from causalq.protocols.hpi_dinov2_20261031 import build_head
from causalq.runtime.cohort_runner import head_state
from test_hpi_backbone import tiny_backbone


def test_b1_preserves_common_weights_and_repeated_slot_logsumexp():
    backbone = tiny_backbone()
    model = build_head(backbone, decoder_channels=32, num_classes=3).eval()
    before = {key: value.clone() for key, value in head_state(model).items()}
    freeze_query_residuals(model)
    verify_frozen_residuals(model)
    for key, value in head_state(model).items():
        if key != 'query_bank.query_residuals':
            assert torch.equal(value, before[key])
    output = model.forward_components(torch.randn(1,3,32,32),include_anchor=True,component_protocol='hpi_dinov2_20261031')
    assert torch.equal(output.query_score_maps[:,:,0],output.query_score_maps[:,:,1])
    assert torch.equal(output.scaled_delta_logits,output.anchor_scaled_delta_logits)
    assert model.effective_independent_queries_per_class == 1
    assert 'query_bank.query_residuals' in head_state(model)
