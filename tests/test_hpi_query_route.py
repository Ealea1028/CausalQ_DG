import torch
from causalq.models.hpi_anchor_only import freeze_query_residuals
from causalq.models.hpi_query_route import QueryFeatureRouter, RoutedQuerySegmentor
from causalq.protocols.hpi_dinov2_20261031 import build_head
from causalq.runtime.feature_pipeline import feature_tensors
from causalq.runtime.identity import tree_digest
from test_hpi_backbone import tiny_backbone


def test_b2_writeback_keeps_shared_inputs_and_gradients():
    backbone = tiny_backbone()
    model = freeze_query_residuals(build_head(backbone, decoder_channels=32, num_classes=3, head_type=RoutedQuerySegmentor))
    model.query_router = QueryFeatureRouter(32, route_width=32)
    features = backbone(torch.randn(2,3,32,32))
    digest = tree_digest(feature_tensors(features))
    out, route = model.forward_routed(features,(32,32))
    assert route.patch_attention.shape == (2,8,3,4)
    assert torch.allclose(route.patch_attention.sum(-1),torch.ones(2,8,3))
    assert route.injected.abs().sum()>0
    out.logits.square().mean().backward()
    assert model.query_router.value.weight.grad.abs().sum()>0
    assert model.get_anchors().grad.abs().sum()>0
    assert model.query_bank.query_residuals.grad is None
    assert tree_digest(feature_tensors(features)) == digest
    closed = model.query_router(features.patch_tokens,model.get_anchors(),(2,2),active_classes=torch.zeros(2,3,dtype=torch.bool))
    assert torch.count_nonzero(closed.injected)==0
    model.eval()
    classes=torch.tensor([0,2]);off,_=model.forward_routed(features,(32,32),closed_classes=classes)
    logits=model.closed_class_logits(features,(32,32),classes)
    assert torch.equal(logits[torch.arange(2),classes],off.base_logits[torch.arange(2),classes])
