import pytest
import torch
from torch.nn import functional as F
from causalq.models.dinov2_wrapper import Attention, DinoVisionTransformer, DINOv2Backbone, convert_hpi_state
from causalq.models.query_segmentor import QuerySegmentor


def tiny_backbone():
    return DINOv2Backbone(DinoVisionTransformer(dim=32, depth=4, heads=4, img_size=32), intermediate_indices=(0, 1, 2, 3))


def test_sdpa_matches_explicit_attention():
    torch.manual_seed(11)
    module = Attention(32, 4).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    q, k, v = module.qkv(x).reshape(2, 7, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
    reference = module.proj(((q @ k.transpose(-2, -1) / 8**.5).softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 32))
    torch.testing.assert_close(module(x), reference, atol=1e-12, rtol=1e-12)


def test_hpi_conversion_changes_only_two_tensors_and_preserves_cls():
    state = {'patch_embed.proj.weight': torch.randn(8, 3, 14, 14),
             'pos_embed': torch.randn(1, 10, 8), 'other': torch.randn(2)}
    converted = convert_hpi_state(state, dim=8, original_grid=3, target_grid=2)
    assert converted['other'] is state['other']
    assert torch.equal(converted['pos_embed'][:, :1], state['pos_embed'][:, :1])
    torch.testing.assert_close(converted['patch_embed.proj.weight'], F.interpolate(state['patch_embed.proj.weight'], (16, 16), mode='bicubic', align_corners=False))
    assert converted['pos_embed'].shape == (1, 5, 8)
    with pytest.raises(ValueError): convert_hpi_state(state)


def test_features_raw_blocks_normalized_tokens_and_strict_keys():
    backbone = tiny_backbone().train()
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
    x = torch.randn(2, 3, 32, 48)
    features = backbone(x)
    assert features.patch_tokens.shape == (2, 6, 32)
    raw = features.intermediate_maps[-1].flatten(2).transpose(1, 2)
    torch.testing.assert_close(features.patch_tokens, backbone.model.norm(raw))
    clone = tiny_backbone()
    clone.load_state_dict(backbone.state_dict(), strict=True)
    assert 'model.blocks.3.ls2.gamma' in clone.state_dict()
    assert 'model.mask_token' in clone.state_dict()
    with pytest.raises(ValueError): backbone(torch.randn(1, 3, 31, 32))


@pytest.mark.parametrize('anchor', [False, True])
def test_preextracted_feature_head_matches_image_api_and_gradients(anchor):
    torch.manual_seed(0)
    model = QuerySegmentor(tiny_backbone(), decoder_channels=32, queries_per_class=2,
                           interaction='static', dropout=0., num_classes=3)
    image = torch.randn(1, 3, 32, 32)
    features = model.backbone(image)
    first = model.forward_components(image, include_anchor=anchor, component_protocol='hpi_dinov2_20261031')
    second = model.forward_features(features, image.shape[-2:], include_anchor=anchor, component_protocol='hpi_dinov2_20261031')
    assert torch.equal(first.logits, second.logits)
    second.logits.square().mean().backward()
    assert model.decoder.classifier.weight.grad is not None
    assert all(p.grad is None for p in model.backbone.parameters())
