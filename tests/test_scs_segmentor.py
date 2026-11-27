import pytest
import torch
from torch import nn

from causalq.models.dinov2_wrapper import DINOv2Backbone, DinoVisionTransformer
from causalq.models.dinov3_wrapper import DINOv3Backbone
from causalq.models.scs import SCSConfig
from causalq.models.scs_segmentor import SCSBackbone, SCSMask2Former, rein_pyramid
from causalq.runtime.scs_training import (
    build_scs_optimizer, load_scs_trainable_state, paired_scs_backward, scs_trainable_state,
)


def test_scs_dino2_preserves_native_features_and_keeps_frozen_suffix_differentiable():
    torch.manual_seed(0)
    native = DINOv2Backbone(DinoVisionTransformer(dim=16, depth=4, heads=2, img_size=64),
                            intermediate_indices=(0, 1, 2, 3))
    model = SCSBackbone(native, SCSConfig(16, 8, 3, 2, 2, (2, 4)), taps=(1, 2, 3, 4)).train()
    images = torch.randn(1, 3, 64, 96)
    original = native(images)
    adapted, aux = model(images, return_aux=True)
    torch.testing.assert_close(adapted.cls_token, original.cls_token, atol=0, rtol=0)
    torch.testing.assert_close(adapted.patch_tokens, original.patch_tokens, atol=0, rtol=0)
    for left, right in zip(adapted.intermediate_maps, original.intermediate_maps):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert aux.shape == (1, 3, 4, 6)
    assert [x.shape[-2:] for x in rein_pyramid(adapted)] == [(16, 24), (8, 12), (4, 6), (2, 3)]
    adapted.intermediate_maps[-1].square().mean().backward()
    assert model.injector.bridges["2"].up.weight.grad.norm() > 0
    assert not native.training and all(p.grad is None for p in native.parameters())


def test_scs_dino3_agrees_with_native_forward():
    transformers = pytest.importorskip("transformers")
    config_class = getattr(transformers, "DINOv3ViTConfig", None)
    if config_class is None:
        pytest.skip("Native DINOv3 requires the SCS Transformers runtime")
    config = config_class(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                          intermediate_size=64, num_register_tokens=2)
    native = DINOv3Backbone(transformers.DINOv3ViTModel(config), intermediate_indices=(1, 2, 3))
    images = torch.randn(1, 3, 64, 96)
    expected = native(images)
    model = SCSBackbone(native, SCSConfig(32, 8, 3, 2, 2, (2, 4)), taps=(1, 2, 3, 4))
    actual, aux = model(images, return_aux=True)
    for field in ("patch_map", "patch_tokens", "cls_token", "register_tokens"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    # Transformers exposes its post-norm last hidden state. The preceding taps are raw.
    for adapted, original in zip(actual.intermediate_maps[:3], expected.intermediate_maps):
        torch.testing.assert_close(adapted, original, rtol=0, atol=0)
    assert aux.shape == (1, 3, 4, 6)


def test_optimizer_groups_and_compact_state():
    native = DINOv2Backbone(DinoVisionTransformer(dim=16, depth=4, heads=2), intermediate_indices=())
    backbone = SCSBackbone(native, SCSConfig(16, 8, 3, 2, 2, (2, 4)), taps=(1, 2, 3, 4))
    model = SCSMask2Former(backbone, nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16)))
    optimizer, scheduler = build_scs_optimizer(model)
    groups = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
    assert set(groups) == {id(p) for p in model.parameters() if p.requires_grad}
    assert groups[id(backbone.injector.query_bank)] == groups[id(model.query_embed.weight)] == 0
    assert groups[id(backbone.injector.read.in_proj_weight)] == 0.05
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * (1 - 1 / 40_000)**0.9)
    state = scs_trainable_state(model)
    assert set(state) == {"head", "query_embed", "injector"}
    load_scs_trainable_state(model, state)


def test_paired_backward_weights_the_two_views_equally():
    class Objective(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def loss(self, images, labels, **kwargs):
            return {"loss": self.weight * images.mean()}

    model = Objective()
    loss = paired_scs_backward(model, torch.tensor([2.0]), torch.tensor([4.0]), torch.tensor([]), scale=0.5)
    assert loss == model.weight.grad == 1.5
