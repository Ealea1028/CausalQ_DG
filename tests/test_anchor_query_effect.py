import math

import pytest
import torch

from causalq.models import QuerySegmentor
from test_strict_source_smoke import TinyFrozenBackbone


class CountBackbone(TinyFrozenBackbone):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return super().forward(images)


def make_model(count=2, centered=True, interaction="static", learned_null=False):
    return QuerySegmentor(CountBackbone(), decoder_channels=32, num_classes=4,
        dropout=.1, queries_per_class=count, num_heads=2, cross_attention_layers=1,
        interaction=interaction, centered_residuals=centered, learned_null=learned_null).eval()


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_anchor_component_algebra_shared_head_and_single_backbone(count, batch):
    model = make_model(count)
    with torch.no_grad():
        model.query_head.alpha.fill_(.4)
    image = torch.randn(batch, 3, 8, 8)
    output = model.forward_components(image, include_anchor=True)
    assert model.backbone.calls == 1
    full = model.get_query_effect(output=output).float()
    anchor = model.get_anchor_query_effect(output=output)
    residual = model.get_residual_contribution(output=output)
    assert model.backbone.calls == 1
    torch.testing.assert_close(full, anchor + residual, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.anchor_logits - output.base_logits, anchor, atol=1e-6, rtol=1e-6)
    assert output.anchor_query_states.shape == (batch, 4, count, 8)
    assert torch.equal(output.anchor_query_states[:, :, 0], model.get_anchors().expand(batch, -1, -1))
    if count == 1:
        assert torch.count_nonzero(residual) == 0


def test_anchor_and_zero_intervention_touch_only_requested_class():
    model = make_model()
    with torch.no_grad():
        model.query_head.alpha.fill_(.2)
    output = model.forward_components(torch.randn(2, 3, 8, 8), include_anchor=True)
    for c in range(4):
        anchor, zero = output.anchor_counterfactual_logits(c), output.counterfactual_logits(c)
        assert torch.equal(anchor[:, c], output.anchor_logits[:, c])
        assert torch.equal(zero[:, c], output.base_logits[:, c])
        for other in set(range(4)) - {c}:
            assert torch.equal(anchor[:, other], output.logits[:, other])
            assert torch.equal(zero[:, other], output.logits[:, other])
    with pytest.raises(IndexError):
        output.anchor_counterfactual_logits(4)


def test_anchor_does_not_depend_on_raw_residual_and_gets_no_residual_gradient():
    model = make_model()
    with torch.no_grad():
        model.query_head.alpha.fill_(.3)
    image = torch.randn(1, 3, 8, 8)
    first = model.get_anchor_query_effect(image)
    with torch.no_grad():
        model.get_query_residuals().add_(torch.randn_like(model.get_query_residuals()) * 3)
    second = model.get_anchor_query_effect(image)
    assert torch.equal(first, second)
    second.sum().backward()
    assert model.get_query_residuals().grad is None
    assert model.get_anchors().grad is not None and model.get_anchors().grad.norm() > 0


def test_zero_effective_residual_gives_equal_full_and_anchor():
    model = make_model(3)
    with torch.no_grad():
        model.query_head.alpha.fill_(.3)
        model.query_bank.query_residuals.zero_()
    output = model.forward_components(torch.randn(1, 3, 8, 8), include_anchor=True)
    assert torch.equal(output.logits, output.anchor_logits)
    assert torch.count_nonzero(model.get_residual_contribution(output=output)) == 0


def test_zero_vectors_are_not_the_branch_removal_reference():
    model = make_model(2)
    with torch.no_grad():
        model.query_head.alpha.fill_(.3)
        model.query_bank.class_anchors.zero_()
    output = model.forward_components(torch.randn(1, 3, 8, 8), include_anchor=True)
    anchor = model.get_anchor_query_effect(output=output)
    torch.testing.assert_close(anchor, torch.full_like(anchor, .3 * math.log(2)), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(anchor) > 0
    assert torch.equal(output.counterfactual_logits(0)[:, 0], output.base_logits[:, 0])


def test_alpha_zero_all_effects_zero():
    model = make_model()
    output = model.forward_components(torch.randn(1, 3, 8, 8), include_anchor=True)
    assert torch.equal(output.logits, output.base_logits)
    assert torch.equal(output.anchor_logits, output.base_logits)
    assert torch.count_nonzero(model.get_anchor_query_effect(output=output)) == 0
    assert torch.count_nonzero(model.get_residual_contribution(output=output)) == 0


def test_opt_in_preserves_default_factual_output_rng_and_state_keys():
    model = make_model().train()
    image = torch.randn(1, 3, 8, 8)
    before = torch.get_rng_state()
    first = model.forward_components(image)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    second = model.forward_components(image, include_anchor=True)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(torch.get_rng_state(), after)
    assert first.anchor_logits is None
    with pytest.raises(RuntimeError, match="not requested"):
        model.get_anchor_query_effect(output=first)
    with pytest.raises(ValueError, match="exactly one"):
        model.get_residual_contribution()


@pytest.mark.parametrize("mode", ["legacy", "one_way", "bidirectional", "null"])
def test_anchor_scope_rejected_before_backbone(mode):
    model = make_model(centered=mode != "legacy", interaction=mode if mode in ("one_way", "bidirectional") else "static",
                       learned_null=mode == "null")
    with pytest.raises(ValueError, match="Anchor-only scope"):
        model.forward_components(torch.randn(1, 3, 8, 8), include_anchor=True)
    assert model.backbone.calls == 0
