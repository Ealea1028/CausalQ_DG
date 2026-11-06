from types import SimpleNamespace

import pytest
import torch

from causalq.losses.query_effect_invariance import effect_invariance_loss, gt_region_prediction_effect
from causalq.runtime.hpi_phase36 import paired_class_forward, sample_present_class


def test_prediction_effect_masks_and_only_differentiates_on():
    label = torch.tensor([[[0, 255, 1]]])
    original = torch.tensor([[[[1., 5., 5.]], [[0., 0., 0.]]]], requires_grad=True)
    photo = torch.zeros_like(original, requires_grad=True)
    off = torch.zeros_like(original, requires_grad=True)
    category = torch.tensor([0])
    effects = torch.stack([gt_region_prediction_effect(logits, off, label, category)
                           for logits in (original, photo)])
    expected = torch.sigmoid(torch.tensor(1.)).item() - .5
    assert effects.tolist() == pytest.approx([expected, 0.])
    loss = effect_invariance_loss(effects)
    assert loss.item() == pytest.approx(expected**2 / 4)
    loss.backward()
    assert original.grad[..., 0].abs().sum() > 0
    assert photo.grad[..., 0].abs().sum() > 0
    assert torch.count_nonzero(original.grad[..., 1:]) == 0
    assert torch.count_nonzero(photo.grad[..., 1:]) == 0
    assert off.grad is None


def test_class_rng_is_independent_replayable_and_gt_present():
    label = torch.tensor([[[4, 13, 255]]])
    before = torch.get_rng_state()
    sampled = [int(sample_present_class(label, step)) for step in range(30)]
    assert set(sampled) == {4, 13}
    assert sampled == [int(sample_present_class(label, step)) for step in range(30)]
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(ValueError, match='GT-present'):
        sample_present_class(torch.full_like(label, 255), 0)


def test_on_off_share_dropout_and_advance_rng_only_once():
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128))
            self.dropout = torch.nn.Dropout(.4)

        def forward_routed(self, features, size):
            return SimpleNamespace(logits=self.dropout(features * self.weight)), None

        def closed_class_logits(self, features, size, category):
            return self.dropout(features * self.weight)

    model = Probe()
    features = torch.ones(128)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        before = torch.get_rng_state()
        model.forward_routed(features, None)
        expected = torch.get_rng_state()
        torch.set_rng_state(before)
        on, _, off = paired_class_forward(model, features, None, torch.tensor([0]), amp=False)
        assert torch.equal(on.logits, off)
        assert on.logits.requires_grad and not off.requires_grad
        assert torch.equal(torch.get_rng_state(), expected)
