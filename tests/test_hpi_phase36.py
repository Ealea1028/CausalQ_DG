from types import SimpleNamespace
import json

import pytest
import torch

from causalq.datasets.segmentation import PairedSegmentationDataset
from causalq.datasets.strict_source import SourceEvalTransform
from causalq.losses.query_effect_invariance import (
    effect_invariance_loss, gt_region_prediction_effect, positive_contribution_loss,
)
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.runtime.cohort_runner import CohortRunner, IndependentHead, capture_rng, commit_head, head_state, restore_head
from causalq.runtime.hpi_phase35 import build_ablation_head
from causalq.runtime.hpi_phase36 import effect_backward, paired_class_forward, sample_present_class
from causalq.runtime.hpi_training import source_diagnostics, verify_freeze
from causalq.runtime.identity import tree_digest
from test_hpi_backbone import tiny_backbone
from test_hpi_runtime import samples


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


@pytest.mark.parametrize(('effects', 'expected', 'gradient'), [
    ([-.05, .01], .04, [-.5, -.5]),
    ([.02, .02], 0., [0., 0.]),
    ([.025, .035], 0., [0., 0.]),
])
def test_positive_contribution_uses_signed_effect_and_stops_at_margin(effects, expected, gradient):
    effects = torch.tensor(effects, requires_grad=True)
    loss = positive_contribution_loss(effects)
    assert loss.item() == pytest.approx(expected, abs=1e-8)
    loss.backward()
    assert effects.grad.tolist() == gradient


def test_phase36_real_heads_resume_independent_cursors_and_report_source(samples, tmp_path):
    freeze = verify_freeze(36)
    assert freeze['variants'] == ['B4', 'B5'] and freeze['total_optimizer_updates'] == 80000
    backbone = tiny_backbone()
    backbone_digest = tree_digest(backbone.state_dict())

    def make(name):
        model = build_ablation_head(backbone, 'B3', None, decoder_channels=32, num_classes=3)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        return IndependentHead(name, model, optimizer, capture_rng())

    baseline = [make(name) for name in freeze['variants']]
    start = [make(name) for name in freeze['variants']]
    assert len({tree_digest(head_state(h.model)) for h in baseline + start}) == 1
    inputs = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        original, photo = [backbone(view) for view in inputs.split(1)]
    label = torch.arange(32 * 32).reshape(32, 32) % 3
    root = tmp_path/'checkpoints'
    identity = {'git_sha': 'phase36-test'}

    def update(runner, tick):
        sample = {'label': label, 'metadata': {'step_zero_based': tick}}
        return runner.update(sample, original, photo, update_fn=effect_backward)['heads']

    reference = CohortRunner(baseline, horizon=3)
    for tick in range(3):
        record = update(reference, tick)
        assert record['B4']['losses'][-1] == 0
        assert record['B5']['losses'][-1] >= 0
    partial = CohortRunner(start, horizon=3)
    update(partial, 0)
    commit_head(start[0], root, identity)
    update(partial, 1)
    commit_head(start[1], root, identity)
    resumed = [make(name) for name in freeze['variants']]
    assert [restore_head(h, root, identity) for h in resumed] == [1, 2]
    recovered = CohortRunner(resumed, horizon=3)
    assert set(update(recovered, 1)) == {'B4'}
    assert set(update(recovered, 2)) == {'B4', 'B5'}
    for expected, actual in zip(baseline, resumed, strict=True):
        assert tree_digest(head_state(expected.model)) == tree_digest(head_state(actual.model))
        assert tree_digest(expected.optimizer.state_dict()) == tree_digest(actual.optimizer.state_dict())
        assert tree_digest(expected.rng) == tree_digest(actual.rng)
    val = PairedSegmentationDataset(samples, transform=SourceEvalTransform(long_edge=32))
    source = tmp_path/'source'
    source_diagnostics(resumed, backbone, val, source, identity, device='cpu', amp=False,
                       profile=load_profile(phase=36), intervention_ids=['00000'])
    for name in freeze['variants']:
        assert json.loads((source/name/'source_metrics.json').read_text())['sample_count'] == 4
        effect = json.loads((source/name/'main_path128.json').read_text())
        assert set(effect['by_image']) == {'00000'}
        macro = effect['image_macro']
        assert macro['population'] == 'equal_weight_images_over_gt_present_classes'
        assert macro['tau'] == pytest.approx([a-b for a,b in zip(macro['on'],macro['off'])], abs=1e-7)
        assert sum(macro[key] for key in ('positive_fraction','negative_fraction','near_zero_fraction')) == pytest.approx(1.)
    assert tree_digest(backbone.state_dict()) == backbone_digest
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_phase36_coordinator_dispatches_only_to_phase36(tmp_path, monkeypatch):
    import sys
    from tools import run_hpi_dinov2_20261031 as coordinator

    sha = 'a' * 40
    monkeypatch.setenv('CAUSALQ_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['coordinator', sha, 'train-b4-b5'])
    monkeypatch.setattr(coordinator, 'verify_checkout', lambda _: None)
    launches = []
    monkeypatch.setattr(coordinator, 'launch', lambda *args: launches.append(args))
    assert coordinator.main() == 0
    assert launches == [(tmp_path/'phase36/B4_B5_aaaaaaa', sha, 'train-b4-b5')]
    assert coordinator.FORMAL_UPDATES['train-b4-b5'] == 80000
