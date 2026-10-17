import math

import pytest
import torch

from causalq.analysis.query_components import (fixed_source_ids, intervention_seed, query_parameter_rows,
                                               responsibility_rows, effect_rows)


def test_fixed_selection_is_order_independent_and_does_not_touch_rng():
    ids = [f"{i:05}" for i in range(40)]
    rng = torch.get_rng_state().clone()
    chosen = fixed_source_ids(ids)
    assert chosen == fixed_source_ids(list(reversed(ids))) and len(set(chosen)) == 20
    assert intervention_seed(chosen[0]) == intervention_seed(chosen[0])
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError): fixed_source_ids(ids + ids)
    with pytest.raises(ValueError): fixed_source_ids(ids, 41)


def test_responsibility_has_known_entropy_activation_deletion_and_paired_variance():
    scores = torch.tensor([math.log(3), 0.]).reshape(1, 1, 2, 1, 1)
    rows = responsibility_rows(scores, scores.flip(2), torch.zeros(1, 2, 2, dtype=torch.long))
    r0, r1 = rows[:2]
    assert r0['mean_responsibility'] == pytest.approx(.75)
    assert r1['mean_responsibility'] == pytest.approx(.25)
    assert r0['mean_pixel_entropy_nats'] == pytest.approx(-.75 * math.log(.75) - .25 * math.log(.25))
    assert r0['above_uniform_activation_rate'] == 1 and r1['above_uniform_activation_rate'] == 0
    assert r0['top1_share_ties_split'] == 1 and r1['top1_share_ties_split'] == 0
    assert r0['cross_view_population_variance'] == pytest.approx(.0625)
    assert r0['slot_logsumexp_deletion_drop'] == pytest.approx(math.log(4))
    assert r1['slot_logsumexp_deletion_drop'] == pytest.approx(math.log(4 / 3))


@pytest.mark.parametrize('slots', [1, 2, 3])
def test_uniform_responsibilities_ties_and_single_slot(slots):
    scores = torch.zeros(2, 2, slots, 2, 2)
    rows = responsibility_rows(scores, scores, torch.zeros(2, 4, 4, dtype=torch.long))
    defined = [r for r in rows if r['class_id'] == 0]
    assert all(r['mean_responsibility'] == pytest.approx(1 / slots) for r in defined)
    assert all(r['mean_pixel_entropy_nats'] == pytest.approx(math.log(slots)) for r in defined)
    assert all(r['top1_share_ties_split'] == pytest.approx(1 / slots) for r in defined)
    assert all(r['above_uniform_activation_rate'] == 0 for r in defined)
    assert all(r['cross_view_population_variance'] == 0 for r in defined)
    if slots == 1: assert all(r['slot_logsumexp_deletion_drop'] is None for r in defined)
    assert all(r['mean_responsibility'] is None for r in rows if r['class_id'] == 1)


def test_absent_and_ignore_regions_are_missing_not_zero():
    scores = torch.ones(1, 2, 2, 2, 2)
    ignored = torch.full((1, 2, 2), 255, dtype=torch.long)
    assert all(r['mean_responsibility'] is None for r in responsibility_rows(scores, scores, ignored))
    maps = {k: torch.ones(1, 2, 2, 2) for k in ('full', 'anchor', 'residual')}
    assert all(r['original_mean_abs'] is None for r in effect_rows(maps, maps, ignored))


def test_effect_variance_energy_normalization_and_ignore_exclusion():
    x = torch.full((1, 2, 2, 2), 2.)
    y = x * 2
    x[..., 1, 1] = 100
    y[..., 1, 1] = -100
    label = torch.tensor([[[0, 1], [0, 255]]])
    maps = lambda v: {k: v for k in ('full', 'anchor', 'residual')}
    rows = effect_rows(maps(x), maps(y), label)
    assert len(rows) == 6
    row = rows[0]
    assert row['valid_pixels'] == 3 and row['inside_pixels'] == 2 and row['outside_pixels'] == 1
    assert row['original_mean_abs'] == 2 and row['photometric_mean_abs'] == 4
    assert row['cross_view_population_variance'] == 1
    assert row['rms_normalized_pair_mse'] == 0
    assert row['original_inside_outside_ratio'] == pytest.approx(1)
    zero = maps(torch.zeros_like(x))
    assert all(r['rms_normalized_pair_mse'] is None for r in effect_rows(zero, zero, label))


def test_centered_parameter_geometry_and_r1_undefined_pairs():
    anchors = torch.tensor([[1., 0.]])
    residual = torch.tensor([[[0., 1.], [0., -1.]]])
    row = query_parameter_rows(anchors, residual)[0]
    assert row['query_pair_cosine'] == pytest.approx(0)
    assert row['anchor_residual_norm_ratio'] == 1
    row = query_parameter_rows(anchors, torch.zeros(1, 1, 2))[0]
    assert row['query_pair_cosine'] is None and row['anchor_residual_norm_ratio'] is None


def test_reject_bad_scores_labels_and_effect_shapes():
    scores = torch.zeros(1, 2, 2, 2, 2)
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError): responsibility_rows(scores, scores[..., :1], labels)
    with pytest.raises(ValueError): responsibility_rows(scores + float('nan'), scores, labels)
    with pytest.raises(ValueError): responsibility_rows(scores, scores, labels + 99)
    maps = {k: torch.zeros(1, 2, 2, 2) for k in ('full', 'anchor', 'residual')}
    with pytest.raises(ValueError): effect_rows(maps, maps, labels[:, :1])
    with pytest.raises(ValueError): query_parameter_rows(torch.zeros(1, 2), torch.zeros(2, 2, 2))
