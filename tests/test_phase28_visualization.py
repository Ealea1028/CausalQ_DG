"""Check aggregation identity, missing-value handling and responsibility scope."""

import math

import numpy as np
import pytest

from tools.plot_phase28_diagnostics import VARIANTS, json_safe, normalized_ratio_grid, ratio, row, value, weighted_usage


def fixture():
    rows = []
    for c in (-1, 0, 1):
        for k, n in [('anchor', 2.), ('residual', 8.)]:
            rows.append({'kind': 'effects', 'population': 'all_classes', 'class_id': c,
                         'component': k, 'metrics': {'rms_normalized_variance': {'mean': n, 'count': 2497}}})
    return {'image_macro_statistics': rows, 'pixel_weighted_responsibility': [
        {'class_id': 0, 'scope': 'all_valid', 'view_index': 0, 'pixels': 100, 'mean_responsibility': [.4, .6]},
        {'class_id': 0, 'scope': 'gt_class', 'view_index': 0, 'pixels': 10, 'mean_responsibility': [.9, .1]}]}


def test_plot_selector_keeps_macro_separate_from_class_rows():
    report = fixture()
    assert value(report, 'effects', 'rms_normalized_variance', component='anchor') == 2
    assert row(report, 'effects', component='anchor')['rms_normalized_variance']['count'] == 2497
    report['image_macro_statistics'].append(report['image_macro_statistics'][0])
    with pytest.raises(ValueError, match='Ambiguous'):row(report, 'effects', component='anchor')


def test_normalized_ratio_heatmap_is_log_of_ratio_and_preserves_undefined():
    reports = {v: fixture() for v in VARIANTS}
    grid = normalized_ratio_grid(reports, 'all_classes', classes=(0, 1))
    np.testing.assert_allclose(grid, math.log10(.25))
    reports['C0']['image_macro_statistics'][-1]['metrics']['rms_normalized_variance']['mean'] = None
    assert np.isnan(normalized_ratio_grid(reports, 'all_classes', classes=(0, 1))[1, 1])
    assert ratio(1, 0) is None and ratio(float('nan'), 1) is None
    assert json_safe({'x': [float('nan'), .25]}) == {'x': [None, .25]}


def test_responsibility_heatmap_uses_explicit_pixel_weighted_scope():
    report = fixture()
    assert weighted_usage(report, scope='gt_class', classes=(0,)).tolist() == [.9]
    assert weighted_usage(report, scope='all_valid', classes=(0,)).tolist() == [.4]
    report['pixel_weighted_responsibility'][1]['pixels'] = 0
    assert np.isnan(weighted_usage(report, scope='gt_class', classes=(0,))[0])
