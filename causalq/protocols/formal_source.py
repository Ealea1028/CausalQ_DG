"""Frozen formal source recipes: M0/M1 and prediction/Full-CQE controls."""

from copy import deepcopy
from pathlib import Path

import yaml

from causalq.protocols.strict_dg import validate_strict_config

ROOT = Path(__file__).resolve().parents[2]


def formal_config(variant):
    if variant == 'M3':
        config = formal_config('M2')
        config['experiment'] = {'id': 'M3_FORMAL_SEED0', 'phase': 25, 'status': 'conditional_on_trained_m1_stability'}
        config['causal_query_effect']['component'] = 'anchor_only'
        return config
    if variant == 'M2':
        config = formal_config('M1')
        config['experiment'] = {'id': 'M2_FORMAL_SEED0', 'phase': 24, 'status': 'formal_training_authorized'}
        config['causal_query_effect'] = {'enabled': True, 'lambda_cqe': 1.0, 'effect_space': 'logits',
            'reference_stop_gradient': True, 'normalization': 'l2_per_class_map', 'present_classes_only': True,
            'valid_pixels_only': True, 'loss': 'smooth_l1', 'smooth_l1_beta': 1.0}
        return config
    if variant == 'P1':
        config = formal_config('M1')
        config['experiment'] = {'id': 'P1_FORMAL_SEED0', 'phase': 23, 'status': 'formal_training_authorized'}
        config['prediction_consistency'] = {'enabled': True, 'lambda_pred': 1.0, 'temperature': 1.0,
            'divergence': 'kl_reference_to_view', 'reference_stop_gradient': True, 'valid_pixels_only': True}
        return config
    if variant not in ('M0', 'M1'):
        raise ValueError('Formal variant must be M0, M1, P1, M2 or M3')
    baseline = yaml.safe_load((ROOT / 'configs/strict_dg/gta_dinov3l_m0.yaml').read_text())
    validate_strict_config(baseline)
    config = deepcopy(baseline)
    config['experiment'] = {'id': variant + '_FORMAL_SEED0', 'phase': 22, 'status': 'formal_training_authorized'}
    config['protocol']['training_authorized'] = True
    config['data']['num_workers'] = 0
    config['validation']['interval'] = 40000
    config['query']['centered_residuals'] = variant == 'M1'
    return config


def validate_formal_config(config, variant):
    if config != formal_config(variant):
        raise ValueError('Formal recipe drift from frozen variant contract')
