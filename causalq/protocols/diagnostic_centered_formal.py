"""Phase30 centered matching controls; the Phase29 release stays immutable."""

import hashlib
import json

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, diagnostic_config, baseline_source_report
from causalq.protocols.diagnostic_formal import release_contract, training_policy

NEW_CENTERED = ('CA', 'CR')


def centered_training_config(variant):
    if variant not in NEW_CENTERED:
        raise ValueError('Only Phase30 CA/CR centered controls are released')
    config = diagnostic_config(variant)
    config['experiment'] = {'id': f'{STUDY}_{variant}_FORMAL_SEED0', 'phase': 30,
                            'status': 'formal_training_authorized'}
    config['protocol']['training_authorized'] = True
    return config


def validate_centered_training_config(config, variant):
    if config != centered_training_config(variant):
        raise ValueError('Phase30 formal recipe drift')


def centered_baseline():
    report = baseline_source_report('C0')
    return {'environment': report['metadata']['environment'],
            'initial_head_sha256': report['initial_head_sha256'],
            'producer_git_sha': report['metadata']['git_sha'], 'baseline_variant': 'C0=M1'}


def centered_training_policy(variant):
    if variant not in NEW_CENTERED:
        raise ValueError('Only Phase30 CA/CR centered controls are released')
    policy = training_policy(variant)
    policy.pop('future_centered_controls')
    return {**policy, 'initialization': 'fresh_seed0_matched_C0', 'next_phase': 31}


def centered_release_contract():
    # Reuse the frozen schedule, sampler, data, geometry and loss policy;
    # the only parameterization change relative to UA/UR is centering.
    frozen = release_contract()
    for key in ('training_configs_sha256', 'future_centered_controls', 'future_centered_updates'):
        frozen.pop(key)
    return {**frozen, 'phase': 30, 'order': list(NEW_CENTERED), 'total_optimizer_updates': 80000,
            'centered_residuals': True, 'matched_baseline': centered_baseline(),
            'training_configs_sha256': {v: hashlib.sha256((ROOT /
                f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml').read_bytes()).hexdigest()
                for v in NEW_CENTERED},
            'consistency_weights': {v: 1. for v in NEW_CENTERED},
            'matching_uncentered_variants': {'CA': 'UA', 'CR': 'UR'},
            'budget_authority': 'original six-run plan; CA/CR budget retained before Phase29 results',
            'next_phase': 31}


def verify_centered_release():
    path = ROOT / 'experiments/DIAGNOSTIC_20261020/phase30_freeze.json'
    if json.loads(path.read_bytes()) != centered_release_contract():
        raise ValueError('Phase30 frozen initialization/config/sampling contract differs')
    return hashlib.sha256(path.read_bytes()).hexdigest()
