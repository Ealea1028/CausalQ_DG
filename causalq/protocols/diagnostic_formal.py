"""Phase29 formal release; preserve the immutable Phase28 diagnostic recipes."""

from copy import deepcopy
import hashlib
import json

from causalq.protocols.diagnostic_20261020 import (
    STUDY, ROOT, NEW_UNCENTERED, MANIFEST_HASHES, baseline_source_report, diagnostic_config, mechanism,
)


def training_config(variant):
    if variant not in NEW_UNCENTERED:
        raise ValueError('Only Phase29 UP/UF/UA/UR training is released')
    config = diagnostic_config(variant)
    config['experiment'] = {'id': f'{STUDY}_{variant}_FORMAL_SEED0', 'phase': 29,
                            'status': 'formal_training_authorized'}
    config['protocol']['training_authorized'] = True
    return config


def validate_training_config(config, variant):
    if config != training_config(variant):
        raise ValueError('Phase29 formal recipe drift')


def matched_baseline():
    report = baseline_source_report('U0')
    return {'environment': deepcopy(report['metadata']['environment']),
            'initial_head_sha256': report['initial_head_sha256'],
            'producer_git_sha': report['metadata']['git_sha'],
            'baseline_variant': 'U0=M0'}


def training_policy(variant):
    mode = mechanism(variant)
    return {'study': STUDY, 'target_feedback_already_seen': True,
            'target_data_accessed': False, 'new_loss_enabled': True,
            'consistency_mechanism': mode, 'consistency_weight': 1.,
            'reference_stop_gradient': True, 'component_protocol': STUDY,
            'raw_anchor_ranking_required': False,
            'style_residual_anchor_detached': False,
            'residual_definition': 'full_effect_minus_repeated_anchor_effect',
            'new_pilot_required': False, 'initialization': 'fresh_seed0_matched_U0',
            'future_centered_controls': ['CA', 'CR'],
            'target_evaluation_authorized_this_phase': False}


def release_contract():
    return {'schema_version': 1, 'phase': 29, 'study': STUDY,
            'order': list(NEW_UNCENTERED), 'seed': 0, 'updates_per_variant': 40000,
            'total_optimizer_updates': 160000, 'checkpoint_interval': 1000,
            'checkpoint_selection': 'fixed_final', 'centered_residuals': False,
            'training_configs_sha256': {v: hashlib.sha256((ROOT /
                f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml').read_bytes()).hexdigest()
                for v in NEW_UNCENTERED},
            'matched_baseline': matched_baseline(), 'source_manifests_sha256': MANIFEST_HASHES,
            'sampler': 'torch.randperm(N, CPU Generator.manual_seed(seed+epoch)); workers0',
            'training_views': ['original', 'photometric'], 'ce_weights': [1., 1.],
            'consistency_weights': {v: 1. for v in NEW_UNCENTERED},
            'source_validation': 'final_only_all2497_longedge1024_aspect_pad16',
            'residual_style_anchor_detached': False, 'reference_stop_gradient': True,
            'raw_anchor_ranking_required': False, 'additional_pilot_required': False,
            'target_feedback_already_seen': True, 'target_access_authorized': False,
            'future_centered_controls': ['CA', 'CR'], 'future_centered_updates': 80000,
            'future_diagnostics_phase': 31, 'future_target_freeze_phase': 32}


def verify_release():
    path = ROOT / 'experiments/DIAGNOSTIC_20261020/phase29_freeze.json'
    if json.loads(path.read_bytes()) != release_contract():
        raise ValueError('Phase29 frozen initialization/config/sampling contract differs')
    return hashlib.sha256(path.read_bytes()).hexdigest()
