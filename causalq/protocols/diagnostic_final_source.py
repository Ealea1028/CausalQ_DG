"""Phase31: diagnose the six completed finals, reuse Phase28 baselines."""

import hashlib
import json
from pathlib import PurePosixPath

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, contract, NEW_UNCENTERED
from causalq.protocols.diagnostic_centered_formal import NEW_CENTERED, centered_training_config
from causalq.protocols.diagnostic_formal import training_config

NEW_FINALS = (*NEW_UNCENTERED, *NEW_CENTERED)
ACCEPTANCE = ROOT / 'experiments/DIAGNOSTIC_20261020'


def final_source_report(variant):
    if variant not in NEW_FINALS:
        raise ValueError('Phase31 only diagnoses six new finals; reuse old diagnostics')
    phase = 30 if variant in NEW_CENTERED else 29
    accepted = json.loads((ACCEPTANCE / f'phase{phase}_acceptance.json').read_bytes())
    if accepted['accepted'] is not True:
        raise ValueError('Need accepted completed training')
    return accepted['formal_report']['reports'][variant]


def final_config(variant):
    final_source_report(variant)  # Reject baseline/unknown names explicitly.
    return centered_training_config(variant) if variant in NEW_CENTERED else training_config(variant)


def final_records():
    records = {}
    for v in NEW_FINALS:
        report = final_source_report(v)
        config_path = ROOT / f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml'
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != report['metadata']['config_sha256']:
            raise ValueError('Final training YAML differs from accepted report')
        phase = report['phase'];producer = report['metadata']['git_sha']
        directory = f'phase{phase}/' + ('CENTERED' if phase == 30 else 'UNCENTERED') + f'_SEED0_{producer[:7]}'
        relative = PurePosixPath(report['checkpoint'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe final checkpoint reference')
        records[v] = {'producer_git_sha': producer,
                      'checkpoint_relative_to_output_root': f'{STUDY}/{directory}/{v}/{relative}',
                      'checkpoint_sha256': report['checkpoint_sha256'], 'head_sha256': report['final_head_sha256'],
                      'config_sha256': report['metadata']['config_sha256'], 'source_miou': report['source_miou']}
    return records


def final_contract():
    old = contract()
    return {'schema_version': 1, 'study': STUDY, 'phase': 31, 'order': list(NEW_FINALS),
            'source_samples_per_variant': 2497, 'views': old['diagnostic_views'],
            'optimizer_updates_this_phase': 0, 'training_authorized': False,
            'target_data_access_authorized': False, 'target_feedback_already_seen': True,
            'new_training_gate': False, 'checkpoint_training_updates': 40000,
            'effect_geometry': old['effect_geometry'], 'gradient_geometry': old['gradient_geometry'],
            'rms_epsilon': old['rms_epsilon'], 'gradient_epsilon': old['gradient_epsilon'],
            'correlation_fisher_clip': old['correlation_fisher_clip'],
            'sampling': 'same source IDs, style_seeds and gradient-crop seeds as Phase28',
            'raw_variance': 'population five-view; pair_raw_variance is separate original/photo1 compatibility',
            'normalized_variance': 'per-image/class/view RMS normalization; not raw variance divided by macro RMS squared',
            'baseline_diagnostics_reused': {'producer_git_sha': '1086bc36c2967bc9ff0f7480faad929b210d393f',
                                          'variants': ['U0', 'C0', 'CP', 'CF'], 'repeat_inference': False},
            'intervention_examples': {'reuse_phase28': True, 'ids': old['preview_ids'], 'export_again': False},
            'acceptance_sha256': {str(p): hashlib.sha256((ACCEPTANCE / f'phase{p}_acceptance.json').read_bytes()).hexdigest()
                                  for p in (28, 29, 30)},
            'final_checkpoints': final_records(),
            'next_phase': '32: local final target freeze, then all six new target evaluations; no tuning'}


def verify_final_contract():
    path = ACCEPTANCE / 'phase31_freeze.json'
    if json.loads(path.read_bytes()) != final_contract():
        raise ValueError('Phase31 frozen source diagnostic contract differs')
    return hashlib.sha256(path.read_bytes()).hexdigest()
