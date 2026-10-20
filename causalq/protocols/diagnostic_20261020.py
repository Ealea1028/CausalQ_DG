"""Source-only selective-stability study; legacy protocols remain immutable."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from causalq.analysis.query_components import intervention_seed
from causalq.datasets.strict_source import load_membership, MANIFEST_HASHES
from causalq.protocols.formal_source import formal_config

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'diagnostic_20261020'
BASELINES = {'U0': 'M0', 'C0': 'M1', 'CP': 'P1', 'CF': 'M2'}
VARIANTS = ('U0', 'UP', 'UF', 'UA', 'UR', 'C0', 'CP', 'CF', 'CA', 'CR')
NEW_UNCENTERED = ('UP', 'UF', 'UA', 'UR')
FREEZE_SHA = '959b031dc5112ee021b1c14dfd8af15d80c998cf7ff510d0065893f69fda95d5'
RMS_EPS = 1e-6
GRAD_EPS = 1e-12


def stable_seed(identity, kind):
    return int(hashlib.sha256(f'{STUDY}:{kind}:{identity}'.encode()).hexdigest()[:8], 16)


def style_seeds(identity):
    return [intervention_seed(identity)] + [stable_seed(identity, f'photo-{i}') for i in (2, 3, 4)]


def preview_ids():
    _, ids = load_membership()
    return random.Random(20261020).sample(sorted(ids), 3)


def mechanism(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown selective-stability variant')
    return {'0': 'none', 'P': 'prediction', 'F': 'full', 'A': 'anchor', 'R': 'residual'}[variant[1]]


def diagnostic_config(variant):
    mode = mechanism(variant)
    config = deepcopy(formal_config({'none': 'M0', 'prediction': 'P1'}.get(mode, 'M2')))
    config['query']['centered_residuals'] = variant.startswith('C')
    config['experiment'] = {'id': f'{STUDY}_{variant}_SEED0', 'phase': 28, 'status': 'planned_not_released_for_training'}
    config['protocol'].update(training_authorized=False, study=STUDY, target_feedback_already_seen=True,
                              raw_anchor_ranking_required=False)
    if mode in ('full', 'anchor', 'residual'):
        config['causal_query_effect']['component'] = mode
    return config


def baseline_records():
    path = ROOT / 'experiments/STRICT_DG_PHASE26/final_evaluation_freeze.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FREEZE_SHA:
        raise ValueError('Original final checkpoint freeze differs')
    freeze = json.loads(path.read_bytes())
    return {name: deepcopy(freeze['variants'][old]) for name, old in BASELINES.items()}


def baseline_source_report(variant):
    old = BASELINES[variant]
    path = (ROOT / f'experiments/STRICT_DG_PHASE22/{old.lower()}_formal_report.json' if old in ('M0', 'M1')
            else ROOT / f'experiments/STRICT_DG_PHASE{23 if old == "P1" else 24}/formal_report.json')
    return json.loads(path.read_bytes())


def contract():
    return {'schema_version': 1, 'study': STUDY, 'active_phase': 28,
            'target_feedback_already_seen': True, 'target_data_access_authorized_this_phase': False,
            'optimizer_updates_this_phase': 0, 'old_anchor_ranking_is_new_training_gate': False,
            'variant_mechanisms': {v: mechanism(v) for v in VARIANTS},
            'new_training_order': [*NEW_UNCENTERED, 'CA', 'CR'],
            'new_formal_updates_planned': 240000, 'training_released_this_phase': False,
            'manifest_sha256': MANIFEST_HASHES, 'source_samples': 2497,
            'diagnostic_views': ['original', 'photometric_01', 'photometric_02', 'photometric_03', 'photometric_04'],
            'effect_geometry': 'source_longedge1024_aspect_pad16; exclude ignore padding',
            'gradient_geometry': 'deterministic_512_crop; TrainTransform original defaults',
            'preview_ids': preview_ids(), 'preview_selection_seed': 20261020,
            'rms_epsilon': RMS_EPS, 'gradient_epsilon': GRAD_EPS,
            'correlation_fisher_clip': 1e-7, 'baseline_checkpoints': baseline_records(),
            'm4_or_additional_mechanisms_authorized': False}


def contract_sha():
    return hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()
