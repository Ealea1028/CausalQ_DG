"""Phase32 final reporting freeze; six new targets, four old results reused."""

import hashlib
import json

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, BASELINES, baseline_records, mechanism
from causalq.protocols.diagnostic_final_source import (
    ACCEPTANCE, NEW_FINALS, final_records, final_config,
)

TARGET_MANIFEST_SHA = '3960bb34afdfbbc6a5a5be887626d7ab03196691de54d0c00eb047153e03ab4c'
FREEZE_PATH = ACCEPTANCE / 'phase32_target_freeze.json'
GEOMETRY = 'original1024x2048; no resize,crop,flip,TTA or sliding; whole-image BF16 inference'
COMPARISONS = (('UA', 'U0'), ('UR', 'U0'), ('UA', 'UR'), ('UA', 'UF'), ('UA', 'UP'), ('CA', 'C0'), ('CR', 'C0'))


def target_contract():
    if hashlib.sha256((ROOT / 'manifests/cityscapes_val500.txt').read_bytes()).hexdigest() != TARGET_MANIFEST_SHA:
        raise ValueError('Frozen target500 manifest differs')
    accepted = json.loads((ACCEPTANCE / 'phase31_acceptance.json').read_bytes())
    if accepted['accepted'] is not True or set(accepted['reports']) != set(NEW_FINALS):
        raise ValueError('Need completed Phase31 source diagnostics')
    old = json.loads((ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes())
    records = final_records()
    for v, r in records.items():
        r.update(config_path=f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml',
                 checkpoint_iteration=40000, centered_residuals=final_config(v)['query']['centered_residuals'],
                 consistency_mechanism=mechanism(v))
    reused = baseline_records()
    for v, r in reused.items():
        r.update(old_variant=BASELINES[v], target_miou=old['target_miou'][BASELINES[v]],
                 target_report_sha256=hashlib.sha256((ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes()).hexdigest(),
                 evaluate_again=False)
    return {'schema_version': 1, 'phase': 32, 'study': STUDY, 'status': 'frozen_before_new_target_access',
            'target_feedback_already_seen': True, 'interpretation': 'exploratory single-seed comparison after old target feedback',
            'new_target_order': list(NEW_FINALS), 'new_models': records, 'reused_models': reused,
            'target_manifest_sha256': TARGET_MANIFEST_SHA, 'target_samples_per_model': 500,
            'total_new_target_images': 3000, 'target_geometry': GEOMETRY,
            'target_label_space': 'Cityscapes19 trainIds,ignore255', 'seed': 0,
            'selection': 'fixed_final_iteration40000_for_all; report all ten; no winner selection',
            'optimizer_updates_this_phase': 0, 'target_labels_optimized': False,
            'target_results_may_change_recipe': False, 'additional_training_authorized': False,
            'additional_seeds_authorized': False, 'additional_pilot_required': False,
            'primary_comparisons': [list(pair) for pair in COMPARISONS],
            'interaction_comparisons': ['(UA-U0)-(CA-C0)', '(UR-U0)-(CR-C0)'],
            'acceptance_sha256': {str(p): hashlib.sha256((ACCEPTANCE / f'phase{p}_acceptance.json').read_bytes()).hexdigest()
                                  for p in (28, 29, 30, 31)},
            'inference': 'model.eval; torch.inference_mode; whole-image BF16; argmax native logits; no optimizer',
            'uncertainty': 'one trained seed; point estimates only; image bootstrap cannot estimate training-seed uncertainty'}


def verify_target_contract():
    if json.loads(FREEZE_PATH.read_bytes()) != target_contract():
        raise ValueError('Phase32 final target freeze differs')
    return hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest()
