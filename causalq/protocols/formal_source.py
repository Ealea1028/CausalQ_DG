"""Phase22 matched no-CQE formal baseline contract, separate from old gates."""

from copy import deepcopy
from pathlib import Path

import yaml

from causalq.protocols.strict_dg import validate_strict_config

ROOT = Path(__file__).resolve().parents[2]


def formal_config(variant):
    if variant not in ('M0', 'M1'):
        raise ValueError('Formal variant must be M0 or M1')
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
        raise ValueError('Formal recipe drift: only M1 residual centering may differ')
