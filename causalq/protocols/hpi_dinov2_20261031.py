"""Independent Phase33 freeze; no import of old acceptance/environment files."""
from pathlib import Path
import yaml
from causalq.models.query_segmentor import QuerySegmentor
from causalq.runtime.identity import tree_digest
import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'hpi_dinov2_20261031'
HORIZON = 40000
HISTORICAL_HEAD_SHA = 'e15b11e3b4d645fc17d936576a5d1fee49b418ce30d5ef265d899788f72d6c7a'
RELEASED_ACTIONS = ('prepare', 'train-baselines', 'performance-check', 'status', 'package')


def load_profile(path=None):
    controlled = ROOT / 'configs' / STUDY / 'runtime_high_throughput.yaml'
    if path is not None and Path(path).resolve() != controlled.resolve():
        raise ValueError('Runtime profile must be the controlled file in this checkout')
    value = yaml.safe_load(controlled.read_text())
    if value['logical_batch_size'] != 1 or value['optimizer_updates_per_variant'] != HORIZON:
        raise ValueError('Scientific batch/budget changed')
    return value


def load_config(variant):
    if variant not in ('U0', 'UA'):
        raise ValueError('Phase33 implements only the U0/UA recipe mapping')
    config = yaml.safe_load((ROOT / 'configs' / STUDY / f'{variant.lower()}_seed0.yaml').read_text())
    if config['protocol']['study'] != STUDY or config['protocol']['training_authorized']:
        raise ValueError('Phase33 must not authorize formal updates')
    return config


def build_head(backbone, *, seed=0, decoder_channels=256, num_classes=19, dropout=.1):
    # Backbone construction/load must never consume the common head sequence.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        return QuerySegmentor(backbone, decoder_channels=decoder_channels, num_classes=num_classes,
                              dropout=dropout, queries_per_class=2, interaction='static',
                              centered_residuals=False, alpha_init=0., temperature=.07)


def initialization_identity(model):
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}
    digest = tree_digest(state)
    return {'initial_head_sha256': digest, 'historical_head_sha256': HISTORICAL_HEAD_SHA,
            'matches_historical_head': digest == HISTORICAL_HEAD_SHA}
