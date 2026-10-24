"""Returned small source-metadata fixtures; no original checkpoints or images."""

from copy import deepcopy
import hashlib
import json

import pytest

from causalq.protocols.diagnostic_20261020 import ROOT, NEW_UNCENTERED
from causalq.protocols.diagnostic_formal import release_contract, verify_release
from tools.accept_phase29_formal import accept, PRODUCER


def make_export(root, mutate=None):
    accepted = json.loads((ROOT / 'experiments/DIAGNOSTIC_20261020/phase29_acceptance.json').read_bytes())
    combined = deepcopy(accepted['formal_report'])
    if mutate:mutate(combined['reports']['UA'])
    files = []
    for variant in NEW_UNCENTERED:
        directory = root / variant
        directory.mkdir()
        report = combined['reports'][variant]
        values = {'formal_report.json': report, 'metadata.json': report['metadata'],
                  'initialization.json': {'head_sha256': report['initial_head_sha256'],
                      'fresh_seed': 0, 'promoted_pilot_checkpoint': False}}
        for name, value in values.items():
            path = directory / name
            path.write_text(json.dumps(value))
            files.append(path)
        path = directory / 'config.yaml'
        path.write_bytes((ROOT / f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml').read_bytes())
        files.append(path)
    for name, value in [('phase29_report.json', combined), ('run_manifest.json', {
        'phase': 29, 'git_sha': PRODUCER, 'phase29_contract_sha256': verify_release(), 'contract': release_contract()})]:
        path = root / name
        path.write_text(json.dumps(value))
        files.append(path)
    (root / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_formal_acceptance_reduces_integer_confusions_without_remote_replay(tmp_path):
    make_export(tmp_path)
    result = accept(tmp_path)
    assert result['accepted'] and result['source_confusion_reduced_independently']
    assert result['formal_report']['source_miou']['UR'] == pytest.approx(.7389118801033604)
    assert result['checkpoints_and_full_traces_available_locally'] is False
    assert len(result['file_sha256']) == 18
    (tmp_path / 'UP/formal_report.json').write_bytes(b'altered')
    with pytest.raises(ValueError, match='SHA256'):accept(tmp_path)


@pytest.mark.parametrize('change', ['budget', 'target', 'centered', 'miou', 'coverage', 'negative_count'])
def test_formal_acceptance_rejects_wrong_training_or_evaluation(change, tmp_path):
    def mutate(report):
        if change == 'budget':report['formal_optimizer_updates'] = 1000
        elif change == 'target':report['target_data_accessed'] = True
        elif change == 'centered':report['metadata']['centered_residuals'] = True
        elif change == 'miou':report['source_evaluation']['class_iou'][0] += .001
        elif change == 'coverage':report['source_evaluation']['sample_count'] = 150
        else:report['source_evaluation']['confusion_matrix'][0][0] = -1
    make_export(tmp_path, mutate)
    with pytest.raises(ValueError):accept(tmp_path)
