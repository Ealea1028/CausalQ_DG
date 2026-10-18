from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.losses.prediction_consistency import prediction_consistency_kl
from causalq.protocols.formal_source import ROOT, formal_config, validate_formal_config
from causalq.utils.checkpoint import load_training_checkpoint
from tools.audit_formal_source_pair import audit_pair, validate_pair
from tools.deterministic_source_ce import deterministic_source_ce
from tools.prediction_source_update import prediction_source_update
from tools.strict_source_resume import capture_rng, restore_rng
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest, source_optimizer_update
from test_strict_source_runner import make_state


def test_returned_formal_pair_confusion_reduction_and_hashes(tmp_path):
    result = audit_pair()
    assert result['formal_pair_accepted']
    assert result['delta_percentage_points'] == pytest.approx(-.03200015278831403)
    assert result['checkpoint_and_full_traces_locally_available'] is False
    source = ROOT / 'experiments/STRICT_DG_PHASE22'
    for path in source.glob('*'):
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    path = tmp_path / 'pair_report.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'):
        audit_pair(tmp_path)


@pytest.mark.parametrize('change', ['budget', 'miou', 'counts', 'target'])
def test_pair_rejects_wrong_budget_metrics_and_target_flags(change):
    report = json.loads((ROOT / 'experiments/STRICT_DG_PHASE22/pair_report.json').read_bytes())
    if change == 'budget': report['reports']['M1']['formal_optimizer_updates'] = 20
    if change == 'miou': report['reports']['M1']['source_evaluation']['class_iou'][0] += .01
    if change == 'counts': report['reports']['M1']['source_evaluation']['confusion_matrix'][0][0] += 1
    if change == 'target': report['reports']['M1']['target_data_accessed'] = True
    with pytest.raises(ValueError): validate_pair(report)


def test_p1_recipe_is_m1_plus_only_fixed_prediction_control():
    config = yaml.safe_load((ROOT / 'configs/strict_dg/formal/p1_seed0.yaml').read_bytes())
    validate_formal_config(config, 'P1')
    assert config['prediction_consistency']['lambda_pred'] == 1
    assert config['prediction_consistency']['temperature'] == 1
    config.pop('prediction_consistency')
    config['experiment'] = formal_config('M1')['experiment']
    assert config == formal_config('M1')


def preparation():
    model, optimizer, dataset, config = make_state()
    model.query_bank.centered_residuals = True
    config['prediction_consistency'] = formal_config('P1')['prediction_consistency']
    sample = dataset[0]
    image, label = sample['image'].unsqueeze(0), sample['label'].unsqueeze(0)
    photo = StyleInterventionBank().photometric_view(image)
    return model, optimizer, config, image, photo, label


def test_p1_update_matches_explicit_detached_teacher_objective_and_two_forwards():
    model, optimizer, config, image, photo, label = preparation()
    label[:, 0, 0] = 255
    reference = deepcopy(model)
    reference_opt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    rng = capture_rng()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *a: calls.append(True))
    record = prediction_source_update(model, optimizer, image, photo, label, 1, config, device='cpu', amp=False)
    hook.remove()
    restore_rng(rng)
    original = reference(image)
    styled = reference(photo)
    loss_original = deterministic_source_ce(original, label)
    loss_photo = deterministic_source_ce(styled, label)
    kl = prediction_consistency_kl(styled, original.detach(), labels=label)
    (loss_original + loss_photo + kl).backward()
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1)
    reference_opt.step()
    assert len(calls) == 2
    assert record['loss_prediction_kl'] == pytest.approx(float(kl.detach()))
    assert record['objective'] == pytest.approx(float((loss_original + loss_photo + kl).detach()))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_zero_prediction_weight_reproduces_m1_step_exactly():
    states = []
    for function in (source_optimizer_update, prediction_source_update):
        model, optimizer, config, image, photo, label = preparation()
        config['prediction_consistency']['lambda_pred'] = 0
        record = function(model, optimizer, image, photo, label, 1, config,
            device='cpu', amp=False, loss_fn=deterministic_source_ce)
        states.append((tree_digest(model.state_dict()), tree_digest(optimizer.state_dict()), record))
    assert states[0][:2] == states[1][:2]
    for key, value in states[0][2].items():
        assert states[1][2][key] == value


def test_p1_runner_exact_restart_replays_stochastic_epoch_boundary(tmp_path):
    for mode, start, stop in [('full', 0, 8), ('prefix', 0, 4), ('resume', 4, 8)]:
        root = tmp_path / mode
        root.mkdir()
        model, optimizer, dataset, config = make_state()
        model.query_bank.centered_residuals = True
        config['prediction_consistency'] = formal_config('P1')['prediction_consistency']
        train_segment(model, optimizer, dataset, StyleInterventionBank(), config, root,
            start=start, stop=stop, metadata={'seed': 0, 'variant': 'P1'},
            resume=tmp_path / 'prefix/state.pth' if start else None, device='cpu', amp=False,
            update_fn=prediction_source_update)
    full = load_training_checkpoint(tmp_path / 'full/state.pth')
    resumed = load_training_checkpoint(tmp_path / 'resume/state.pth')
    for key in ('trainable_model', 'optimizer', 'rng', 'sampler'):
        assert tree_digest(full[key]) == tree_digest(resumed[key])
    assert (tmp_path / 'full/train.jsonl').read_bytes() == (tmp_path / 'prefix/train.jsonl').read_bytes() + (tmp_path / 'resume/train.jsonl').read_bytes()


def test_p1_fresh_import_never_loads_legacy_or_target():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname in ('tools.train','causalq.datasets.cityscapes'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.train_formal_source,tools.prediction_source_update,tools.audit_formal_source_pair
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
