from copy import deepcopy
import json
import subprocess
import sys

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.formal_source import ROOT, formal_config, validate_formal_config
from causalq.utils.checkpoint import load_training_checkpoint
from tools.audit_prediction_source_result import audit_prediction, validate_prediction_result
from tools.deterministic_source_ce import deterministic_source_ce
from tools.full_cqe_source_update import full_cqe_source_update
from tools.strict_source_resume import capture_rng, restore_rng
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest, source_optimizer_update
from test_strict_source_runner import make_state


def test_p1_returned_hashes_and_independent_confusion(tmp_path):
    result = audit_prediction()
    assert result['formal_p1_accepted']
    assert result['source_miou'] == pytest.approx(.7384414154795075)
    assert result['delta_vs_m1_percentage_points'] == pytest.approx(-.01002096614956)
    assert result['checkpoint_and_full_traces_locally_available'] is False
    source = ROOT / 'experiments/STRICT_DG_PHASE23'
    for name in ('formal_report.json', 'REPORT_SHA256SUMS'):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    path = tmp_path / 'formal_report.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'): audit_prediction(tmp_path)


@pytest.mark.parametrize('change', ['budget', 'iou', 'target'])
def test_p1_rejects_inconsistent_evidence(change):
    report = json.loads((ROOT / 'experiments/STRICT_DG_PHASE23/formal_report.json').read_bytes())
    if change == 'budget': report['formal_optimizer_updates'] = 20
    if change == 'iou': report['source_evaluation']['class_iou'][0] += .01
    if change == 'target': report['target_data_accessed'] = True
    with pytest.raises(ValueError): validate_prediction_result(report)


def test_m2_recipe_is_m1_plus_only_original_full_cqe():
    config = yaml.safe_load((ROOT / 'configs/strict_dg/formal/m2_seed0.yaml').read_bytes())
    validate_formal_config(config, 'M2')
    assert config['causal_query_effect']['lambda_cqe'] == 1
    assert config['causal_query_effect']['normalization'] == 'l2_per_class_map'
    assert config['causal_query_effect']['reference_stop_gradient'] is True
    config.pop('causal_query_effect')
    config['experiment'] = formal_config('M1')['experiment']
    assert config == formal_config('M1')


def preparation(alpha=.1):
    model, optimizer, dataset, config = make_state()
    model.query_bank.centered_residuals = True
    with torch.no_grad(): model.query_head.alpha.fill_(alpha)
    config['causal_query_effect'] = formal_config('M2')['causal_query_effect']
    sample = dataset[0]
    image, label = sample['image'].unsqueeze(0), sample['label'].unsqueeze(0)
    photo = StyleInterventionBank().photometric_view(image)
    return model, optimizer, config, image, photo, label


@pytest.mark.parametrize('alpha', [0., .1])
def test_m2_matches_explicit_full_effect_objective_two_forwards(alpha):
    model, optimizer, config, image, photo, label = preparation(alpha)
    label[:, 0, 0] = 255
    reference = deepcopy(model)
    reference_opt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    rng = capture_rng()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *a: calls.append(True))
    record = full_cqe_source_update(model, optimizer, image, photo, label, 1, config, device='cpu', amp=False)
    hook.remove()
    restore_rng(rng)
    original = reference.forward_components(image)
    styled = reference.forward_components(photo)
    assert original.anchor_scaled_delta_logits is None and styled.anchor_scaled_delta_logits is None
    loss_original = deterministic_source_ce(original.logits, label)
    loss_photo = deterministic_source_ce(styled.logits, label)
    cqe = causal_query_effect_loss(styled.scaled_delta_logits, original.scaled_delta_logits.detach(), label)
    (loss_original + loss_photo + cqe).backward()
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1)
    reference_opt.step()
    assert len(calls) == 2
    assert record['loss_full_cqe'] == pytest.approx(float(cqe.detach()))
    assert record['loss_full_cqe'] == 0 if alpha == 0 else record['loss_full_cqe'] > 0
    assert record['objective'] == pytest.approx(float((loss_original + loss_photo + cqe).detach()))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_zero_full_cqe_weight_reproduces_m1_update_exactly():
    states = []
    for function in (source_optimizer_update, full_cqe_source_update):
        model, optimizer, config, image, photo, label = preparation()
        config['causal_query_effect']['lambda_cqe'] = 0
        record = function(model, optimizer, image, photo, label, 1, config,
            device='cpu', amp=False, loss_fn=deterministic_source_ce)
        states.append((tree_digest(model.state_dict()), tree_digest(optimizer.state_dict()), record))
    assert states[0][:2] == states[1][:2]
    for key, value in states[0][2].items(): assert states[1][2][key] == value


def test_m2_runner_exact_resume_with_crop_style_dropout_and_epoch_rollover(tmp_path):
    for mode, start, stop in [('full', 0, 8), ('prefix', 0, 4), ('resume', 4, 8)]:
        root = tmp_path / mode
        root.mkdir()
        model, optimizer, dataset, config = make_state()
        model.query_bank.centered_residuals = True
        config['causal_query_effect'] = formal_config('M2')['causal_query_effect']
        train_segment(model, optimizer, dataset, StyleInterventionBank(), config, root,
            start=start, stop=stop, metadata={'seed': 0, 'variant': 'M2'},
            resume=tmp_path / 'prefix/state.pth' if start else None, device='cpu', amp=False,
            update_fn=full_cqe_source_update)
    full = load_training_checkpoint(tmp_path / 'full/state.pth')
    resumed = load_training_checkpoint(tmp_path / 'resume/state.pth')
    for key in ('trainable_model', 'optimizer', 'rng', 'sampler'):
        assert tree_digest(full[key]) == tree_digest(resumed[key])
    assert (tmp_path / 'full/train.jsonl').read_bytes() == (tmp_path / 'prefix/train.jsonl').read_bytes() + (tmp_path / 'resume/train.jsonl').read_bytes()


def test_m2_fresh_import_blocks_legacy_and_target():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname in ('tools.train','causalq.datasets.cityscapes'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.train_formal_source,tools.full_cqe_source_update,tools.audit_prediction_source_result
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
