import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest
import torch

from causalq.models.dinov3_wrapper import DINOv3Backbone
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.identity import file_sha
from causalq.runtime.lora_l1_formal import (DIRECTORY, VARIANTS, L1Trainer, core_l1_head,
                                          load_contract, checkpoint_rows)
from causalq.runtime.lora_training import compact_state, load_state_file, rng_digest, seed_training_rng
from test_hpi_backbone import tiny_backbone
from test_lora_backbones import FakeHF
from test_lora_phase40 import packet
from tools.check_lora_resume import compare
from tools.train_lora_core_d3_l1 import package, completed, status, feature_record


def build(variant='UA', kind='d2', horizon=3):
    torch.manual_seed(7)
    base = tiny_backbone() if kind == 'd2' else DINOv3Backbone(FakeHF(), intermediate_indices=(1, 2, 3, 4))
    model = core_l1_head(base, variant, decoder_channels=32, num_classes=3, blocks=(2, 3))
    return L1Trainer(model, variant, horizon=horizon)


@pytest.mark.parametrize('kind', ['d2', 'd3'])
def test_common_initialization_and_separate_adapters_preserve_rng(kind):
    models = [build(variant, kind) for variant in VARIANTS]
    assert len({m.initial_head_sha256 for m in models}) == 1
    assert len({m.initial_adapter_sha256 for m in models}) == 1
    assert len({id(m.model.backbone) for m in models}) == 4
    frozen = {id(p): p.clone() for p in models[0].frozen}
    before = rng_digest(capture_rng())
    models[0].advance(packet())
    assert before == rng_digest(capture_rng())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in models[0].model.backbone.named_parameters() if n.endswith('.B'))
    assert all(p.grad is None and torch.equal(p, frozen[id(p)]) for p in models[0].frozen)
    assert all(m.step == 0 for m in models[1:])


@pytest.mark.parametrize('variant', VARIANTS)
def test_l1_continuous_and_restored_model_optimizer_rng_are_exact(tmp_path, variant):
    identity = {'synthetic': True, 'variant': variant, 'lora_enabled': True}
    seed_training_rng(0); continuous = build(variant)
    continuous.advance(packet()); continuous.advance(packet(8))
    expected_rng = rng_digest(capture_rng())
    seed_training_rng(0); prefix = build(variant)
    prefix.advance(packet()); prefix.commit(tmp_path, identity)
    saved = load_state_file(tmp_path/'00000001/state.pt')
    assert any(k.startswith('backbone.') for k in saved['model'])
    assert not any('.base.weight' in k for k in saved['model'])
    resumed = build(variant); assert resumed.restore(tmp_path, identity) == 1
    resumed.advance(packet(8))
    assert compare(compact_state(continuous.model), compact_state(resumed.model)) == 0
    assert compare(continuous.optimizer.state_dict(), resumed.optimizer.state_dict()) == 0
    assert rng_digest(capture_rng()) == expected_rng
    assert resumed.commit(tmp_path, identity)['step'] == 2


def test_failed_optimizer_cannot_commit_a_partial_update(tmp_path, monkeypatch):
    import causalq.runtime.lora_l1_formal as formal
    trainer = build()
    def fail(*args, **kwargs):
        with torch.no_grad(): next(p for p in trainer.model.parameters() if p.requires_grad).add_(1)
        raise RuntimeError('interrupted update')
    monkeypatch.setattr(formal, 'update', fail)
    with pytest.raises(RuntimeError): trainer.advance(packet())
    assert trainer.tainted and trainer.step == 0
    assert trainer.commit(tmp_path, {'test': True}) is None
    assert not list(tmp_path.rglob('state.pt'))


@pytest.mark.parametrize('fault', ['identity', 'cursor', 'lora', 'optimizer', 'rng', 'hash', 'partial'])
def test_resume_rejects_changed_or_incomplete_snapshot(tmp_path, fault):
    trainer = build(); identity = {'lora': True}; trainer.advance(packet()); trainer.commit(tmp_path, identity)
    folder = tmp_path/'00000001'; state = load_state_file(folder/'state.pt')
    marker = json.loads((folder/'complete.json').read_text())
    if fault == 'identity': identity = {'lora': False}
    if fault == 'cursor': state['source_cursor'] = 0
    if fault == 'lora': state['model'].pop(next(k for k in state['model'] if k.startswith('backbone.')))
    if fault == 'optimizer': state['optimizer']['state'].pop(next(iter(state['optimizer']['state'])))
    if fault == 'rng': state['rng'].pop('numpy')
    if fault == 'partial': (folder/'state.pt.partial').write_bytes(b'partial')
    if fault in ('cursor', 'lora', 'optimizer', 'rng'):
        torch.save(state, folder/'state.pt'); marker['sha256'] = file_sha(folder/'state.pt')
        (folder/'complete.json').write_text(json.dumps(marker))
    if fault == 'hash': (folder/'state.pt').write_bytes(b'bad')
    with pytest.raises((ValueError, KeyError, OSError)): build().restore(tmp_path, identity)


def test_cursor_frozen_weights_and_unadapted_backbone_guards():
    trainer = build()
    with pytest.raises(ValueError): trainer.advance(packet(8))
    with torch.no_grad(): trainer.frozen[0].add_(1)
    with pytest.raises(ValueError, match='mutated'): trainer.advance(packet())
    with pytest.raises(ValueError): L1Trainer(copy.deepcopy(trainer.model), 'B3')


def test_exact_contract_and_source_only_phase():
    config, freeze = load_contract()
    assert config['phase'] == 42 and config['backbones'] == ['dinov3_vitl16']
    assert config['lora_enabled'] and config['target_access_authorized'] is False
    assert config['formal_steps']*len(config['variants']) == 160000
    assert freeze['checkpoint_retention'] == 'all segments; conservative full-history budget'


def test_feature_record_keeps_shapes_without_tensors():
    features = SimpleNamespace(patch_tokens=torch.empty(4, 1024, 1024, device='meta', requires_grad=True),
                              cls_token=torch.empty(4, 1024, device='meta'),
                              register_tokens=torch.empty(4, 4, 1024, device='meta'),
                              intermediate_maps=[torch.empty(4, 1024, 32, 32, device='meta')]*4)
    record = feature_record(features)
    assert record['patch_tokens'] == [4, 1024, 1024] and record['patch_requires_grad']
    json.dumps(record)
    features.cls_token = None
    with pytest.raises(ValueError): feature_record(features)


def test_export_never_includes_checkpoint_payload_or_complete_trace(tmp_path):
    output = tmp_path/'run'; output.mkdir()
    (output/'report.json').write_text('{"ok":false}')
    (output/'state.pt').write_bytes(b'checkpoint')
    (output/'input_trace.jsonl').write_text('{"source":1}\n')
    (output/'worker.log').write_text('worker log')
    (output/'exit_code.txt').write_text('controller_exit_code=0\n')
    archive = tmp_path/'return.tar.gz'; package(output, archive)
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {'report.json', 'exit_code.txt', 'worker.log.tail', 'export_manifest.json'}
        manifest = json.load(bundle.extractfile('export_manifest.json'))
        assert not manifest['input_trace.jsonl']['included']
    assert (output/'state.pt').read_bytes() == b'checkpoint'
    with pytest.raises(ValueError): package(output, archive)


def test_completed_model_requires_byte_exact_final_and_matching_variant(tmp_path):
    sha = 'a'*40; folder = tmp_path/'U0'; checkpoint = folder/'checkpoints/00040000'; checkpoint.mkdir(parents=True)
    payload = checkpoint/'state.pt'; payload.write_bytes(b'synthetic')
    identity = {'phase': 42, 'backbone': 'dinov3_vitl16', 'variant': 'U0', 'lora_enabled': True,
                'training_producer_git_sha': sha, 'data_root': str(tmp_path.resolve()),
                'pretrained_root': str(tmp_path.resolve()), 'freeze_sha256': file_sha(DIRECTORY/'phase42_freeze.json')}
    marker = {'step': 40000, 'identity': identity, 'sha256': file_sha(payload)}
    (checkpoint/'complete.json').write_text(json.dumps(marker))
    report = {'ok': True, 'status': 'complete', 'git_sha': sha, 'identity': identity, 'base_unchanged': True,
              'adapter_changed': True, 'performance_target_met': True, 'checkpoint': {'sha256': marker['sha256']}}
    (folder/'training_report.json').write_text(json.dumps(report))
    args = SimpleNamespace(expected_sha=sha, data_root=tmp_path, pretrained_root=tmp_path)
    assert completed(tmp_path, 'U0', args)['ok']
    payload.write_bytes(b'corrupt')
    with pytest.raises(ValueError): completed(tmp_path, 'U0', args)


def test_coordinator_skips_verified_completed_models_without_recounting_updates(tmp_path, monkeypatch):
    import tools.train_lora_core_d3_l1 as cli
    args = SimpleNamespace(output=tmp_path, expected_sha='a'*40)
    monkeypatch.setattr(cli, 'load_contract', lambda: None)
    monkeypatch.setattr(cli, 'require_retirement', lambda: None)
    monkeypatch.setattr(cli, 'hints', lambda *a: {v: 40000 for v in VARIANTS})
    monkeypatch.setattr(cli, 'completed', lambda *a: {'ok': True, 'formal_optimizer_updates_this_invocation': 40000})
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=100 << 30))
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *a, **k: pytest.fail('Completed models must not spawn training'))
    assert cli.run(args) == 0
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['formal_optimizer_updates_total'] == 160000
    assert report['formal_optimizer_updates_this_invocation'] == 0


def test_stale_worker_report_cannot_certify_a_new_failed_attempt(tmp_path, monkeypatch):
    import tools.train_lora_core_d3_l1 as cli
    args = SimpleNamespace(output=tmp_path, expected_sha='a'*40, data_root=tmp_path, pretrained_root=tmp_path,
                           lock_fds=(8, 9))
    folder = tmp_path/'U0'; folder.mkdir(); (folder/'training_report.json').write_text('{"ok":false}')
    monkeypatch.setattr(cli, 'load_contract', lambda: None)
    monkeypatch.setattr(cli, 'require_retirement', lambda: None)
    monkeypatch.setattr(cli, 'hints', lambda *a: {v: 0 for v in VARIANTS})
    monkeypatch.setattr(cli, 'completed', lambda *a: None)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=100 << 30))
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    child = SimpleNamespace(returncode=2, poll=lambda: 2)
    def launch(*a, **kwargs):
        assert kwargs['start_new_session'] is True and kwargs['pass_fds'] == (8, 9)
        return child
    monkeypatch.setattr(cli.subprocess, 'Popen', launch)
    assert cli.run(args) == 2
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['status'] == 'failed' and 'Worker exited 2' in report['error']
    assert report['formal_optimizer_updates_this_invocation'] == 0
