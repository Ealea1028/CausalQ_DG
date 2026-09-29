import pytest
import torch
import copy
import sys
from types import ModuleType, SimpleNamespace
import json
from pathlib import Path

from tools.rein_schedule_smoke import accepted_protocol, compact_state, scheduled_lr, restore_protocol_config, rebuild_protocol, training_budget, validate_slide_gate


def test_bounded_pilot_budget_cannot_start_formal_training():
    assert training_budget(20) == 80
    assert training_budget(500) == 2000
    with pytest.raises(ValueError):
        training_budget(40000)


def test_pilot_requires_original_resolution_slide_acceptance():
    report = dict(ok=True, stage='complete', sample_count=5, target_labels_optimized=False,
                  git_sha='59d522bcac07a73c4cd0b1cb1185983a389b7281',
                  samples=[dict(prediction_shape=[1024, 2048], gt_shape=[1024, 2048]) for _ in range(5)])
    validate_slide_gate(report)
    report['samples'][0]['gt_shape'] = [512, 512]
    with pytest.raises(ValueError):
        validate_slide_gate(report)


def protocol_fixture():
    from tools.rein_protocol_adapter import adapt_protocol
    upstream = dict(
        model=dict(backbone=dict(init_cfg="old"), decode_head=dict(transformer_decoder=dict(
            layer_cfg=dict(cross_attn_cfg=dict(num_heads=8))))),
        train_dataloader=dict(dataset=dict(pipeline=[dict(type="LoadAnnotations"),
            dict(type="RandomCrop", crop_size=(512, 512))])),
        val_dataloader=dict(dataset=dict(datasets=[dict(type="CityscapesDataset", pipeline=[],
            scale=(1024, 512))])),
        optim_wrapper=dict(optimizer=dict(betas=(0.9, 0.999))),
        default_hooks=dict(checkpoint=dict(interval=4000)),
        model_test_sizes=dict(crop_size=(512, 512), stride=(341, 341)),
        scales=[256, 512, 1024])
    adapted = adapt_protocol(upstream, Path('/data'), Path('/images'), Path('/labels'))
    report = dict(ok=True, stage="complete", exact_project_pairing=True, source_pairs=24966,
                  target_pairs=500, samples=[{}]*10, target_labels_optimized=False,
                  adapted_protocol=json.loads(json.dumps(adapted)))
    return upstream, report


def test_rebuild_preserves_all_original_container_types_and_values():
    upstream, report = protocol_fixture()
    original = copy.deepcopy(report)
    result = rebuild_protocol(report, upstream)
    crop = next(step for step in result['train_dataloader']['dataset']['pipeline']
                if step['type'] == 'ReinValidRandomCrop')
    assert crop['crop_size'] == (512, 512) and isinstance(crop['crop_size'], tuple)
    assert isinstance(result['optim_wrapper']['optimizer']['betas'], tuple)
    assert isinstance(result['model_test_sizes']['stride'], tuple)
    assert isinstance(result['scales'], list)
    assert json.loads(json.dumps(result)) == report['adapted_protocol']
    assert report == original


def test_rebuild_rejects_config_drift():
    upstream, report = protocol_fixture()
    upstream['scales'].append(2048)
    with pytest.raises(ValueError, match='differs'):
        rebuild_protocol(report, upstream)


def test_json_config_restoration_routes_through_recursive_config(monkeypatch):
    # OpenMMLab is intentionally absent locally; real integration is remote.
    calls = []
    def recursive(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: recursive(item) for key, item in value.items()})
        return value
    module = ModuleType("mmengine.config")
    def factory(value):
        calls.append(copy.deepcopy(value))
        return recursive(value)
    module.Config = factory
    upstream, report = protocol_fixture()
    factory.fromfile = lambda path: SimpleNamespace(to_dict=lambda: copy.deepcopy(upstream))
    monkeypatch.setitem(sys.modules, "mmengine.config", module)
    original = copy.deepcopy(report)
    result = restore_protocol_config(report, Path('/rein'))
    assert result.model.decode_head.transformer_decoder.layer_cfg.cross_attn_cfg.num_heads == 8
    assert json.loads(json.dumps(calls[0])) == report['adapted_protocol'] and report == original
    module.Config = lambda value: value
    module.Config.fromfile = factory.fromfile
    with pytest.raises(AttributeError):
        restore_protocol_config(report, Path('/rein'))


def test_update_schedule_units():
    assert scheduled_lr(0) == 1e-4
    assert scheduled_lr(40000) == 0
    assert scheduled_lr(20) > scheduled_lr(80)
    with pytest.raises(ValueError):
        scheduled_lr(-1)


def test_protocol_requires_complete_coverage():
    report = dict(ok=True, stage="complete", exact_project_pairing=True, source_pairs=24966,
                  target_pairs=500, samples=[{}]*10, target_labels_optimized=False, adapted_protocol={"a": 1})
    assert accepted_protocol(report) == {"a": 1}
    report["target_pairs"] = 50
    with pytest.raises(ValueError):
        accepted_protocol(report)


def test_compact_state_excludes_frozen_weights_and_retains_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model[0].requires_grad_(False)
    state = compact_state(model)
    assert "0.weight" not in state and "0.bias" not in state
    assert "1.weight" in state and "1.running_mean" in state and "1.num_batches_tracked" in state
    original = state["1.weight"].clone()
    with torch.no_grad():
        model[1].weight.add_(1)
    assert torch.equal(original, state["1.weight"])
