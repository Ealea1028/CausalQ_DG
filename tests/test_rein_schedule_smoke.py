import pytest
import torch
import copy
import sys
from types import ModuleType, SimpleNamespace

from tools.rein_schedule_smoke import accepted_protocol, compact_state, scheduled_lr, restore_protocol_config


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
    monkeypatch.setitem(sys.modules, "mmengine.config", module)
    raw = {"model": {"decode_head": {"transformer_decoder": {
        "layer_cfg": {"cross_attn_cfg": {"num_heads": 8}}}}}}
    report = dict(ok=True, stage="complete", exact_project_pairing=True, source_pairs=24966,
                  target_pairs=500, samples=[{}]*10, target_labels_optimized=False, adapted_protocol=raw)
    original = copy.deepcopy(report)
    result = restore_protocol_config(report)
    assert result.model.decode_head.transformer_decoder.layer_cfg.cross_attn_cfg.num_heads == 8
    assert calls == [raw] and report == original
    module.Config = lambda value: value
    with pytest.raises(AttributeError):
        restore_protocol_config(report)


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
