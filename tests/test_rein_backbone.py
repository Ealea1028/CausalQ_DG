import pytest
import torch

from tools.check_rein_backbone import validate_load_keys, validate_outputs


def test_complete_pretrained_coverage_with_only_new_adapters():
    source = {f"backbone.{i}" for i in range(343)}
    validate_load_keys(source, source | {"reins.scale"}, ["reins.scale"], [])
    with pytest.raises(ValueError, match="coverage"):
        validate_load_keys(source - {"backbone.0"}, source | {"reins.scale"}, [], [])
    with pytest.raises(ValueError, match="Only"):
        validate_load_keys(source, source | {"reins.scale"}, [], ["extra"])


def test_output_contract_and_finiteness():
    # Expanded views keep this CPU contract test lightweight.
    features = [torch.zeros(1).expand(1, 1024, size, size) for size in (128, 64, 32, 16)]
    queries = torch.zeros(100, 256)
    validate_outputs(features, queries)
    with pytest.raises(ValueError, match="pyramid"):
        validate_outputs(features[:3], queries)
    with pytest.raises(ValueError, match="query"):
        validate_outputs(features, queries[:1])
    queries[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        validate_outputs(features, queries)
