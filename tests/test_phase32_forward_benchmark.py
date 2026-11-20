"""CPU coverage for the report arithmetic and two-checkout entry guard."""

from types import SimpleNamespace

import pytest

from tools import benchmark_phase32_forward as bench


def test_fps_uses_reciprocal_of_mean_frame_latency():
    result = bench.latency_statistics([10.0, 20.0, 30.0, 40.0])
    assert result == {'frames': 4, 'mean_ms': 25.0, 'p50_ms': 25.0, 'p95_ms': 40.0, 'fps': 40.0}


@pytest.mark.parametrize('samples', [[], [0.0], [-1.0], [float('nan')]])
def test_invalid_latency_is_not_reported(samples):
    with pytest.raises(ValueError):
        bench.latency_statistics(samples)


def test_entry_rejects_wrong_inference_checkout_before_cuda(monkeypatch):
    monkeypatch.setattr(bench, '_git', lambda *_: 'wrong')
    monkeypatch.setattr(bench.torch.cuda, 'is_available', lambda: pytest.fail('CUDA inspected for wrong checkout'))
    with pytest.raises(ValueError, match='historical Phase32'):
        bench.benchmark(SimpleNamespace())
