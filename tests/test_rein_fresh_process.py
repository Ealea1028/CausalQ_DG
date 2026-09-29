import pytest
import torch

from tools.rein_fresh_process import compare_snapshots
from tools.rein_training_state import cpu_tree


def snapshot():
    return dict(indices=[22892, 5750, 13022, 5583], losses=[1., 2., 3., 4.],
                model={'weight': torch.tensor([1.])}, optimizer={'step': torch.tensor(21.)},
                scheduler={'last_step': 21}, sampler={'cursor': 84}, rng={'cpu': torch.tensor([2])})


def test_independent_snapshot_comparison_accepts_exact_and_small_float_error():
    first = snapshot()
    second = cpu_tree(first)
    second['model']['weight'].add_(1e-7)
    assert compare_snapshots(first, second)['maximum_parameter_difference'] < 1e-6


@pytest.mark.parametrize('field,value', [('indices', [0, 1, 2, 3]), ('losses', [1., 2., float('nan'), 4.]),
                                      ('scheduler', {'last_step': 20}), ('sampler', {'cursor': 80}),
                                      ('rng', {'cpu': torch.tensor([3])}),
                                      ('model', {'weight': torch.tensor([2.])})])
def test_comparison_rejects_drift(field, value):
    first, second = snapshot(), snapshot()
    second[field] = value
    with pytest.raises(ValueError):
        compare_snapshots(first, second)
