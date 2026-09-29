import random

import numpy as np
import pytest
import torch

from tools.rein_training_state import SourceSampler, capture_rng, compare_trees, cpu_tree, restore_rng


def test_sampler_safe_reload_across_epoch_boundary(tmp_path):
    sampler = SourceSampler(7, seed=0)
    assert sampler.take(5) == torch.randperm(7, generator=torch.Generator().manual_seed(0)).tolist()[:5]
    path = tmp_path/'sampler.pth'
    torch.save(sampler.state_dict(), path)
    expected = sampler.take(17)
    restored = SourceSampler(7, seed=999)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.take(17) == expected
    assert restored.epoch == sampler.epoch == 3
    assert compare_trees(restored.state_dict(), sampler.state_dict()) == 0


def test_sampler_rejects_bad_permutation_and_cursor():
    sampler = SourceSampler(3)
    state = sampler.state_dict()
    state['order'] = [0, 0, 1]
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    state = sampler.state_dict()
    state['cursor'] = 4
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


def test_rng_primitive_checkpoint_replays_cpu_streams(tmp_path):
    original = capture_rng()
    try:
        path = tmp_path/'rng.pth'
        torch.save(capture_rng(), path)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(torch.load(path, weights_only=True))
        actual = (random.random(), np.random.rand(), torch.rand(3))
        assert expected[:2] == actual[:2]
        assert torch.equal(expected[2], actual[2])
    finally:
        restore_rng(original)


def test_cpu_optimizer_and_dropout_continuation_roundtrip(tmp_path):
    original = capture_rng()
    try:
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: (1-i/40000)**.9)

        def step():
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(4, 3)).square().mean().backward()
            optimizer.step()
            scheduler.step()

        step()
        path = tmp_path/'training.pth'
        torch.save(dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                        scheduler=cpu_tree(scheduler.state_dict()), rng=capture_rng()), path)
        step()
        expected_model = cpu_tree(model.state_dict())
        expected_optimizer = cpu_tree(optimizer.state_dict())
        payload = torch.load(path, weights_only=True)
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        restore_rng(payload['rng'])
        step()
        assert compare_trees(expected_model, cpu_tree(model.state_dict())) == 0
        assert compare_trees(expected_optimizer, cpu_tree(optimizer.state_dict())) == 0
    finally:
        restore_rng(original)


def test_tree_comparison_rejects_missing_state_and_nonfinite():
    with pytest.raises(ValueError):
        compare_trees(dict(a=1), dict(b=1))
    with pytest.raises(ValueError):
        compare_trees(torch.tensor([1.]), torch.tensor([float('nan')]))
    with pytest.raises(ValueError):
        compare_trees(torch.tensor([1]), torch.tensor([2]))
