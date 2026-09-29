import random

import numpy as np
import pytest
import torch

from tools.rein_training_state import SourceSampler, capture_rng, compare_trees, cpu_tree, restore_rng, restore_optimizer_scheduler


def test_mmengine_pop_on_load_does_not_mutate_saved_optimizer_or_drift_base_lr():
    class Wrapper:
        base_param_settings = dict(lr=1e-4)

        def load_state_dict(self, state):
            # Match the pinned BaseOptimWrapper destructive-pop contract.
            base = state.pop('base_param_settings', None)
            if base is not None:
                self.base_param_settings = base
            self.state = state

        def state_dict(self):
            return dict(self.state, base_param_settings=self.base_param_settings)

    class Scheduler:
        def load_state_dict(self, state):
            self.state = state

        def state_dict(self):
            return self.state

    wrapper, scheduler = Wrapper(), Scheduler()
    saved = dict(state={0: dict(step=torch.tensor(20.))}, param_groups=[dict(lr=9e-5)],
                 base_param_settings=dict(lr=9e-5, params=torch.tensor([0.])))
    scheduled = dict(last_step=20, values=[9e-5])
    original = cpu_tree(saved)
    for _ in range(2):
        restore_optimizer_scheduler(wrapper, scheduler, saved, scheduled)
        assert wrapper.base_param_settings['lr'] == 9e-5
        wrapper.base_param_settings['lr'] = 8e-5
        wrapper.state['state'][0]['step'].add_(1)
        scheduler.state['values'][0] = 0.
        assert compare_trees(saved, original) == 0
        assert scheduled['values'] == [9e-5]
    incomplete = dict(original)
    incomplete.pop('base_param_settings')
    with pytest.raises(ValueError, match='base_param_settings'):
        restore_optimizer_scheduler(wrapper, scheduler, incomplete, scheduled)


def test_scalar_mismatch_identifies_nested_path_without_relaxing_equality():
    with pytest.raises(ValueError, match=r'optimizer.base_param_settings.lr'):
        compare_trees(dict(base_param_settings=dict(lr=1e-4)),
                      dict(base_param_settings=dict(lr=9e-5)), 'optimizer')


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
