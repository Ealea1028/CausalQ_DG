"""Serializable source sampler and RNG state for bounded REIN continuation."""

import random


class SourceSampler:
    def __init__(self, size, seed=0):
        import torch
        if type(size) is not int or size <= 0:
            raise ValueError('Positive source size required')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor = 0
        self.epoch = 0

    def take(self, count):
        import torch
        if type(count) is not int or count < 0:
            raise ValueError('Nonnegative count required')
        result = []
        while len(result) < count:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.cursor = 0
                self.epoch += 1
            end = min(self.size, self.cursor + count - len(result))
            result.extend(self.order[self.cursor:end])
            self.cursor = end
        return result

    def state_dict(self):
        return dict(size=self.size, order=list(self.order), cursor=self.cursor, epoch=self.epoch,
                    generator=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if (state['size'] != self.size or sorted(state['order']) != list(range(self.size))
                or type(state['cursor']) is not int or not 0 <= state['cursor'] <= self.size
                or type(state['epoch']) is not int or state['epoch'] < 0):
            raise ValueError('Invalid source sampler state')
        self.generator.set_state(state['generator'])
        self.order = list(state['order'])
        self.cursor, self.epoch = state['cursor'], state['epoch']


def capture_rng(include_cuda=False):
    import numpy as np
    import torch
    state = np.random.get_state()
    # Avoid numpy pickle globals: checkpoint accepts tensors and primitives only.
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=torch.cuda.get_rng_state_all() if include_cuda else [])


def restore_rng(state, include_cuda=False):
    import numpy as np
    import torch
    random.setstate(state['python'])
    values = state['numpy']
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state['torch_cpu'])
    if include_cuda:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device coverage mismatch')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def compare_trees(first, second, path='state'):
    """Require structural equality; GPU floating arithmetic allows small error."""
    import torch
    if type(first) is not type(second):
        raise ValueError(f'Continuation state type mismatch at {path}')
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError(f'Continuation state keys mismatch at {path}')
        return max((compare_trees(first[k], second[k], f'{path}.{k}') for k in first), default=0.)
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise ValueError(f'Continuation state length mismatch at {path}')
        return max((compare_trees(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(first, second))), default=0.)
    if isinstance(first, torch.Tensor):
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError(f'Continuation tensor contract mismatch at {path}')
        if first.is_floating_point():
            if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                raise ValueError(f'Nonfinite continuation tensor at {path}')
            if not torch.allclose(first, second, atol=1e-6, rtol=1e-5):
                raise ValueError(f'Continuation floating tensor mismatch at {path}')
            return (first-second).abs().max().item() if first.numel() else 0.
        if not torch.equal(first, second):
            raise ValueError(f'Continuation integer tensor mismatch at {path}')
    elif first != second:
        raise ValueError(f'Continuation scalar mismatch at {path}: {first!r} != {second!r}')
    return 0.


def restore_optimizer_scheduler(wrapper, scheduler, optimizer_state, scheduler_state):
    """MMEngine pops base_param_settings on load: never pass checkpoint storage."""
    if getattr(wrapper, 'base_param_settings', None) is not None and 'base_param_settings' not in optimizer_state:
        raise ValueError('Checkpoint lacks optimizer base_param_settings')
    wrapper.load_state_dict(cpu_tree(optimizer_state))
    scheduler.load_state_dict(cpu_tree(scheduler_state))
    compare_trees(cpu_tree(wrapper.state_dict()), optimizer_state, 'restored_optimizer')
    compare_trees(cpu_tree(scheduler.state_dict()), scheduler_state, 'restored_scheduler')


def verify_next_update(model, dataset, wrapper, scheduler, sampler, payload):
    """Two replays from the SAME disk checkpoint, not a fresh-process resume."""
    import math
    import torch
    from tools.check_rein_segmentor import validate_losses
    from tools.check_rein_slide_eval import restore_compact
    from tools.rein_schedule_smoke import compact_state, scheduled_lr

    results = []
    for replay in range(2):
        restore_compact(model, payload['model'])
        restore_optimizer_scheduler(wrapper, scheduler, payload['optimizer'], payload['scheduler'])
        sampler.load_state_dict(payload['sampler'])
        wrapper.initialize_count_status(model, 80, 84)
        wrapper.zero_grad(set_to_none=True)
        model.train(True)
        restore_rng(payload['rng'], include_cuda=True)
        indices, losses_seen = sampler.take(4), []
        for micro, index in enumerate(indices, 1):
            item = dataset[index]
            batch = model.data_preprocessor(dict(inputs=[item['inputs']], data_samples=[item['data_samples']]), training=True)
            if not (batch['data_samples'][0].gt_sem_seg.data != 255).any().item():
                raise ValueError('Continuation crop has no valid pixels')
            with wrapper.optim_context(model):
                total = validate_losses(model.loss(batch['inputs'], batch['data_samples']))
            wrapper.backward(wrapper.scale_loss(total))
            if any(p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all().item())
                   for p in model.parameters()):
                raise FloatingPointError('Invalid continuation gradient')
            if wrapper.should_update() != (micro == 4):
                raise ValueError('Continuation accumulation boundary mismatch')
            losses_seen.append(total.item())
            if micro == 4:
                wrapper.step()
                wrapper.zero_grad(set_to_none=True)
                scheduler.step()
            del total
        if any(abs(group['lr']-scheduled_lr(21)) > 1e-10 for group in wrapper.optimizer.param_groups):
            raise ValueError('Continuation LR mismatch')
        results.append(dict(indices=indices, losses=losses_seen, model=compact_state(model),
                            optimizer=cpu_tree(wrapper.state_dict()), scheduler=cpu_tree(scheduler.state_dict()),
                            sampler=sampler.state_dict()))
    if results[0]['indices'] != results[1]['indices']:
        raise ValueError('Continuation source indices mismatch')
    loss_error = max(abs(a-b) for a, b in zip(results[0]['losses'], results[1]['losses']))
    if not math.isfinite(loss_error) or loss_error > 1e-4:
        raise ValueError('Continuation loss mismatch')
    parameter_error = compare_trees(results[0]['model'], results[1]['model'])
    if parameter_error > 1e-5:
        raise ValueError('Continuation parameter difference exceeds 1e-5')
    optimizer_error = compare_trees(results[0]['optimizer'], results[1]['optimizer'], 'optimizer')
    compare_trees(results[0]['scheduler'], results[1]['scheduler'])
    compare_trees(results[0]['sampler'], results[1]['sampler'])
    restore_compact(model, payload['model'])
    restore_optimizer_scheduler(wrapper, scheduler, payload['optimizer'], payload['scheduler'])
    sampler.load_state_dict(payload['sampler'])
    wrapper.initialize_count_status(model, 80, 84)
    wrapper.zero_grad(set_to_none=True)
    restore_rng(payload['rng'], include_cuda=True)
    return dict(continuation_replay_ok=True, checkpoint_updates=20, next_update=21,
                replay_count=2, extra_microbatches_executed=8, source_indices=results[0]['indices'],
                maximum_loss_difference=loss_error, maximum_parameter_difference=parameter_error,
                maximum_optimizer_tensor_difference=optimizer_error, fresh_process_resume_verified=False)
