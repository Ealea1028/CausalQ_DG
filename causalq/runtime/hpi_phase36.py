"""Phase36 class deletion training with replayed dropout."""
import torch

from causalq.losses.hpi_localization import patch_localization_loss
from causalq.losses.query_effect_invariance import effect_invariance_loss, gt_region_prediction_effect
from causalq.losses.source_ce import deterministic_source_ce
from .input_stream import input_seed

EFFECT_CLASS_SEED = 20261031


def sample_present_class(label, step):
    """A local absolute-step RNG pairs the class across views and heads."""
    present = torch.unique(label)
    present = present[present != 255]
    if not present.numel():
        raise ValueError('Class deletion requires a GT-present class')
    seed = input_seed(EFFECT_CLASS_SEED, step, 'gta5', 'main_path_class')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    index = int(torch.randint(present.numel(), (), generator=generator))
    return present[index:index+1]


def paired_class_forward(model, features, output_size, category, *, amp):
    """Off replays the on-forward RNG and restores the post-on state."""
    device = category.device
    devices = [device] if device.type == 'cuda' else []
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if devices else None
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        on, route = model.forward_routed(features, output_size)
    # Keep the autocast scopes separate so no cached weight cast crosses branches.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.set_rng_state(cpu_state)
        if devices:
            torch.cuda.set_rng_state(cuda_state, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            off = model.closed_class_logits(features, output_size, category)
    return on, route, off


def effect_backward(head, original, photo, label, output_size, *, amp=False):
    category = sample_present_class(label, head.cursor)
    ce_losses, localization, effects = [], [], []
    for features in (original, photo):
        on, route, off = paired_class_forward(head.model, features, output_size, category, amp=amp)
        ce_losses.append(deterministic_source_ce(on.logits, label))
        localization.append(.05 * patch_localization_loss(route.match_logits, label, features.patch_map.shape[-2:]))
        effects.append(gt_region_prediction_effect(on.logits, off, label, category))
    loc = sum(localization)
    invariance = effect_invariance_loss(torch.stack(effects))
    (sum(ce_losses) + loc + invariance).backward()
    return torch.stack((*ce_losses, loc, invariance)).detach()
