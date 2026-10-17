"""Unweighted valid-pixel CE without the CUDA 2-D NLL reduction kernel."""

import torch
from torch.nn import functional as F


BACKEND = "fp32_log_softmax_gather_valid_mean_v1"


def deterministic_source_ce(logits, labels, *, ignore_index=255):
    if logits.ndim != 4 or labels.ndim != 3:
        raise ValueError("Expected BCHW logits and BHW labels")
    if logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Logits and labels must have matching batch/spatial dimensions")
    if labels.dtype != torch.long:
        raise ValueError("Source labels must have torch.long dtype")
    valid = labels != ignore_index
    count = valid.sum()
    if not count:
        raise ValueError("Segmentation batch contains no valid pixels")
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[1])).any():
        raise ValueError("Source class ID outside logits class range")
    safe_labels = labels.masked_fill(~valid, 0)
    log_prob = F.log_softmax(logits.float(), dim=1)
    per_pixel = -log_prob.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    return per_pixel.masked_fill(~valid, 0).sum() / count


def cuda_preflight():
    """Check GPU loss/gradient against native CPU CE and exact GPU replay.

    Called only by the AutoDL launcher, before loading weights or datasets.
    Tolerances here compare equivalent reduction implementations; continuation
    still requires exact state/trace equality with no tolerance.
    """
    if not torch.are_deterministic_algorithms_enabled() or not torch.cuda.is_available():
        raise RuntimeError("CE preflight requires AutoDL CUDA with strict determinism")
    values = torch.linspace(-8, 8, 2 * 19 * 8 * 8).reshape(2, 19, 8, 8)
    labels = torch.arange(2 * 8 * 8).reshape(2, 8, 8).remainder(19)
    labels.reshape(-1)[::7] = 255
    reference = values.clone().requires_grad_()
    reference_loss = F.cross_entropy(reference, labels, ignore_index=255)
    reference_loss.backward()
    results = []
    for _ in range(2):
        value = values.to("cuda:0").requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = deterministic_source_ce(value, labels.to("cuda:0"))
        loss.backward()
        results.append((loss.detach().cpu(), value.grad.detach().cpu()))
    loss_error = float((results[0][0] - reference_loss.detach()).abs())
    gradient_error = float((results[0][1] - reference.grad).abs().max())
    if not (torch.isfinite(results[0][0]) and torch.isfinite(results[0][1]).all()
            and loss_error <= 1e-5 and gradient_error <= 1e-6):
        raise ValueError("Deterministic GPU CE differs from native CPU reference")
    if not all(torch.equal(a, b) for a, b in zip(results[0], results[1])):
        raise ValueError("Deterministic GPU CE replay differs")
    return {"backend": BACKEND, "cpu_reference_loss_max_error": loss_error,
            "cpu_reference_gradient_max_error": gradient_error, "gpu_replay_exact": True,
            "strict_determinism_enabled": True}
