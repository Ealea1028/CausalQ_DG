"""Equivalent CE values/derivatives without the nondeterministic CUDA NLL path."""

import pytest
import torch
from torch.nn import functional as F

from tools.deterministic_source_ce import deterministic_source_ce, cuda_preflight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ignore", [255, -100])
def test_loss_and_gradient_match_native_and_closed_form(dtype, ignore):
    generator = torch.Generator().manual_seed(831)
    values = torch.randn(2, 5, 7, 9, generator=generator).to(dtype).transpose(2, 3)
    labels = torch.randint(0, 5, (2, 9, 7), generator=generator)
    labels[:, ::2, ::3] = ignore
    first = values.detach().requires_grad_()
    second = values.detach().requires_grad_()
    loss = deterministic_source_ce(first, labels, ignore_index=ignore)
    reference = F.cross_entropy(second.float(), labels, ignore_index=ignore)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-5, atol=1e-7)
    valid = labels != ignore
    safe = labels.masked_fill(~valid, 0)
    expected = (values.float().softmax(1) - F.one_hot(safe, 5).permute(0, 3, 1, 2))
    expected *= valid.unsqueeze(1) / valid.sum()
    torch.testing.assert_close(first.grad, expected.to(dtype), rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(first.grad.permute(0, 2, 3, 1)[~valid]) == 0
    assert loss.dtype == torch.float32


def test_one_valid_pixel_and_extreme_logits():
    values = torch.tensor([[[[1000., 0.]], [[-1000., 1.]], [[-2000., 2.]]]], requires_grad=True)
    labels = torch.tensor([[[1, 255]]])
    loss = deterministic_source_ce(values, labels)
    assert loss.item() == 2000
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert torch.equal(values.grad[..., 1], torch.zeros_like(values.grad[..., 1]))


@pytest.mark.parametrize("labels,match", [
    (torch.full((1, 2, 2), 255), "no valid"),
    (torch.full((1, 2, 2), -1), "class ID"),
    (torch.full((1, 2, 2), 3), "class ID"),
    (torch.zeros(1, 2, 2), "torch.long"),
    (torch.zeros(1, 3, 2, dtype=torch.long), "matching"),
])
def test_invalid_targets_rejected(labels, match):
    with pytest.raises(ValueError, match=match):
        deterministic_source_ce(torch.zeros(1, 3, 2, 2), labels)


def test_strict_mode_never_calls_native_cross_entropy(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(F, "cross_entropy", lambda *a, **k: pytest.fail("native NLL path used"))
    try:
        torch.use_deterministic_algorithms(True)
        value = torch.randn(1, 3, 2, 2, requires_grad=True)
        loss = deterministic_source_ce(value, torch.zeros(1, 2, 2, dtype=torch.long))
        loss.backward()
        assert torch.isfinite(value.grad).all()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def test_cuda_preflight_refuses_local_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="AutoDL CUDA"):
        cuda_preflight()
