import pytest
import torch

from causalq.models.rein_query_residual import ReinClassQueryResidual


def test_zero_initialized_bridge_is_exact_identity():
    model = ReinClassQueryResidual(num_classes=4, queries_per_class=2,
                                   hidden_channels=8, alpha_init=0.)
    base = torch.randn(2, 4, 6, 7)
    output = model(base)
    assert torch.equal(output.logits, base)
    assert output.query_residual.shape == base.shape


def test_counterfactual_removes_only_selected_class_residual():
    model = ReinClassQueryResidual(num_classes=4, queries_per_class=2,
                                   hidden_channels=8, alpha_init=.5)
    output = model(torch.randn(2, 4, 6, 7))
    counterfactual = output.counterfactual_logits(2)
    assert torch.equal(counterfactual[:, 2], output.base_logits[:, 2])
    assert torch.equal(counterfactual[:, :2], output.logits[:, :2])
    assert torch.equal(counterfactual[:, 3:], output.logits[:, 3:])
    assert torch.allclose(output.logits[:, 2] - counterfactual[:, 2],
                          output.class_effect(2))


def test_segmentation_gradient_opens_alpha_then_reaches_query_route():
    model = ReinClassQueryResidual(num_classes=4, queries_per_class=2,
                                   hidden_channels=8, alpha_init=0.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    base = torch.randn(2, 4, 6, 7)
    target = torch.randint(0, 4, (2, 6, 7))
    first = torch.nn.functional.cross_entropy(model(base).logits, target)
    first.backward()
    assert model.alpha.grad is not None and model.alpha.grad.abs().item() > 0
    assert model.query_bank.grad is not None and model.query_bank.grad.abs().sum().item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = torch.nn.functional.cross_entropy(model(base).logits, target)
    second.backward()
    assert model.query_bank.grad is not None and model.query_bank.grad.abs().sum().item() > 0
    assert model.pixel_projection.weight.grad is not None
    assert model.pixel_projection.weight.grad.abs().sum().item() > 0


def test_input_and_class_guards():
    model = ReinClassQueryResidual(num_classes=4)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 4, 4))
    output = model(torch.randn(1, 4, 4, 4))
    with pytest.raises(IndexError):
        output.counterfactual_logits(4)
