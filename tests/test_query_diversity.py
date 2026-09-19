from pathlib import Path

import pytest
import torch

from causalq.losses import query_diversity_loss
from tools.train import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_orthogonal_query_residuals_have_zero_diversity_loss() -> None:
    residuals = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)

    assert query_diversity_loss(residuals).item() == pytest.approx(0.0)


def test_duplicated_query_residuals_have_unit_diversity_loss() -> None:
    residuals = torch.ones(4, 3, 8)

    assert query_diversity_loss(residuals).item() == pytest.approx(1.0)


def test_diversity_loss_is_scale_invariant_and_backpropagates() -> None:
    residuals = torch.randn(2, 3, 5, requires_grad=True)
    scales = torch.tensor([0.5, 2.0, 7.0]).view(1, 3, 1)

    first = query_diversity_loss(residuals)
    second = query_diversity_loss(residuals * scales)
    first.backward()

    assert torch.allclose(first, second, atol=1e-6)
    assert residuals.grad is not None
    assert torch.isfinite(residuals.grad).all()


def test_single_query_has_connected_zero_loss() -> None:
    residuals = torch.randn(2, 1, 4, requires_grad=True)
    loss = query_diversity_loss(residuals)

    loss.backward()

    assert loss.item() == 0.0
    assert residuals.grad is not None


def test_phase10_config_enables_only_the_fixed_full_objective() -> None:
    config = load_config(ROOT / "configs/full/gta_dinov3l_full.yaml")

    assert config["experiment"]["phase"] == 10
    assert config["prediction_consistency"]["enabled"] is True
    assert config["causal_query_effect"]["enabled"] is True
    assert config["query_diversity"] == {
        "enabled": True,
        "lambda_div": 0.01,
        "target": "query_residuals",
        "loss": "off_diagonal_cosine_squared",
    }
