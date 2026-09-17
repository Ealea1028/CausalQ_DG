from pathlib import Path

import pytest
import torch

from causalq.losses import prediction_consistency_kl
from tools.train import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase8_config_is_prediction_control_only() -> None:
    config = load_config(
        ROOT / "configs/consistency/gta_dinov3l_pred_cons.yaml"
    )

    assert config["experiment"]["phase"] == 8
    assert config["prediction_consistency"]["enabled"] is True
    assert config["prediction_consistency"]["reference_stop_gradient"] is True
    assert config["prediction_consistency"]["valid_pixels_only"] is True
    assert "causal_query_effect" not in config


def test_identical_predictions_have_zero_consistency_loss() -> None:
    logits = torch.randn(2, 4, 3, 5)
    loss = prediction_consistency_kl(logits, logits)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reference_is_stop_gradient_and_view_receives_gradient() -> None:
    view = torch.randn(1, 3, 2, 2, requires_grad=True)
    reference = torch.randn(1, 3, 2, 2, requires_grad=True)

    prediction_consistency_kl(view, reference).backward()

    assert view.grad is not None
    assert torch.isfinite(view.grad).all()
    assert reference.grad is None


def test_ignore_pixels_do_not_affect_prediction_consistency() -> None:
    reference = torch.tensor([[[[3.0, 3.0]], [[0.0, 0.0]]]])
    first = torch.tensor([[[[0.0, 20.0]], [[3.0, -20.0]]]])
    second = first.clone()
    second[:, :, :, 1] = torch.tensor([[-100.0], [100.0]])
    labels = torch.tensor([[[0, 255]]])

    first_loss = prediction_consistency_kl(first, reference, labels=labels)
    second_loss = prediction_consistency_kl(second, reference, labels=labels)

    assert first_loss == pytest.approx(second_loss, abs=1e-6)


def test_prediction_consistency_validates_contract() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        prediction_consistency_kl(
            torch.randn(1, 3, 2, 2),
            torch.randn(1, 3, 3, 2),
        )
    with pytest.raises(ValueError, match="temperature"):
        prediction_consistency_kl(
            torch.randn(1, 3, 2, 2),
            torch.randn(1, 3, 2, 2),
            temperature=0.0,
        )
