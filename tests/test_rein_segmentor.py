import pytest
import torch

from tools.check_rein_segmentor import synthetic_label, validate_losses, validate_prediction


def test_synthetic_label_has_all_classes_and_ignore_pixels():
    label = synthetic_label()
    assert label.shape == (1, 512, 512) and label.dtype == torch.long
    assert label.unique().tolist() == list(range(19)) + [255]
    assert (label[:, :16] == 255).all()


def losses():
    stages = ["decode"] + [f"decode.d{i}" for i in range(9)]
    return {f"{stage}.loss_{kind}": torch.tensor(1.0, requires_grad=True)
            for stage in stages for kind in ("cls", "mask", "dice")}


def test_full_loss_contract_is_differentiable():
    values = losses()
    total = validate_losses(values)
    assert total.item() == 30
    total.backward()
    assert all(value.grad.item() == 1 for value in values.values())
    values.pop("decode.loss_cls")
    with pytest.raises(ValueError, match="30"):
        validate_losses(values)


def test_loss_contract_rejects_nan_and_nonscalar_values():
    values = losses()
    values["decode.loss_cls"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError):
        validate_losses(values)
    values["decode.loss_cls"] = torch.ones(2)
    with pytest.raises(FloatingPointError):
        validate_losses(values)


def test_prediction_shape_and_finiteness():
    scores = torch.zeros(1).expand(1, 19, 512, 512)
    validate_prediction(scores)
    with pytest.raises(ValueError):
        validate_prediction(scores[:, :18])
    with pytest.raises(FloatingPointError):
        validate_prediction(torch.tensor(float("nan")).expand_as(scores))
