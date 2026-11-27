import pytest
import torch

from causalq.models.scs import SCSConfig, SCSInjector, scs_auxiliary_loss


def test_scs_default_parameter_count_and_layer_validation():
    config = SCSConfig()
    model = SCSInjector(config)
    assert sum(p.numel() for p in model.parameters()) == config.parameter_count == 1_194_240
    with pytest.raises(ValueError, match="one-based"):
        SCSConfig(selected_layers=(0, 4))


def test_scs_zero_output_then_task_gradient_and_padding():
    torch.manual_seed(12)
    model = SCSInjector(SCSConfig(16, 8, 3, 2, 2, (2, 4)))
    patches = torch.randn(2, 6, 16)
    valid = torch.tensor([[True] * 5 + [False]] * 2)
    first, _ = model(2, patches, patch_valid=valid)
    after, aux = model(4, first, patch_valid=valid, return_aux=True)
    torch.testing.assert_close(after, patches, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters())
    (after.square().mean() + aux.square().mean()).backward()
    assert all(bridge.up.weight.grad.norm() > 0 for bridge in model.bridges.values())
    assert model.query_bank.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    first, _ = model(2, patches, patch_valid=valid)
    after, _ = model(4, first, patch_valid=valid)
    after.square().mean().backward()
    assert model.bridges["2"].down.weight.grad.norm() > 0
    assert model.query_bank.grad.norm() > 0
    assert model.write.in_proj_weight.grad.norm() > 0
    torch.testing.assert_close(after[~valid], patches[~valid], rtol=0, atol=0)


def test_scs_auxiliary_ignore_and_resize():
    logits = torch.randn(1, 3, 2, 3, requires_grad=True)
    labels = torch.full((1, 4, 6), 255, dtype=torch.long)
    loss = scs_auxiliary_loss(logits, labels)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0
    labels[0, 0, 1] = 2
    expected = -torch.nn.functional.interpolate(
        logits, (4, 6), mode="bilinear", align_corners=False
    ).log_softmax(dim=1)[0, 2, 0, 1]
    torch.testing.assert_close(scs_auxiliary_loss(logits, labels), expected)
