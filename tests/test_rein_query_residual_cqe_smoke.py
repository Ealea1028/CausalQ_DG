import torch

from tools.check_rein_query_residual_cqe import matched_objective


def test_matched_objective_reconstructs_no_cqe_control():
    original = torch.tensor(2.0)
    photometric = torch.tensor(4.0)
    cqe = torch.tensor(0.25)
    loss = matched_objective(original, photometric, cqe, enabled=False)
    assert loss.item() == 3.0


def test_matched_objective_adds_only_existing_cqe_term():
    original = torch.tensor(2.0)
    photometric = torch.tensor(4.0)
    cqe = torch.tensor(0.25, requires_grad=True)
    loss = matched_objective(original, photometric, cqe, enabled=True)
    assert loss.item() == 3.25
    loss.backward()
    assert cqe.grad.item() == 1.0


def test_cqe_coefficient_is_applied_once():
    loss = matched_objective(torch.tensor(1.0), torch.tensor(3.0),
                             torch.tensor(0.5), enabled=True, lambda_cqe=2.0)
    assert loss.item() == 3.0
