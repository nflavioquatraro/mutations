"""Mathematical and validation tests for BAL-003."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AuxiliaryLoadBalancingError,
    AuxiliaryLoadBalancingLoss,
    LocalMiniMoEObjectiveReport,
)


def test_balanced_assignments_have_unit_loss_and_gradients() -> None:
    probabilities = torch.tensor(
        [[[0.8, 0.2], [0.2, 0.8]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = AuxiliaryLoadBalancingLoss()(probabilities)

    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad).item() > 0


def test_token_weights_exclude_padding_and_collapse_is_penalized() -> None:
    probabilities = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    token_weights = torch.tensor([[1, 1, 0]])

    loss = AuxiliaryLoadBalancingLoss()(
        probabilities,
        token_weights=token_weights,
    )

    # Both active tokens prefer expert zero, whose mean probability is 0.85.
    assert loss.item() == pytest.approx(2 * 0.85)


def test_zero_token_weight_returns_differentiable_zero() -> None:
    probabilities = torch.tensor(
        [[[0.6, 0.4], [0.4, 0.6]]],
        requires_grad=True,
    )

    loss = AuxiliaryLoadBalancingLoss()(
        probabilities,
        token_weights=torch.zeros((1, 2)),
    )

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(
        probabilities.grad,
        torch.zeros_like(probabilities),
    )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (torch.tensor([[[0.6, 0.5]]]), "sum to one"),
        (torch.tensor([[[1.1, -0.1]]]), "non-negative"),
        (torch.tensor([[[float("nan"), 0.0]]]), "finite"),
    ],
)
def test_invalid_probabilities_are_rejected(
    probabilities: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(AuxiliaryLoadBalancingError, match=message):
        AuxiliaryLoadBalancingLoss()(probabilities)


def test_objective_report_requires_finite_nonnegative_components() -> None:
    report = LocalMiniMoEObjectiveReport(
        specialization_loss=2.0,
        load_balancing_loss=1.0,
        total_loss=2.1,
    )

    assert report.total_loss == 2.1
    with pytest.raises(
        AuxiliaryLoadBalancingError,
        match="non-negative",
    ):
        LocalMiniMoEObjectiveReport(
            specialization_loss=2.0,
            load_balancing_loss=-1.0,
            total_loss=1.9,
        )
