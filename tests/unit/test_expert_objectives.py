"""Deterministic unit tests for TRN-005 expert objectives."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from anchored_moe import (
    ExpertObjectiveConfig,
    ExpertObjectiveError,
    ExpertWeightDrift,
    ReplayKLPreservation,
)


def test_objective_config_validates_weights_temperature_and_fields() -> None:
    config = ExpertObjectiveConfig(
        replay_kl_weight=0.4,
        drift_weight=0.2,
        replay_temperature=2.0,
        teacher_logits_key="teacher",
        replay_token_mask_key="replay",
    )

    assert config.replay_kl_enabled
    assert config.drift_enabled

    with pytest.raises(ValueError, match="replay_kl_weight"):
        ExpertObjectiveConfig(replay_kl_weight=-1.0)
    with pytest.raises(ValueError, match="replay_temperature"):
        ExpertObjectiveConfig(replay_temperature=0.0)
    with pytest.raises(ValueError, match="must differ"):
        ExpertObjectiveConfig(
            teacher_logits_key="same",
            replay_token_mask_key="same",
        )


def test_replay_kl_matches_explicit_masked_float64_calculation() -> None:
    student = torch.tensor(
        [[[1.0, -0.5, 0.2], [0.3, 0.1, -0.7]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    anchor = torch.tensor(
        [[[0.5, 0.2, -0.1], [-0.2, 0.4, 0.7]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[1.0, 0.25]], dtype=torch.float64)
    temperature = 1.7

    actual = ReplayKLPreservation(temperature)(
        student,
        anchor,
        token_mask=mask,
    )
    expected_per_token = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(anchor.detach() / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = (
        (expected_per_token * mask).sum()
        / mask.sum()
        * temperature**2
    )
    actual.backward()

    assert torch.equal(actual, expected)
    assert student.grad is not None
    assert anchor.grad is None


def test_replay_kl_zero_mask_is_differentiable_zero() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    anchor = torch.randn(2, 3, 5)

    loss = ReplayKLPreservation()(
        student,
        anchor,
        token_mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_replay_kl_rejects_shape_device_and_mask_errors() -> None:
    objective = ReplayKLPreservation()
    student = torch.randn(2, 3, 5)
    anchor = torch.randn(2, 3, 5)

    with pytest.raises(ExpertObjectiveError, match="identical shapes"):
        objective(student, anchor[:, :2])
    with pytest.raises(ExpertObjectiveError, match="token_mask shape"):
        objective(student, anchor, token_mask=torch.ones(2, 2))
    with pytest.raises(ExpertObjectiveError, match="non-negative"):
        objective(
            student,
            anchor,
            token_mask=torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
            ),
        )


def test_weight_drift_is_normalized_and_gradients_reach_live_parameters() -> None:
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    objective = ExpertWeightDrift(
        (("first", first), ("second", second))
    )

    assert objective.parameter_names == ("first", "second")
    assert objective.parameter_count == 6
    assert objective().item() == 0.0

    with torch.no_grad():
        first.add_(1.0)
        second[0, 0].sub_(2.0)
    loss = objective()
    loss.backward()

    assert loss.item() == pytest.approx((1.0 + 1.0 + 4.0) / 6.0)
    assert first.grad is not None
    assert second.grad is not None
    assert torch.equal(first.grad, torch.tensor([2.0 / 6.0, 2.0 / 6.0]))
    assert second.grad[0, 0].item() == pytest.approx(-4.0 / 6.0)


def test_weight_drift_snapshots_are_detached_and_validate_names() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    objective = ExpertWeightDrift((("expert.weight", parameter),))

    with torch.no_grad():
        parameter.fill_(4.0)

    assert objective().item() == pytest.approx(9.0)
    with pytest.raises(ExpertObjectiveError, match="Duplicate"):
        ExpertWeightDrift(
            (("duplicate", parameter), ("duplicate", nn.Parameter(torch.ones(1))))
        )
