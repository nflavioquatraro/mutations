"""Unit acceptance tests for BAL-004 loss-free selection biases."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from anchored_moe import (
    LOSS_FREE_BIAS_FORMAT_VERSION,
    LossFreeBiasBalancer,
    LossFreeBiasBalancingError,
    LossFreeBiasConfig,
)


def test_selection_biases_preserve_raw_scores_and_gradient_boundary() -> None:
    balancer = LossFreeBiasBalancer(
        ("code:0", "code:1"),
        dtype=torch.float64,
    )
    with torch.no_grad():
        balancer.biases.copy_(torch.tensor([-0.25, 0.5], dtype=torch.float64))
    raw_scores = torch.tensor(
        [[0.75, 0.1], [0.2, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    raw_snapshot = raw_scores.detach().clone()

    result = balancer.apply_for_selection(raw_scores)

    assert result.raw_scores is raw_scores
    assert torch.equal(raw_scores.detach(), raw_snapshot)
    assert torch.allclose(
        result.selection_scores,
        torch.tensor([[0.5, 0.6], [-0.05, 0.3]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.equal(result.biases, balancer.biases)
    assert result.biases.data_ptr() != balancer.biases.data_ptr()
    result.selection_scores.sum().backward()
    assert torch.equal(raw_scores.grad, torch.ones_like(raw_scores))
    assert tuple(balancer.parameters()) == ()
    assert balancer.biases.requires_grad is False
    assert balancer.biases.grad is None


def test_overloaded_bias_decreases_and_anchor_assignments_are_excluded() -> None:
    balancer = LossFreeBiasBalancer(
        ("math:0", "math:1"),
        config=LossFreeBiasConfig(update_rate=0.1),
    )

    report = balancer.update(torch.tensor([1, 1, 1, 2, 0]))

    assert report.assignment_loads == pytest.approx((3.0, 1.0))
    assert report.mean_load == pytest.approx(2.0)
    assert report.load_errors == pytest.approx((-1.0, 1.0))
    assert report.previous_biases == (0.0, 0.0)
    assert report.updated_biases == pytest.approx((-0.1, 0.1))
    assert report.maximum_load_violation == pytest.approx(0.5)
    assert report.update_step == 1
    assert balancer.update_step == 1

    balanced = balancer.update(torch.tensor([1, 2, 0, 0]))
    assert balanced.assignment_loads == (1.0, 1.0)
    assert balanced.load_errors == (0.0, 0.0)
    assert balanced.updated_biases == pytest.approx((-0.1, 0.1))
    assert balanced.maximum_load_violation == 0.0
    assert balanced.update_step == 2


def test_weighted_pair_assignments_exclude_padding_occurrences() -> None:
    balancer = LossFreeBiasBalancer(
        ("a", "b"),
        config=LossFreeBiasConfig(update_rate=0.25),
    )
    assignments = torch.tensor([[1, 2], [1, 0], [2, 0]])
    weights = torch.tensor([[1, 1], [1, 0], [0, 0]])

    report = balancer.update(
        assignments,
        assignment_weights=weights,
    )

    assert report.assignment_loads == (2.0, 1.0)
    assert report.updated_biases == pytest.approx((-0.25, 0.25))


def test_zero_specialist_load_advances_history_without_changing_biases() -> None:
    balancer = LossFreeBiasBalancer(("a", "b"))

    report = balancer.update(torch.zeros((2, 3), dtype=torch.long))

    assert report.assignment_loads == (0.0, 0.0)
    assert report.mean_load == 0.0
    assert report.maximum_load_violation == 0.0
    assert report.updated_biases == (0.0, 0.0)
    assert report.update_step == 1


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda balancer: balancer.apply_for_selection(
                torch.zeros(2, dtype=torch.float64)
            ),
            "dtype",
        ),
        (
            lambda balancer: balancer.apply_for_selection(
                torch.tensor([[0.0, float("nan")]])
            ),
            "finite",
        ),
        (
            lambda balancer: balancer.update(
                torch.tensor([1, 3], dtype=torch.long)
            ),
            "outside",
        ),
        (
            lambda balancer: balancer.update(
                torch.tensor([1, 2], dtype=torch.int32)
            ),
            "torch.long",
        ),
        (
            lambda balancer: balancer.update(
                torch.tensor([1, 2]),
                assignment_weights=torch.tensor([1.0, -1.0]),
            ),
            "non-negative",
        ),
    ],
)
def test_invalid_scores_and_assignments_are_rejected(
    operation,
    message: str,
) -> None:
    with pytest.raises(LossFreeBiasBalancingError, match=message):
        operation(LossFreeBiasBalancer(("a", "b")))


def test_config_and_identifiers_are_strict() -> None:
    config = LossFreeBiasConfig(update_rate=0.025)
    assert LossFreeBiasConfig.from_dict(config.to_dict()) == config
    with pytest.raises(LossFreeBiasBalancingError, match="positive"):
        LossFreeBiasConfig(update_rate=0.0)
    with pytest.raises(LossFreeBiasBalancingError, match="at least two"):
        LossFreeBiasBalancer(("only",))
    with pytest.raises(LossFreeBiasBalancingError, match="unique"):
        LossFreeBiasBalancer(("same", "same"))


def test_safetensors_round_trip_preserves_state_and_rejects_bad_metadata(
    tmp_path,
) -> None:
    balancer = LossFreeBiasBalancer(
        ("code:0", "code:1", "code:2"),
        config=LossFreeBiasConfig(update_rate=0.05),
        dtype=torch.float64,
    )
    balancer.update(torch.tensor([1, 1, 2, 3, 0]))
    destination = tmp_path / "loss-free-bias.safetensors"

    balancer.save_safetensors(destination)
    restored = LossFreeBiasBalancer.from_safetensors(destination)

    assert LOSS_FREE_BIAS_FORMAT_VERSION == 1
    assert restored.expert_ids == balancer.expert_ids
    assert restored.config == balancer.config
    assert restored.update_step == balancer.update_step
    assert torch.equal(restored.biases, balancer.biases)
    assert restored.biases.data_ptr() != balancer.biases.data_ptr()
    assert tuple(restored.parameters()) == ()
    with pytest.raises(FileExistsError):
        balancer.save_safetensors(destination)

    malformed = tmp_path / "malformed.safetensors"
    metadata = {
        "anchored_moe": json.dumps(
            {
                "config": {"update_rate": 0.05},
                "expert_ids": ["code:0", "code:1", "code:2"],
                "format": "wrong",
                "format_version": 999,
                "strategy": "loss_free_selection_bias",
            }
        )
    }
    save_file(
        {
            "biases": torch.zeros(3),
            "update_step": torch.zeros((), dtype=torch.int64),
        },
        malformed,
        metadata=metadata,
    )
    with pytest.raises(LossFreeBiasBalancingError, match="format"):
        LossFreeBiasBalancer.from_safetensors(malformed)
