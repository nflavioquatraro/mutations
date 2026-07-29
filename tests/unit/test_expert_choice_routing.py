"""Mathematical and validation acceptance tests for BAL-005."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from anchored_moe import (
    ExpertChoiceConfig,
    ExpertChoiceRouting,
    ExpertChoiceRoutingError,
)


def test_experts_select_fixed_buckets_with_variable_token_arity() -> None:
    logits = torch.tensor(
        [
            [[4.0, 4.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 3.0], [9.0, 9.0, 9.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_snapshot = logits.detach().clone()
    token_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )

    decision = ExpertChoiceRouting(
        ExpertChoiceConfig(expert_capacity=1)
    )(
        logits,
        expert_ids=("code:0", "code:1", "code:2"),
        token_mask=token_mask,
    )

    assert decision.expert_ids == ("code:0", "code:1", "code:2")
    assert decision.expert_capacity == 1
    assert decision.token_shape == torch.Size((2, 3))
    assert torch.equal(
        decision.selected_token_indices,
        torch.tensor([[0], [0], [3]]),
    )
    assert torch.equal(
        decision.token_assignment_counts,
        torch.tensor([[2, 0, 0], [1, 0, 0]]),
    )
    assert torch.equal(
        decision.anchor_fallback_mask,
        torch.tensor(
            [[False, True, True], [False, True, False]]
        ),
    )
    assert torch.equal(decision.eligible_token_mask, token_mask)
    assert torch.equal(
        decision.mask_for_expert("code:1"),
        torch.tensor(
            [[True, False, False], [False, False, False]]
        ),
    )
    assert torch.equal(logits.detach(), logits_snapshot)

    decision.selected_gates.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()
    assert torch.count_nonzero(logits.grad).item() > 0


def test_exact_affinity_ties_use_flattened_token_order() -> None:
    logits = torch.zeros((2, 2, 2))

    first = ExpertChoiceRouting(ExpertChoiceConfig(expert_capacity=2))(
        logits,
        expert_ids=("a", "b"),
    )
    second = ExpertChoiceRouting(ExpertChoiceConfig(expert_capacity=2))(
        logits,
        expert_ids=("a", "b"),
    )

    expected_indices = torch.tensor([[0, 1], [0, 1]])
    assert torch.equal(first.selected_token_indices, expected_indices)
    assert torch.equal(second.selected_token_indices, expected_indices)
    assert torch.equal(
        first.token_assignment_counts,
        torch.tensor([[2, 2], [0, 0]]),
    )
    assert torch.equal(
        first.anchor_fallback_mask,
        torch.tensor([[False, False], [True, True]]),
    )
    assert torch.equal(first.selected_gates, torch.full((2, 2), 0.5))


def test_config_round_trip_and_strict_capacity_contract() -> None:
    config = ExpertChoiceConfig(expert_capacity=3)

    assert ExpertChoiceConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ExpertChoiceRoutingError, match="positive integer"):
        ExpertChoiceConfig(expert_capacity=0)
    with pytest.raises(ExpertChoiceRoutingError, match="fields"):
        ExpertChoiceConfig.from_dict(
            {"strategy": "expert_choice", "expert_capacity": 3, "extra": 1}
        )
    with pytest.raises(ExpertChoiceRoutingError, match="Unsupported"):
        ExpertChoiceConfig.from_dict(
            {"strategy": "token_choice", "expert_capacity": 3}
        )

    routing = ExpertChoiceRouting(config)
    with pytest.raises(ExpertChoiceRoutingError, match="cannot exceed"):
        routing(
            torch.zeros((2, 2)),
            expert_ids=("a", "b"),
        )


@pytest.mark.parametrize(
    ("logits", "expert_ids", "token_mask", "message"),
    [
        (torch.zeros(4, 2, dtype=torch.long), ("a", "b"), None, "floating"),
        (
            torch.tensor([[0.0, float("nan")]]),
            ("a", "b"),
            None,
            "finite",
        ),
        (torch.zeros(4, 3), ("a", "b"), None, "width"),
        (torch.zeros(4, 2), ("a", "a"), None, "unique"),
        (torch.zeros(4, 2), ("only",), None, "at least two"),
        (
            torch.zeros(4, 2),
            ("a", "b"),
            torch.ones(4),
            "boolean",
        ),
        (
            torch.zeros(4, 2),
            ("a", "b"),
            torch.ones((2, 2), dtype=torch.bool),
            "token shape",
        ),
    ],
)
def test_invalid_scores_identifiers_and_masks_are_rejected(
    logits: torch.Tensor,
    expert_ids: tuple[str, ...],
    token_mask: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ExpertChoiceRoutingError, match=message):
        ExpertChoiceRouting(ExpertChoiceConfig(expert_capacity=1))(
            logits,
            expert_ids=expert_ids,
            token_mask=token_mask,
        )


def test_decision_rejects_inconsistent_audit_metadata() -> None:
    decision = ExpertChoiceRouting(
        ExpertChoiceConfig(expert_capacity=1)
    )(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        expert_ids=("a", "b"),
    )

    with pytest.raises(
        ExpertChoiceRoutingError,
        match="does not match sparse assignments",
    ):
        replace(
            decision,
            token_assignment_counts=torch.zeros_like(
                decision.token_assignment_counts
            ),
        )
    with pytest.raises(
        ExpertChoiceRoutingError,
        match="mark exactly",
    ):
        replace(
            decision,
            anchor_fallback_mask=torch.zeros_like(
                decision.anchor_fallback_mask
            ),
        )
    with pytest.raises(KeyError, match="Unknown"):
        decision.mask_for_expert("missing")
