"""Acceptance tests for anchored hard top-1 selection."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import AnchoredTop1Selection, RouteDecision


def test_anchored_top1_selects_anchor_or_best_positive_expert() -> None:
    policy = AnchoredTop1Selection()
    scores = torch.tensor(
        [
            [[-0.5, -0.1], [0.3, -0.2], [0.2, 0.7]],
            [[-1.0, 0.1], [0.4, 0.2], [-0.3, -0.4]],
        ]
    )

    decision = policy(scores, expert_ids=("code", "math"))

    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
    )
    assert torch.equal(
        decision.selected_scores,
        torch.tensor([[0.0, 0.3, 0.7], [0.1, 0.4, 0.0]]),
    )
    assert decision.expert_ids == ("code", "math")
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([[True, False, False], [False, False, True]]),
    )
    assert torch.equal(
        decision.mask_for_expert("code"),
        torch.tensor([[False, True, False], [False, True, False]]),
    )
    assert torch.equal(
        decision.mask_for_expert("math"),
        torch.tensor([[False, False, True], [True, False, False]]),
    )
    assert torch.equal(decision.expert_mask, ~decision.anchor_mask)


def test_ties_prefer_anchor_then_earliest_appended_expert() -> None:
    policy = AnchoredTop1Selection()
    scores = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.2],
            [0.1, 0.9, 0.9],
        ]
    )

    decision = policy(
        scores,
        expert_ids=("first", "second", "third"),
    )

    assert torch.equal(decision.selected_indices, torch.tensor([0, 1, 2]))
    assert torch.equal(decision.selected_scores, torch.tensor([0.0, 0.5, 0.9]))


def test_empty_registry_routes_every_token_to_anchor() -> None:
    policy = AnchoredTop1Selection()
    scores = torch.empty(2, 3, 0, dtype=torch.float64)

    decision = policy(scores, expert_ids=())

    assert decision.selected_indices.shape == (2, 3)
    assert decision.selected_scores.shape == (2, 3)
    assert decision.selected_scores.dtype == torch.float64
    assert decision.selected_indices.device == scores.device
    assert decision.selected_scores.device == scores.device
    assert torch.equal(decision.selected_indices, torch.zeros(2, 3, dtype=torch.long))
    assert torch.equal(decision.selected_scores, torch.zeros(2, 3, dtype=torch.float64))
    assert decision.anchor_mask.all()


def test_selection_is_deterministic_and_does_not_mutate_scores() -> None:
    policy = AnchoredTop1Selection()
    scores = torch.tensor([[0.2, -0.1], [0.4, 0.4]], dtype=torch.float64)
    original = scores.clone()

    first = policy.select(scores, expert_ids=["code", "math"])
    second = policy.select(scores, expert_ids=["code", "math"])

    assert torch.equal(scores, original)
    assert torch.equal(first.selected_indices, second.selected_indices)
    assert torch.equal(first.selected_scores, second.selected_scores)
    assert first.expert_ids == second.expert_ids == ("code", "math")


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 0.1]]),
        torch.tensor([[float("inf"), 0.1]]),
        torch.tensor([[-float("inf"), 0.1]]),
    ],
)
def test_selection_rejects_nonfinite_scores(scores: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite"):
        AnchoredTop1Selection()(scores, expert_ids=("code", "math"))


def test_selection_rejects_invalid_score_tensors() -> None:
    policy = AnchoredTop1Selection()

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy([[0.1]], expert_ids=("code",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        policy(torch.ones(2, 1, dtype=torch.int64), expert_ids=("code",))
    with pytest.raises(ValueError, match="final expert"):
        policy(torch.tensor(0.1), expert_ids=())
    with pytest.raises(ValueError, match="must match"):
        policy(torch.randn(2, 3), expert_ids=("code", "math"))


@pytest.mark.parametrize(
    "expert_ids",
    [
        "code",
        ("code", "code"),
        ("", "math"),
        (" code", "math"),
        ("code", 1),
    ],
)
def test_selection_rejects_invalid_expert_ids(expert_ids: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AnchoredTop1Selection()(
            torch.randn(2, 2),
            expert_ids=expert_ids,  # type: ignore[arg-type]
        )


def test_route_decision_rejects_inconsistent_public_construction() -> None:
    with pytest.raises(TypeError, match="torch.long"):
        RouteDecision(
            selected_indices=torch.tensor([0.0]),
            selected_scores=torch.tensor([0.0]),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="same shape"):
        RouteDecision(
            selected_indices=torch.tensor([0]),
            selected_scores=torch.tensor([0.0, 0.0]),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="outside"):
        RouteDecision(
            selected_indices=torch.tensor([2]),
            selected_scores=torch.tensor([0.5]),
            expert_ids=("code",),
        )


def test_route_decision_rejects_unknown_expert_mask() -> None:
    decision = AnchoredTop1Selection()(
        torch.tensor([[0.2]]),
        expert_ids=("code",),
    )

    with pytest.raises(KeyError, match="Unknown expert_id"):
        decision.mask_for_expert("math")
