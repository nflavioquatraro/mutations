"""Acceptance tests for adaptive anchor/top-1/top-2 selection."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AdaptiveRouteDecision,
    AdaptiveTop1Top2Selection,
    SelectionPolicy,
)


def test_adaptive_selection_exposes_anchor_top1_and_top2_routes() -> None:
    scores = torch.tensor(
        [
            [-0.5, -0.1, -0.3],
            [0.8, -0.2, -0.3],
            [0.8, 0.25, 0.1],
            [0.8, 0.3, 0.1],
            [0.5, 0.5, 0.2],
            [0.2, 0.9, 0.9],
        ]
    )

    decision = AdaptiveTop1Top2Selection(mix_threshold=0.25)(
        scores,
        expert_ids=("first", "second", "third"),
    )

    assert isinstance(AdaptiveTop1Top2Selection(), SelectionPolicy)
    assert AdaptiveTop1Top2Selection.policy_name == "adaptive_top1_top2"
    assert torch.equal(
        decision.ranked_indices,
        torch.tensor(
            [[2, 3], [1, 2], [1, 2], [1, 2], [1, 2], [2, 3]]
        ),
    )
    assert torch.equal(
        decision.ranked_scores,
        torch.tensor(
            [
                [-0.1, -0.3],
                [0.8, -0.2],
                [0.8, 0.25],
                [0.8, 0.3],
                [0.5, 0.5],
                [0.9, 0.9],
            ]
        ),
    )
    assert torch.equal(
        decision.route_arity,
        torch.tensor([0, 1, 1, 2, 2, 2]),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([True, False, False, False, False, False]),
    )
    assert torch.equal(
        decision.top1_mask,
        torch.tensor([False, True, True, False, False, False]),
    )
    assert torch.equal(
        decision.top2_mask,
        torch.tensor([False, False, False, True, True, True]),
    )
    assert torch.equal(
        decision.mask_for_expert("first"),
        torch.tensor([False, True, True, True, True, False]),
    )
    assert torch.equal(
        decision.mask_for_expert("second"),
        torch.tensor([False, False, False, True, True, True]),
    )
    assert torch.equal(
        decision.mask_for_expert("third"),
        torch.tensor([False, False, False, False, False, True]),
    )


def test_zero_threshold_requires_two_positive_scores_for_top2() -> None:
    decision = AdaptiveTop1Top2Selection()(
        torch.tensor([[0.8, 0.2], [0.8, 0.0], [0.8, -0.1]]),
        expert_ids=("code", "math"),
    )

    assert torch.equal(decision.route_arity, torch.tensor([2, 1, 1]))


def test_empty_and_single_expert_registries_use_safe_padding() -> None:
    empty_scores = torch.empty(2, 3, 0, dtype=torch.float64)
    empty = AdaptiveTop1Top2Selection()(empty_scores, expert_ids=())

    assert empty.ranked_indices.shape == (2, 3, 2)
    assert empty.ranked_scores.shape == (2, 3, 2)
    assert empty.route_arity.shape == (2, 3)
    assert empty.ranked_scores.dtype == torch.float64
    assert torch.equal(
        empty.ranked_indices,
        torch.zeros(2, 3, 2, dtype=torch.long),
    )
    assert torch.equal(
        empty.ranked_scores,
        torch.zeros(2, 3, 2, dtype=torch.float64),
    )
    assert empty.anchor_mask.all()

    single = AdaptiveTop1Top2Selection()(
        torch.tensor([[-0.2], [0.0], [0.4]], dtype=torch.float64),
        expert_ids=("only",),
    )
    assert torch.equal(
        single.ranked_indices,
        torch.tensor([[1, 0], [1, 0], [1, 0]]),
    )
    assert torch.equal(
        single.ranked_scores,
        torch.tensor(
            [[-0.2, 0.0], [0.0, 0.0], [0.4, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(single.route_arity, torch.tensor([0, 0, 1]))
    assert not single.top2_mask.any()


def test_adaptive_selection_is_deterministic_and_does_not_mutate_scores() -> None:
    scores = torch.tensor(
        [[[0.4, 0.2, 0.2], [0.8, 0.8, 0.1]]],
        dtype=torch.float64,
    )
    original = scores.clone()
    policy = AdaptiveTop1Top2Selection(mix_threshold=0.15)

    first = policy.select(scores, expert_ids=["code", "math", "science"])
    second = policy.select(scores, expert_ids=["code", "math", "science"])

    assert torch.equal(scores, original)
    assert torch.equal(first.ranked_indices, second.ranked_indices)
    assert torch.equal(first.ranked_scores, second.ranked_scores)
    assert torch.equal(first.route_arity, second.route_arity)
    assert first.ranked_scores.dtype == scores.dtype
    assert first.ranked_scores.device == scores.device
    assert first.route_arity.device == scores.device
    assert first.expert_ids == second.expert_ids


@pytest.mark.parametrize(
    "threshold",
    [True, "0.1", -0.1, float("nan"), float("inf")],
)
def test_adaptive_selection_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="mix_threshold"):
        AdaptiveTop1Top2Selection(threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 0.1]]),
        torch.tensor([[float("inf"), 0.1]]),
        torch.tensor([[-float("inf"), 0.1]]),
    ],
)
def test_adaptive_selection_rejects_nonfinite_scores(
    scores: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        AdaptiveTop1Top2Selection()(scores, expert_ids=("code", "math"))


def test_adaptive_selection_rejects_invalid_inputs() -> None:
    policy = AdaptiveTop1Top2Selection()

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy([[0.1]], expert_ids=("code",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        policy(torch.ones(2, 1, dtype=torch.int64), expert_ids=("code",))
    with pytest.raises(ValueError, match="final expert"):
        policy(torch.tensor(0.1), expert_ids=())
    with pytest.raises(ValueError, match="must match"):
        policy(torch.randn(2, 3), expert_ids=("code", "math"))
    with pytest.raises(ValueError, match="duplicates"):
        policy(torch.randn(2, 2), expert_ids=("code", "code"))


def test_adaptive_route_decision_rejects_inconsistent_public_construction() -> None:
    with pytest.raises(TypeError, match="torch.long"):
        AdaptiveRouteDecision(
            ranked_indices=torch.tensor([[1.0, 0.0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="token-leading"):
        AdaptiveRouteDecision(
            ranked_indices=torch.tensor([[1, 0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([[1]]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="ranked slot"):
        AdaptiveRouteDecision(
            ranked_indices=torch.tensor([[0, 0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="distinct"):
        AdaptiveRouteDecision(
            ranked_indices=torch.tensor([[1, 1]]),
            ranked_scores=torch.tensor([[0.5, 0.4]]),
            route_arity=torch.tensor([2]),
            expert_ids=("code", "math"),
        )


def test_adaptive_route_decision_rejects_unknown_expert_mask() -> None:
    decision = AdaptiveTop1Top2Selection()(
        torch.tensor([[0.2, 0.3]]),
        expert_ids=("code", "math"),
    )

    with pytest.raises(KeyError, match="Unknown expert_id"):
        decision.mask_for_expert("science")
