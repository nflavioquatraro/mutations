"""Acceptance tests for confidence-gated, budget-bounded dynamic-k selection."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    DynamicKRouteDecision,
    DynamicKSelection,
    SelectionPolicy,
)


def test_dynamic_k_selects_variable_arity_under_a_hard_budget() -> None:
    scores = torch.tensor(
        [
            [-0.5, -0.1, -0.3, -0.4],
            [0.8, 0.2, -0.3, 0.1],
            [0.8, 0.3, 0.26, 0.1],
            [0.8, 0.3, 0.25, 0.1],
            [0.5, 0.5, 0.7, 0.6],
        ]
    )

    decision = DynamicKSelection(
        max_experts=3,
        activation_threshold=0.25,
    )(
        scores,
        expert_ids=("first", "second", "third", "fourth"),
    )

    assert isinstance(DynamicKSelection(max_experts=1), SelectionPolicy)
    assert DynamicKSelection.policy_name == "dynamic_k"
    assert decision.max_experts == 3
    assert torch.equal(
        decision.ranked_indices,
        torch.tensor(
            [
                [2, 3, 4],
                [1, 2, 4],
                [1, 2, 3],
                [1, 2, 3],
                [3, 4, 1],
            ]
        ),
    )
    assert torch.equal(
        decision.ranked_scores,
        torch.tensor(
            [
                [-0.1, -0.3, -0.4],
                [0.8, 0.2, 0.1],
                [0.8, 0.3, 0.26],
                [0.8, 0.3, 0.25],
                [0.7, 0.6, 0.5],
            ]
        ),
    )
    assert torch.equal(decision.route_arity, torch.tensor([0, 1, 3, 2, 3]))
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([True, False, False, False, False]),
    )
    assert torch.equal(
        decision.specialist_mask,
        torch.tensor([False, True, True, True, True]),
    )
    assert torch.equal(
        decision.mask_for_arity(3),
        torch.tensor([False, False, True, False, True]),
    )
    assert torch.equal(
        decision.mask_for_expert("first"),
        torch.tensor([False, True, True, True, True]),
    )
    assert torch.equal(
        decision.mask_for_expert("third"),
        torch.tensor([False, False, True, False, True]),
    )


def test_compute_budget_retains_only_the_globally_strongest_specialists() -> None:
    decision = DynamicKSelection(max_experts=2)(
        torch.tensor([[0.4, 0.8, 0.6, 0.7]]),
        expert_ids=("first", "second", "third", "fourth"),
    )

    assert torch.equal(decision.ranked_indices, torch.tensor([[2, 4]]))
    assert torch.equal(decision.ranked_scores, torch.tensor([[0.8, 0.7]]))
    assert torch.equal(decision.route_arity, torch.tensor([2]))
    assert not decision.mask_for_expert("third").any()


def test_zero_threshold_requires_strictly_positive_specialist_scores() -> None:
    decision = DynamicKSelection(max_experts=3)(
        torch.tensor(
            [
                [0.8, 0.0, -0.1],
                [0.0, 0.0, -0.1],
                [0.8, 0.1, 0.2],
            ]
        ),
        expert_ids=("code", "math", "science"),
    )

    assert torch.equal(decision.route_arity, torch.tensor([1, 0, 3]))


def test_empty_and_small_registries_use_inactive_padding() -> None:
    empty_scores = torch.empty(2, 0, dtype=torch.float64)
    empty = DynamicKSelection(max_experts=3)(
        empty_scores,
        expert_ids=(),
    )

    assert empty.ranked_indices.shape == (2, 3)
    assert empty.ranked_scores.shape == (2, 3)
    assert empty.ranked_scores.dtype == torch.float64
    assert torch.equal(empty.ranked_indices, torch.zeros(2, 3, dtype=torch.long))
    assert torch.equal(
        empty.ranked_scores,
        torch.zeros(2, 3, dtype=torch.float64),
    )
    assert torch.equal(empty.route_arity, torch.zeros(2, dtype=torch.long))
    assert empty.anchor_mask.all()

    single = DynamicKSelection(
        max_experts=3,
        activation_threshold=0.2,
    )(
        torch.tensor([[-0.1], [0.2], [0.3]], dtype=torch.float64),
        expert_ids=("only",),
    )
    assert torch.equal(
        single.ranked_indices,
        torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
    )
    assert torch.equal(
        single.ranked_scores,
        torch.tensor(
            [[-0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(single.route_arity, torch.tensor([0, 0, 1]))


def test_dynamic_k_is_deterministic_differentiable_and_does_not_mutate_scores() -> None:
    scores = torch.tensor(
        [[[0.4, 0.8, 0.8], [0.7, 0.2, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    original = scores.detach().clone()
    policy = DynamicKSelection(max_experts=2, activation_threshold=0.3)

    first = policy(scores, expert_ids=["code", "math", "science"])
    second = policy(scores, expert_ids=["code", "math", "science"])

    assert torch.equal(scores.detach(), original)
    assert torch.equal(first.ranked_indices, second.ranked_indices)
    assert torch.equal(first.ranked_scores, second.ranked_scores)
    assert torch.equal(first.route_arity, second.route_arity)
    assert torch.equal(first.ranked_indices, torch.tensor([[[2, 3], [1, 2]]]))
    assert first.ranked_scores.dtype == scores.dtype
    assert first.ranked_scores.device == scores.device
    assert first.route_arity.device == scores.device
    assert first.expert_ids == ("code", "math", "science")

    first.ranked_scores.sum().backward()
    assert torch.equal(
        scores.grad,
        torch.tensor(
            [[[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("max_experts", [True, 1.5, "2", 0, -1])
def test_dynamic_k_rejects_invalid_compute_budgets(max_experts: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_experts"):
        DynamicKSelection(max_experts=max_experts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "threshold",
    [True, "0.1", -0.1, float("nan"), float("inf")],
)
def test_dynamic_k_rejects_invalid_activation_thresholds(
    threshold: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="activation_threshold"):
        DynamicKSelection(
            max_experts=2,
            activation_threshold=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 0.1]]),
        torch.tensor([[float("inf"), 0.1]]),
        torch.tensor([[-float("inf"), 0.1]]),
    ],
)
def test_dynamic_k_rejects_nonfinite_scores(scores: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite"):
        DynamicKSelection(max_experts=2)(
            scores,
            expert_ids=("code", "math"),
        )


def test_dynamic_k_rejects_invalid_selection_inputs() -> None:
    policy = DynamicKSelection(max_experts=2)

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


def test_dynamic_k_decision_rejects_inconsistent_public_construction() -> None:
    with pytest.raises(TypeError, match="torch.long"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1.0, 0.0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="token-leading"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1, 0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([[1]]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="at least one"):
        DynamicKRouteDecision(
            ranked_indices=torch.empty(2, 0, dtype=torch.long),
            ranked_scores=torch.empty(2, 0),
            route_arity=torch.zeros(2, dtype=torch.long),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="padding index zero"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1, 1]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="padding score zero"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1, 0]]),
            ranked_scores=torch.tensor([[0.5, 0.1]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
        )
    with pytest.raises(ValueError, match="distinct"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1, 1]]),
            ranked_scores=torch.tensor([[0.5, 0.4]]),
            route_arity=torch.tensor([2]),
            expert_ids=("code", "math"),
        )
    with pytest.raises(ValueError, match="retained specialists"):
        DynamicKRouteDecision(
            ranked_indices=torch.tensor([[1, 0]]),
            ranked_scores=torch.tensor([[0.5, 0.0]]),
            route_arity=torch.tensor([2]),
            expert_ids=("code",),
        )


def test_dynamic_k_decision_validates_mask_queries() -> None:
    decision = DynamicKSelection(max_experts=2)(
        torch.tensor([[0.2, 0.3]]),
        expert_ids=("code", "math"),
    )

    with pytest.raises(KeyError, match="Unknown expert_id"):
        decision.mask_for_expert("science")
    with pytest.raises(TypeError, match="arity"):
        decision.mask_for_arity(True)
    with pytest.raises(ValueError, match="max_experts"):
        decision.mask_for_arity(3)
