"""Acceptance tests for anchor-veto weighted top-2 selection."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import AnchorVetoTop2Selection, Top2RouteDecision


def test_anchor_veto_requires_two_strictly_positive_specialists() -> None:
    scores = torch.tensor(
        [
            [-0.5, -0.1],
            [0.8, -0.2],
            [0.2, 0.7],
            [0.2, 0.0],
            [0.5, 0.5],
        ]
    )

    decision = AnchorVetoTop2Selection()(
        scores,
        expert_ids=("code", "math"),
    )

    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[0, 2], [1, 0], [2, 1], [1, 0], [1, 2]]),
    )
    assert torch.equal(
        decision.selected_scores,
        torch.tensor(
            [[0.0, -0.1], [0.8, 0.0], [0.7, 0.2], [0.2, 0.0], [0.5, 0.5]]
        ),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([True, True, False, True, False]),
    )
    assert torch.equal(decision.expert_pair_mask, ~decision.anchor_mask)
    assert torch.equal(
        decision.mask_for_expert("code"),
        torch.tensor([False, True, True, True, True]),
    )
    assert torch.equal(
        decision.mask_for_expert("math"),
        torch.tensor([True, False, True, False, True]),
    )


def test_top2_ties_prefer_anchor_then_append_order() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.9, 0.9],
        ]
    )

    decision = AnchorVetoTop2Selection()(
        scores,
        expert_ids=("first", "second", "third"),
    )

    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[0, 1], [1, 2], [2, 3]]),
    )
    assert torch.equal(
        decision.selected_scores,
        torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.9, 0.9]]),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([True, False, False]),
    )


def test_empty_and_single_expert_registries_are_always_anchor_vetoed() -> None:
    empty_scores = torch.empty(2, 3, 0, dtype=torch.float64)
    empty = AnchorVetoTop2Selection()(empty_scores, expert_ids=())

    assert empty.selected_indices.shape == (2, 3, 2)
    assert empty.selected_scores.shape == (2, 3, 2)
    assert empty.selected_scores.dtype == torch.float64
    assert torch.equal(
        empty.selected_indices,
        torch.zeros(2, 3, 2, dtype=torch.long),
    )
    assert torch.equal(
        empty.selected_scores,
        torch.zeros(2, 3, 2, dtype=torch.float64),
    )
    assert empty.anchor_mask.all()
    assert not empty.expert_pair_mask.any()

    single = AnchorVetoTop2Selection()(
        torch.tensor([[0.4], [-0.2], [0.0]]),
        expert_ids=("only",),
    )
    assert torch.equal(
        single.selected_indices,
        torch.tensor([[1, 0], [0, 1], [0, 1]]),
    )
    assert single.anchor_mask.all()


def test_top2_selection_is_deterministic_and_does_not_mutate_scores() -> None:
    scores = torch.tensor(
        [[[0.4, 0.2], [0.8, 0.8]]],
        dtype=torch.float64,
    )
    original = scores.clone()
    policy = AnchorVetoTop2Selection()

    first = policy.select(scores, expert_ids=["code", "math"])
    second = policy.select(scores, expert_ids=["code", "math"])

    assert torch.equal(scores, original)
    assert torch.equal(first.selected_indices, second.selected_indices)
    assert torch.equal(first.selected_scores, second.selected_scores)
    assert first.selected_scores.dtype == scores.dtype
    assert first.selected_scores.device == scores.device
    assert first.expert_ids == second.expert_ids == ("code", "math")


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 0.1]]),
        torch.tensor([[float("inf"), 0.1]]),
        torch.tensor([[-float("inf"), 0.1]]),
    ],
)
def test_top2_selection_rejects_nonfinite_scores(scores: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite"):
        AnchorVetoTop2Selection()(scores, expert_ids=("code", "math"))


def test_top2_selection_rejects_invalid_inputs() -> None:
    policy = AnchorVetoTop2Selection()

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


def test_top2_route_decision_rejects_inconsistent_public_construction() -> None:
    with pytest.raises(TypeError, match="torch.long"):
        Top2RouteDecision(
            selected_indices=torch.tensor([[0.0, 0.0]]),
            selected_scores=torch.tensor([[0.0, 0.0]]),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="same shape"):
        Top2RouteDecision(
            selected_indices=torch.tensor([[0, 0]]),
            selected_scores=torch.tensor([0.0, 0.0]),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="size two"):
        Top2RouteDecision(
            selected_indices=torch.tensor([[0, 0, 0]]),
            selected_scores=torch.tensor([[0.0, 0.0, 0.0]]),
            expert_ids=(),
        )
    with pytest.raises(ValueError, match="outside"):
        Top2RouteDecision(
            selected_indices=torch.tensor([[0, 2]]),
            selected_scores=torch.tensor([[0.0, 0.5]]),
            expert_ids=("code",),
        )


def test_top2_route_decision_rejects_unknown_expert_mask() -> None:
    decision = AnchorVetoTop2Selection()(
        torch.tensor([[0.2, 0.3]]),
        expert_ids=("code", "math"),
    )

    with pytest.raises(KeyError, match="Unknown expert_id"):
        decision.mask_for_expert("science")
