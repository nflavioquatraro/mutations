"""Acceptance tests for overflow-to-anchor capacity fallback."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    OverflowToAnchorCapacity,
    Top2CapacityDecision,
    Top2ToTop1Capacity,
)


def test_unavailable_selected_experts_fall_back_to_anchor() -> None:
    routes = AnchoredTop1Selection()(
        torch.tensor([[0.4, -0.2], [0.1, 0.7], [-0.1, -0.2]]),
        expert_ids=("code", "math"),
    )
    availability = torch.tensor(
        [[False, True], [True, False], [False, False]]
    )

    capacity = OverflowToAnchorCapacity()(
        routes,
        expert_availability=availability,
    )

    assert torch.equal(capacity.routes.selected_indices, torch.tensor([0, 0, 0]))
    assert torch.equal(capacity.routes.selected_scores, torch.zeros(3))
    assert torch.equal(
        capacity.overflow_mask,
        torch.tensor([True, True, False]),
    )
    assert capacity.routes.expert_ids == ("code", "math")


def test_available_or_unspecified_capacity_preserves_routes() -> None:
    routes = AnchoredTop1Selection()(
        torch.tensor([[0.4], [-0.1]]),
        expert_ids=("code",),
    )
    policy = OverflowToAnchorCapacity()

    unspecified = policy(routes)
    available = policy(
        routes,
        expert_availability=torch.ones(2, 1, dtype=torch.bool),
    )

    assert unspecified.routes is routes
    assert available.routes is routes
    assert not unspecified.overflow_mask.any()
    assert not available.overflow_mask.any()


def test_empty_registry_capacity_is_well_defined() -> None:
    routes = AnchoredTop1Selection()(
        torch.empty(2, 0),
        expert_ids=(),
    )

    result = OverflowToAnchorCapacity()(
        routes,
        expert_availability=torch.empty(2, 0, dtype=torch.bool),
    )

    assert result.routes is routes
    assert torch.equal(result.overflow_mask, torch.zeros(2, dtype=torch.bool))


def test_capacity_rejects_invalid_availability() -> None:
    routes = AnchoredTop1Selection()(
        torch.tensor([[0.4]]),
        expert_ids=("code",),
    )
    policy = OverflowToAnchorCapacity()

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy(routes, expert_availability=[[True]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.bool"):
        policy(routes, expert_availability=torch.ones(1, 1))
    with pytest.raises(ValueError, match="route shape"):
        policy(
            routes,
            expert_availability=torch.ones(1, 2, dtype=torch.bool),
        )


def test_top2_capacity_degrades_to_survivor_or_anchor() -> None:
    routes = AnchorVetoTop2Selection()(
        torch.tensor(
            [
                [2.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
                [1.0, -0.2],
            ]
        ),
        expert_ids=("code", "math"),
    )
    availability = torch.tensor(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
        ]
    )
    availability_before = availability.clone()

    capacity = Top2ToTop1Capacity()(
        routes,
        expert_availability=availability,
    )

    assert torch.equal(availability, availability_before)
    assert torch.equal(
        capacity.effective_indices,
        torch.tensor([[1, 2], [1, 0], [2, 0], [0, 0], [0, 0]]),
    )
    assert torch.equal(
        capacity.effective_scores,
        torch.tensor(
            [
                [2.0, 1.0],
                [2.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(capacity.route_arity, torch.tensor([2, 1, 1, 0, 0]))
    assert torch.equal(
        capacity.overflow_mask,
        torch.tensor(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        ),
    )
    assert torch.equal(
        capacity.anchor_mask,
        torch.tensor([False, False, False, True, True]),
    )
    assert torch.equal(
        capacity.top1_mask,
        torch.tensor([False, True, True, False, False]),
    )
    assert torch.equal(
        capacity.top2_mask,
        torch.tensor([True, False, False, False, False]),
    )
    assert torch.equal(
        capacity.mask_for_expert("math"),
        torch.tensor([True, False, True, False, False]),
    )


def test_top2_capacity_without_availability_preserves_approved_pairs() -> None:
    routes = AnchorVetoTop2Selection()(
        torch.tensor([[2.0, 1.0], [1.0, -0.2]]),
        expert_ids=("code", "math"),
    )

    capacity = Top2ToTop1Capacity()(routes)

    assert torch.equal(capacity.route_arity, torch.tensor([2, 0]))
    assert torch.equal(
        capacity.effective_indices,
        torch.tensor([[1, 2], [0, 0]]),
    )
    assert not capacity.overflow_mask.any()


def test_top2_capacity_validates_availability_before_routing() -> None:
    routes = AnchorVetoTop2Selection()(
        torch.tensor([[2.0, 1.0]]),
        expert_ids=("code", "math"),
    )
    policy = Top2ToTop1Capacity()

    with pytest.raises(TypeError, match="torch.Tensor"):
        policy(routes, expert_availability=[[True, True]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.bool"):
        policy(routes, expert_availability=torch.ones(1, 2))
    with pytest.raises(ValueError, match="token shape"):
        policy(
            routes,
            expert_availability=torch.ones(1, 3, dtype=torch.bool),
        )


def test_top2_capacity_result_rejects_active_anchor_padding() -> None:
    with pytest.raises(ValueError, match="Active capacity routes"):
        Top2CapacityDecision(
            effective_indices=torch.tensor([[0, 0]]),
            effective_scores=torch.tensor([[1.0, 0.0]]),
            route_arity=torch.tensor([1]),
            expert_ids=("code",),
            overflow_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
