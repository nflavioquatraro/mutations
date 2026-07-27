"""Acceptance tests for overflow-to-anchor capacity fallback."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import AnchoredTop1Selection, OverflowToAnchorCapacity


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
