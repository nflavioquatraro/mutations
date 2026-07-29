"""Acceptance tests for RTR-008 explicit cross-layer route stickiness."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AnchoredTop1Selection,
    CrossLayerStickinessConfig,
    CrossLayerStickinessError,
    CrossLayerTop1Stickiness,
    RouteDecision,
)


def _routes(
    indices: list[int],
    *,
    expert_ids: tuple[str, ...] = ("code", "math"),
    dtype: torch.dtype = torch.float64,
) -> RouteDecision:
    """Build a finite previous-layer decision for one-dimensional fixtures."""

    return RouteDecision(
        selected_indices=torch.tensor(indices, dtype=torch.long),
        selected_scores=torch.zeros(len(indices), dtype=dtype),
        expert_ids=expert_ids,
    )


def test_config_round_trip_and_validation_are_strict() -> None:
    config = CrossLayerStickinessConfig(switch_margin=0.25)

    assert config.to_dict() == {
        "strategy": "cross_layer_top1_stickiness",
        "switch_margin": 0.25,
    }
    assert CrossLayerStickinessConfig.from_dict(config.to_dict()) == config

    with pytest.raises(TypeError, match="real number"):
        CrossLayerStickinessConfig(switch_margin=True)
    with pytest.raises(ValueError, match="finite"):
        CrossLayerStickinessConfig(switch_margin=float("inf"))
    with pytest.raises(ValueError, match="greater than or equal"):
        CrossLayerStickinessConfig(switch_margin=-0.1)
    with pytest.raises(ValueError, match="fields"):
        CrossLayerStickinessConfig.from_dict(
            {"strategy": config.strategy, "switch_margin": 0.1, "extra": 1}
        )
    with pytest.raises(ValueError, match="Unsupported"):
        CrossLayerStickinessConfig.from_dict(
            {"strategy": "other", "switch_margin": 0.1}
        )


def test_zero_margin_is_exact_anchored_top1_even_for_route_ties() -> None:
    scores = torch.tensor(
        [[0.5, 0.5], [-0.1, -0.2], [0.0, 0.0]],
        dtype=torch.float64,
    )
    scores_before = scores.clone()
    previous = _routes([2, 1, 1])

    result = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.0)
    )(
        scores,
        expert_ids=("code", "math"),
        previous_routes=previous,
    )
    baseline = AnchoredTop1Selection()(
        scores,
        expert_ids=("code", "math"),
    )

    assert torch.equal(result.routes.selected_indices, baseline.selected_indices)
    assert torch.equal(result.routes.selected_scores, baseline.selected_scores)
    assert torch.equal(result.routes.selected_indices, torch.tensor([1, 0, 0]))
    assert not result.stabilized_mask.any()
    assert torch.equal(scores, scores_before)


def test_positive_margin_stabilizes_anchor_and_experts_until_strictly_beaten() -> None:
    scores = torch.tensor(
        [
            [0.2, -1.0],
            [-0.2, -0.5],
            [0.125, 0.375],
            [0.125, 0.5],
            [0.5, 0.125],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    previous = _routes([0, 1, 1, 1, 2])

    result = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.25)
    )(
        scores,
        expert_ids=("code", "math"),
        previous_routes=previous,
    )

    assert torch.equal(
        result.baseline_routes.selected_indices,
        torch.tensor([1, 0, 2, 2, 1]),
    )
    assert torch.equal(
        result.routes.selected_indices,
        torch.tensor([0, 1, 1, 2, 1]),
    )
    assert torch.equal(
        result.routes.selected_scores,
        torch.tensor([0.0, -0.2, 0.125, 0.5, 0.5], dtype=torch.float64),
    )
    assert torch.equal(
        result.retained_previous_mask,
        torch.tensor([True, True, True, False, False]),
    )
    assert torch.equal(
        result.stabilized_mask,
        torch.tensor([True, True, True, False, False]),
    )
    assert torch.equal(
        result.switch_mask,
        torch.tensor([False, False, False, True, True]),
    )
    assert not result.fallback_mask.any()

    result.routes.selected_scores.sum().backward()
    assert torch.equal(
        scores.grad,
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )


def test_previous_expert_ids_realign_and_missing_experts_fall_back() -> None:
    previous = _routes(
        [1, 2, 3, 0],
        expert_ids=("math", "removed", "code"),
    )
    scores = torch.tensor(
        [[0.1, 0.2], [0.9, 0.1], [0.2, 0.19], [-0.1, -0.2]],
        dtype=torch.float64,
    )

    result = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.1)
    )(
        scores,
        expert_ids=("code", "math"),
        previous_routes=previous,
    )

    assert torch.equal(
        result.previous_aligned_indices,
        torch.tensor([2, -1, 1, 0]),
    )
    assert torch.equal(
        result.previous_available_mask,
        torch.tensor([True, False, True, True]),
    )
    assert torch.equal(
        result.routes.selected_indices,
        torch.tensor([2, 1, 1, 0]),
    )
    assert torch.equal(
        result.fallback_mask,
        torch.tensor([False, True, False, False]),
    )


def test_explicit_sequence_boundary_and_empty_registry_use_anchor_baseline() -> None:
    result = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.5)
    )(
        torch.empty(2, 0),
        expert_ids=(),
        previous_routes=None,
    )

    assert result.routes.anchor_mask.all()
    assert result.baseline_routes is result.routes
    assert torch.equal(
        result.previous_aligned_indices,
        torch.tensor([-1, -1]),
    )
    assert result.fallback_mask.all()
    assert not result.retained_previous_mask.any()


def test_incompatible_previous_decisions_fail_before_stabilization() -> None:
    stabilizer = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.1)
    )

    with pytest.raises(TypeError, match="RouteDecision"):
        stabilizer(
            torch.zeros(2, 1),
            expert_ids=("code",),
            previous_routes=torch.zeros(2),  # type: ignore[arg-type]
        )
    with pytest.raises(CrossLayerStickinessError, match="token shape"):
        stabilizer(
            torch.zeros(2, 1),
            expert_ids=("code",),
            previous_routes=_routes([0], expert_ids=("code",)),
        )
