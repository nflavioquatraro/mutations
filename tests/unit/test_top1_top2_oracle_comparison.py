"""Acceptance tests for EVAL-004 top-1/top-2 oracle comparison."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    CounterfactualRoute,
    CounterfactualRouteLosses,
    RouteDecision,
    Top2RouteDecision,
    compare_top1_top2_to_oracle,
)


def _table(*, include_pair: bool = True) -> CounterfactualRouteLosses:
    """Return deterministic route losses for four evaluated tokens."""

    routes = [
        CounterfactualRoute.anchor(),
        CounterfactualRoute.single("code"),
        CounterfactualRoute.single("math"),
    ]
    losses = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 0.4, 0.6],
            [0.5, 1.0, 1.1],
            [0.7, 0.9, 0.6],
        ]
    )
    if include_pair:
        routes.append(CounterfactualRoute.pair("code", "math"))
        losses = torch.cat(
            (
                losses,
                torch.tensor([[0.5], [0.8], [0.4], [0.6]]),
            ),
            dim=-1,
        )
    return CounterfactualRouteLosses(
        losses,
        expert_ids=("code", "math"),
        routes=routes,
        loss_name="cross_entropy",
    )


def _top1() -> RouteDecision:
    """Return selected anchor/code/anchor/math routes."""

    return RouteDecision(
        selected_indices=torch.tensor([0, 1, 0, 2]),
        selected_scores=torch.tensor([0.0, 2.0, 0.0, 1.5]),
        expert_ids=("code", "math"),
    )


def _top2() -> Top2RouteDecision:
    """Return pair/pair/anchor-veto/pair routes."""

    return Top2RouteDecision(
        selected_indices=torch.tensor([[1, 2], [2, 1], [0, 1], [2, 1]]),
        selected_scores=torch.tensor(
            [[2.0, 1.0], [1.5, 1.0], [0.0, -0.5], [1.5, 1.0]]
        ),
        expert_ids=("code", "math"),
    )


def test_top1_top2_comparison_reports_quality_and_destructive_mixing() -> None:
    result = compare_top1_top2_to_oracle(_table(), _top1(), _top2())

    assert torch.allclose(
        result.top1.selected_losses,
        torch.tensor([1.0, 0.4, 0.5, 0.6]),
    )
    assert torch.allclose(
        result.top2.selected_losses,
        torch.tensor([0.5, 0.8, 0.5, 0.6]),
    )
    assert torch.allclose(
        result.top1.oracle_losses,
        torch.tensor([0.5, 0.4, 0.4, 0.6]),
    )
    assert torch.equal(
        result.top1.oracle_candidate_indices,
        torch.tensor([3, 1, 3, 2]),
    )
    assert torch.allclose(
        result.best_single_losses,
        torch.tensor([2.0, 0.4, 1.0, 0.6]),
    )
    assert torch.allclose(
        result.best_pair_losses,
        torch.tensor([0.5, 0.8, 0.4, 0.6]),
    )
    assert torch.allclose(
        result.second_expert_gain,
        torch.tensor([0.5, -0.4, 0.0, 0.0]),
    )
    assert torch.allclose(
        result.oracle_pair_gain,
        torch.tensor([1.5, -0.4, 0.6, 0.0]),
    )
    assert torch.equal(
        result.destructive_mixture_mask,
        torch.tensor([False, True, False, False]),
    )
    assert result.mean_top1_regret == pytest.approx(0.15)
    assert result.mean_top2_regret == pytest.approx(0.125)
    assert result.mean_second_expert_gain == pytest.approx(1.0 / 30.0)
    assert result.mean_oracle_pair_gain == pytest.approx(0.425)
    assert result.destructive_mixture_fraction == pytest.approx(1.0 / 3.0)


def test_top1_top2_comparison_requires_pair_coverage_and_alignment() -> None:
    with pytest.raises(ValueError, match="at least one sampled pair"):
        compare_top1_top2_to_oracle(_table(include_pair=False), _top1(), _top2())

    missing_pair = CounterfactualRouteLosses(
        torch.ones(4, 6),
        expert_ids=("code", "math", "bio"),
        routes=(
            CounterfactualRoute.anchor(),
            CounterfactualRoute.single("code"),
            CounterfactualRoute.single("math"),
            CounterfactualRoute.single("bio"),
            CounterfactualRoute.pair("code", "bio"),
            CounterfactualRoute.pair("math", "bio"),
        ),
        loss_name="cross_entropy",
    )
    top1 = RouteDecision(
        selected_indices=torch.zeros(4, dtype=torch.long),
        selected_scores=torch.zeros(4),
        expert_ids=("code", "math", "bio"),
    )
    top2 = Top2RouteDecision(
        selected_indices=torch.tensor([[1, 2]] * 4),
        selected_scores=torch.ones(4, 2),
        expert_ids=("code", "math", "bio"),
    )
    with pytest.raises(ValueError, match="not evaluated"):
        compare_top1_top2_to_oracle(missing_pair, top1, top2)

    wrong_order = RouteDecision(
        selected_indices=torch.zeros(4, dtype=torch.long),
        selected_scores=torch.zeros(4),
        expert_ids=("math", "code"),
    )
    with pytest.raises(ValueError, match="share expert_ids"):
        compare_top1_top2_to_oracle(_table(), wrong_order, _top2())
