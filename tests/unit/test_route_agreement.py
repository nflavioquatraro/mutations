"""Acceptance tests for exact route-agreement evaluation."""

from __future__ import annotations

import pytest
import torch

from anchored_moe.evaluation import (
    assert_route_decisions_equal,
    compare_route_decisions,
)
from anchored_moe.policies import RouteDecision


def _decision(indices: list[int]) -> RouteDecision:
    return RouteDecision(
        selected_indices=torch.tensor(indices),
        selected_scores=torch.zeros(len(indices)),
        expert_ids=("code",),
    )


def test_route_agreement_reports_exact_and_partial_matches() -> None:
    expected = _decision([0, 1, 1, 0])

    exact = compare_route_decisions(_decision([0, 1, 1, 0]), expected)
    partial = compare_route_decisions(_decision([0, 0, 1, 1]), expected)

    assert exact.exact
    assert exact.agreement_fraction == 1.0
    assert partial.exact is False
    assert partial.matching_decisions == 2
    assert partial.total_decisions == 4
    assert partial.agreement_fraction == 0.5


def test_route_agreement_assertion_has_diagnostics() -> None:
    with pytest.raises(AssertionError, match="1/2"):
        assert_route_decisions_equal(
            _decision([0, 0]),
            _decision([0, 1]),
            name="layer-3",
        )


def test_route_agreement_requires_identical_registry_order() -> None:
    actual = _decision([0])
    expected = RouteDecision(
        selected_indices=torch.tensor([0]),
        selected_scores=torch.tensor([0.0]),
        expert_ids=("math",),
    )

    with pytest.raises(ValueError, match="expert_ids"):
        compare_route_decisions(actual, expected)
