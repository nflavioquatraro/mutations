"""Exact route-agreement evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from anchored_moe.policies.selection import RouteDecision


@dataclass(frozen=True)
class RouteAgreement:
    """Exact candidate-assignment agreement for two route decisions.

    Attributes:
        exact:
            Whether every candidate assignment is identical.
        matching_decisions:
            Number of equal token-layer candidate indices.
        total_decisions:
            Total number of compared decisions.
        agreement_fraction:
            ``matching_decisions / total_decisions``. Two empty decisions have
            agreement one.
    """

    exact: bool
    matching_decisions: int
    total_decisions: int
    agreement_fraction: float


def compare_route_decisions(
    actual: RouteDecision,
    expected: RouteDecision,
) -> RouteAgreement:
    """Compare candidate assignments under one identical registry ordering.

    Selected scores are intentionally excluded: route agreement measures
    executed candidates, while score calibration is evaluated separately.

    Raises:
        TypeError: If either value is not a :class:`RouteDecision`.
        ValueError:
            If token shapes or stable expert registry orderings differ.
    """

    if not isinstance(actual, RouteDecision):
        raise TypeError("actual must be a RouteDecision.")
    if not isinstance(expected, RouteDecision):
        raise TypeError("expected must be a RouteDecision.")
    if actual.expert_ids != expected.expert_ids:
        raise ValueError("Route decisions must share identical expert_ids.")
    if actual.selected_indices.shape != expected.selected_indices.shape:
        raise ValueError("Route decisions must share one token shape.")

    actual_indices = actual.selected_indices.detach().to(device="cpu")
    expected_indices = expected.selected_indices.detach().to(device="cpu")
    equal = actual_indices == expected_indices
    matching = int(equal.sum().item())
    total = equal.numel()
    return RouteAgreement(
        exact=bool(equal.all().item()),
        matching_decisions=matching,
        total_decisions=total,
        agreement_fraction=1.0 if total == 0 else matching / total,
    )


def assert_route_decisions_equal(
    actual: RouteDecision,
    expected: RouteDecision,
    *,
    name: str,
) -> RouteAgreement:
    """Assert exact route assignments and return agreement diagnostics.

    Args:
        actual: Decision emitted by the system under evaluation.
        expected: Reference decision under the same expert ordering.
        name: Human-readable route name for assertion diagnostics.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")
    result = compare_route_decisions(actual, expected)
    if not result.exact:
        raise AssertionError(
            f"{name} route decisions differ: "
            f"{result.matching_decisions}/{result.total_decisions} agree "
            f"({result.agreement_fraction:.6f})."
        )
    return result
