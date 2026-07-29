"""Mathematical route-selection tests for POL-004."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    HierarchicalLocalScores,
    HierarchicalPackageTop1LocalTop2Selection,
    HierarchicalSelectionError,
)


def _selected_fixture() -> tuple[
    HierarchicalPackageTop1LocalTop2Selection,
    object,
    tuple[HierarchicalLocalScores, ...],
    dict[str, tuple[str, ...]],
]:
    """Return staged package and local scores with an anchor row."""

    policy = HierarchicalPackageTop1LocalTop2Selection()
    package_scores = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0], [1.0, 1.0]]
    )
    package_routes = policy.select_packages(
        package_scores,
        package_ids=("code-package", "math-package"),
    )
    topology = {
        "code-package": ("code-a", "code-b", "code-c"),
        "math-package": ("math-a", "math-b"),
    }
    groups = (
        HierarchicalLocalScores(
            package_id="code-package",
            flat_token_indices=torch.tensor([0, 3]),
            expert_scores=torch.tensor(
                [[-2.0, -1.0, -3.0], [0.5, 0.5, 0.1]]
            ),
            expert_ids=topology["code-package"],
        ),
        HierarchicalLocalScores(
            package_id="math-package",
            flat_token_indices=torch.tensor([1]),
            expert_scores=torch.tensor([[-4.0, -2.0]]),
            expert_ids=topology["math-package"],
        ),
    )
    return policy, package_routes, groups, topology


def test_package_gate_and_unconditional_local_top2_are_deterministic() -> None:
    policy, package_routes, groups, topology = _selected_fixture()

    decision = policy.select_locals(
        package_routes,
        groups,
        expert_ids_by_package=topology,
    )

    # Package ties use package order, while both-negative local scores still
    # choose a pair because the anchor was already resolved at stage one.
    assert torch.equal(
        decision.package_routes.selected_indices,
        torch.tensor([1, 2, 0, 1]),
    )
    assert decision.expert_routes.expert_ids == (
        "code-a",
        "code-b",
        "code-c",
        "math-a",
        "math-b",
    )
    assert torch.equal(
        decision.expert_routes.selected_indices,
        torch.tensor([[2, 1], [5, 4], [0, 0], [1, 2]]),
    )
    assert torch.equal(
        decision.expert_routes.selected_scores,
        torch.tensor([[-1.0, -2.0], [-2.0, -4.0], [0.0, 0.0], [0.5, 0.5]]),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([False, False, True, False]),
    )
    assert decision.expert_package_ids == (
        "code-package",
        "code-package",
        "code-package",
        "math-package",
        "math-package",
    )


def test_local_groups_must_exactly_cover_active_package_rows() -> None:
    policy, package_routes, groups, topology = _selected_fixture()

    with pytest.raises(HierarchicalSelectionError, match="missing local scores"):
        policy.select_locals(
            package_routes,
            groups[:1],
            expert_ids_by_package=topology,
        )

    wrong_rows = (
        HierarchicalLocalScores(
            package_id="code-package",
            flat_token_indices=torch.tensor([0, 2]),
            expert_scores=groups[0].expert_scores,
            expert_ids=groups[0].expert_ids,
        ),
        groups[1],
    )
    with pytest.raises(HierarchicalSelectionError, match="do not exactly match"):
        policy.select_locals(
            package_routes,
            wrong_rows,
            expert_ids_by_package=topology,
        )


def test_topology_rejects_cross_package_expert_identity() -> None:
    policy, package_routes, groups, _ = _selected_fixture()
    duplicate_topology = {
        "code-package": ("shared", "code-b"),
        "math-package": ("shared", "math-b"),
    }

    with pytest.raises(HierarchicalSelectionError, match="shared by packages"):
        policy.select_locals(
            package_routes,
            groups,
            expert_ids_by_package=duplicate_topology,
        )


def test_selected_local_scores_preserve_gradients_only_for_ranked_pair() -> None:
    policy = HierarchicalPackageTop1LocalTop2Selection()
    package_routes = policy.select_packages(
        torch.tensor([[2.0]]),
        package_ids=("code-package",),
    )
    local_values = torch.tensor(
        [[0.5, 2.0, 1.0]],
        requires_grad=True,
    )
    decision = policy.select_locals(
        package_routes,
        (
            HierarchicalLocalScores(
                package_id="code-package",
                flat_token_indices=torch.tensor([0]),
                expert_scores=local_values,
                expert_ids=("code-a", "code-b", "code-c"),
            ),
        ),
        expert_ids_by_package={
            "code-package": ("code-a", "code-b", "code-c")
        },
    )

    decision.expert_routes.selected_scores.sum().backward()

    assert torch.equal(local_values.grad, torch.tensor([[0.0, 1.0, 1.0]]))


def test_all_anchor_package_routes_require_no_local_groups() -> None:
    policy = HierarchicalPackageTop1LocalTop2Selection()
    package_routes = policy.select_packages(
        torch.zeros((2, 1)),
        package_ids=("code-package",),
    )

    decision = policy.select_locals(
        package_routes,
        (),
        expert_ids_by_package={"code-package": ("code-a", "code-b")},
    )

    assert torch.equal(
        decision.expert_routes.selected_indices,
        torch.zeros((2, 2), dtype=torch.long),
    )
    assert decision.anchor_mask.all().item()
