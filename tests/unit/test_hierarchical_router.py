"""Topology and selected-row scoring tests for RTR-005."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    HierarchicalRouterError,
    HierarchicalRouterScoreBank,
    LogicalAppendOnlyRouter,
)


def _logical_router(
    heads: tuple[tuple[int, str, tuple[float, ...], float], ...],
) -> LogicalAppendOnlyRouter:
    """Build one frozen logical score bank with explicit affine rows."""

    feature_size = len(heads[0][2])
    bank = AppendOnlyRouterHeads(feature_size)
    for layer_index, identifier, weight, bias in heads:
        head = AffineRouterHead(
            feature_size,
            layer_index=layer_index,
            expert_id=identifier,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
            head.linear.bias.fill_(bias)
        bank.append(head)
    return LogicalAppendOnlyRouter.from_independent_heads(bank)


def _hierarchy() -> tuple[
    HierarchicalRouterScoreBank,
    LogicalAppendOnlyRouter,
]:
    """Return a two-package hierarchy and its mutable source package bank."""

    package_router = _logical_router(
        (
            (0, "code-package", (1.0, 0.0), 0.0),
            (0, "math-package", (0.0, 1.0), 0.0),
        )
    )
    code_router = _logical_router(
        (
            (0, "code-a", (1.0, 0.0), 0.0),
            (0, "code-b", (0.0, 1.0), 0.0),
        )
    )
    math_router = _logical_router(
        (
            (0, "math-a", (-1.0, 0.0), 0.0),
            (0, "math-b", (0.0, -1.0), 0.0),
        )
    )
    return (
        HierarchicalRouterScoreBank(
            package_router,
            {
                "code-package": code_router,
                "math-package": math_router,
            },
        ),
        package_router,
    )


def test_scores_only_selected_package_rows_and_skips_anchor() -> None:
    hierarchy, _ = _hierarchy()
    features = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0], [1.0, 1.0]]
    )
    package_indices = torch.tensor([1, 2, 0, 1])

    package_scores = hierarchy.score_packages(features, layer_index=0)
    groups = hierarchy.score_selected_locals(
        features,
        package_indices=package_indices,
        package_ids=("code-package", "math-package"),
        layer_index=0,
    )

    assert torch.equal(
        package_scores,
        torch.tensor([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0], [1.0, 1.0]]),
    )
    assert tuple(group.package_id for group in groups) == (
        "code-package",
        "math-package",
    )
    assert torch.equal(groups[0].flat_token_indices, torch.tensor([0, 3]))
    assert torch.equal(groups[0].expert_scores, torch.tensor([[2.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(groups[1].flat_token_indices, torch.tensor([1]))
    assert torch.equal(groups[1].expert_scores, torch.tensor([[0.0, -3.0]]))
    assert all(2 not in group.flat_token_indices.tolist() for group in groups)
    assert hierarchy.global_expert_ids_for_layer(0) == (
        "code-a",
        "code-b",
        "math-a",
        "math-b",
    )
    assert hierarchy.expert_package_ids_for_layer(0) == (
        "code-package",
        "code-package",
        "math-package",
        "math-package",
    )


def test_hierarchy_copies_source_router_state() -> None:
    hierarchy, source_package_router = _hierarchy()
    features = torch.tensor([[2.0, 3.0]])
    before = hierarchy.score_packages(features, layer_index=0)

    with torch.no_grad():
        source_package_router.layers["0"].weight.add_(100.0)
        source_package_router.layers["0"].bias.add_(100.0)

    assert torch.equal(hierarchy.score_packages(features, layer_index=0), before)
    assert not torch.equal(source_package_router(features, layer_index=0), before)
    assert not tuple(hierarchy.parameters())


def test_hierarchy_rejects_nonrectangular_or_ambiguous_topology() -> None:
    package_router = _logical_router(
        (
            (0, "code", (1.0, 0.0), 0.0),
            (0, "math", (0.0, 1.0), 0.0),
            (1, "code", (1.0, 0.0), 0.0),
            (1, "math", (0.0, 1.0), 0.0),
        )
    )
    one_expert = _logical_router(
        (
            (0, "only", (1.0, 0.0), 0.0),
            (1, "only", (1.0, 0.0), 0.0),
        )
    )
    valid_math = _logical_router(
        (
            (0, "shared", (1.0, 0.0), 0.0),
            (0, "math-b", (0.0, 1.0), 0.0),
            (1, "shared", (1.0, 0.0), 0.0),
            (1, "math-b", (0.0, 1.0), 0.0),
        )
    )

    with pytest.raises(HierarchicalRouterError, match="at least two"):
        HierarchicalRouterScoreBank(
            package_router,
            {"code": one_expert, "math": valid_math},
        )

    code_with_shared_id = _logical_router(
        (
            (0, "shared", (1.0, 0.0), 0.0),
            (0, "code-b", (0.0, 1.0), 0.0),
            (1, "shared", (1.0, 0.0), 0.0),
            (1, "code-b", (0.0, 1.0), 0.0),
        )
    )
    with pytest.raises(HierarchicalRouterError, match="shared by packages"):
        HierarchicalRouterScoreBank(
            package_router,
            {"code": code_with_shared_id, "math": valid_math},
        )


def test_selected_local_scoring_validates_exact_identity_and_shape() -> None:
    hierarchy, _ = _hierarchy()
    features = torch.ones((2, 2))

    with pytest.raises(HierarchicalRouterError, match="exact package order"):
        hierarchy.score_selected_locals(
            features,
            package_indices=torch.tensor([1, 2]),
            package_ids=("math-package", "code-package"),
            layer_index=0,
        )
    with pytest.raises(HierarchicalRouterError, match="token-leading"):
        hierarchy.score_selected_locals(
            features,
            package_indices=torch.tensor([[1, 2]]),
            package_ids=hierarchy.package_ids,
            layer_index=0,
        )

