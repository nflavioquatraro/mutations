"""Verify RTR-005/POL-004 package gating and package-local top-2 routing."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    HierarchicalPackageTop1LocalTop2Selection,
    HierarchicalRouterScoreBank,
    compile_logical_router,
)


def _logical_router(
    rows: tuple[tuple[str, tuple[float, ...]], ...],
):
    """Compile deterministic affine rows into one frozen RTR-003 bank."""

    heads = AppendOnlyRouterHeads(feature_size=len(rows[0][1]))
    for identifier, weight in rows:
        head = AffineRouterHead(
            len(weight),
            layer_index=0,
            expert_id=identifier,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
        heads.append(head)
    return compile_logical_router(heads)


def main() -> None:
    """Run package top-1, selected-package scoring, and local top-2."""

    router = HierarchicalRouterScoreBank(
        _logical_router(
            (("code-package", (1.0, 0.0)), ("math-package", (0.0, 1.0)))
        ),
        {
            "code-package": _logical_router(
                (("code-a", (1.0, 0.0)), ("code-b", (0.0, 1.0)))
            ),
            "math-package": _logical_router(
                (("math-a", (-1.0, 0.0)), ("math-b", (0.0, -1.0)))
            ),
        },
    )
    features = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0], [1.0, 1.0]]
    )
    policy = HierarchicalPackageTop1LocalTop2Selection()
    package_routes = policy.select_packages(
        router.score_packages(features, layer_index=0),
        package_ids=router.package_ids_for_layer(0),
    )
    local_scores = router.score_selected_locals(
        features,
        package_indices=package_routes.selected_indices,
        package_ids=package_routes.expert_ids,
        layer_index=0,
    )
    decision = policy.select_locals(
        package_routes,
        local_scores,
        expert_ids_by_package={
            package_id: router.expert_ids_for_package(
                package_id,
                layer_index=0,
            )
            for package_id in router.package_ids
        },
    )

    assert package_routes.selected_indices.tolist() == [1, 2, 0, 1]
    assert decision.expert_routes.selected_indices.tolist() == [
        [1, 2],
        [3, 4],
        [0, 0],
        [1, 2],
    ]
    assert all(
        2 not in group.flat_token_indices.tolist() for group in local_scores
    )
    print(
        "RTR-005/POL-004 validation passed: "
        f"packages={package_routes.selected_indices.tolist()}, "
        f"experts={decision.expert_routes.selected_indices.tolist()}, "
        "local_scored_tokens="
        f"{sum(group.expert_scores.shape[0] for group in local_scores)}"
    )


if __name__ == "__main__":
    main()
