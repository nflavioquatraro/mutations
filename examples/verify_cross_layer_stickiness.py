"""Verify RTR-008 deterministic cross-layer anchored top-1 stickiness."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    CrossLayerStickinessConfig,
    CrossLayerTop1Stickiness,
    compile_logical_router,
)


def _router():
    """Return a frozen two-layer router with controlled preference changes."""

    heads = AppendOnlyRouterHeads(feature_size=1)
    for layer_index, weights in (
        (2, (1.0, -1.0)),
        (4, (-0.5, 0.5)),
    ):
        for expert_id, weight in zip(("code", "math"), weights, strict=True):
            head = AffineRouterHead(
                1,
                layer_index=layer_index,
                expert_id=expert_id,
                trainable=False,
            )
            with torch.no_grad():
                head.linear.weight.fill_(weight)
            heads.append(head)
    return compile_logical_router(heads).eval()


def main() -> None:
    """Run an explicit adjacent-layer route sequence and print its audit data."""

    router = _router()
    features = torch.tensor([[-1.0], [-0.1], [0.1], [1.0]])
    first_scores = router(features, layer_index=2)
    first_routes = AnchoredTop1Selection()(
        first_scores,
        expert_ids=router.expert_ids_for_layer(2),
    )
    second_scores = router(features, layer_index=4)
    stabilizer = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.2)
    )
    result = stabilizer(
        second_scores,
        expert_ids=router.expert_ids_for_layer(4),
        previous_routes=first_routes,
    )

    assert result.baseline_routes.selected_indices.tolist() == [1, 1, 2, 2]
    assert result.routes.selected_indices.tolist() == [1, 2, 1, 2]
    assert result.stabilized_mask.tolist() == [False, True, True, False]
    assert result.switch_mask.tolist() == [True, False, False, True]
    assert tuple(router.parameters()) == ()
    print(
        "RTR-008 validation passed: "
        f"strategy={stabilizer.config.strategy}, "
        f"switch_margin={stabilizer.config.switch_margin}, "
        f"baseline={result.baseline_routes.selected_indices.tolist()}, "
        f"effective={result.routes.selected_indices.tolist()}, "
        f"stabilized_tokens={int(result.stabilized_mask.sum().item())}, "
        "hidden_state=false"
    )


if __name__ == "__main__":
    main()
