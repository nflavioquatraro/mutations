"""RTR-008 integration with the frozen logical router score-bank contract."""

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


def _logical_router():
    """Build two layers whose ordinary top-1 preferences cross by magnitude."""

    heads = AppendOnlyRouterHeads(feature_size=1)
    for layer_index, code_weight, math_weight in (
        (2, 1.0, -1.0),
        (4, -0.5, 0.5),
    ):
        for expert_id, weight in (
            ("code", code_weight),
            ("math", math_weight),
        ):
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


def test_logical_router_sequence_is_stabilized_without_hidden_state_or_mutation() -> None:
    router = _logical_router()
    features = torch.tensor([[-1.0], [-0.1], [0.1], [1.0]])
    router_before = {
        name: value.clone() for name, value in router.state_dict().items()
    }
    top1 = AnchoredTop1Selection()
    first_scores = router(features, layer_index=2)
    previous = top1(first_scores, expert_ids=router.expert_ids_for_layer(2))
    second_scores = router(features, layer_index=4)
    stabilizer = CrossLayerTop1Stickiness(
        CrossLayerStickinessConfig(switch_margin=0.2)
    )

    first_result = stabilizer(
        second_scores,
        expert_ids=router.expert_ids_for_layer(4),
        previous_routes=previous,
    )
    repeated_result = stabilizer(
        second_scores,
        expert_ids=router.expert_ids_for_layer(4),
        previous_routes=previous,
    )

    assert torch.equal(
        previous.selected_indices,
        torch.tensor([2, 2, 1, 1]),
    )
    assert torch.equal(
        first_result.baseline_routes.selected_indices,
        torch.tensor([1, 1, 2, 2]),
    )
    assert torch.equal(
        first_result.routes.selected_indices,
        torch.tensor([1, 2, 1, 2]),
    )
    assert torch.equal(
        first_result.stabilized_mask,
        torch.tensor([False, True, True, False]),
    )
    assert torch.equal(
        first_result.switch_mask,
        torch.tensor([True, False, False, True]),
    )
    assert torch.equal(
        repeated_result.routes.selected_indices,
        first_result.routes.selected_indices,
    )
    assert torch.equal(
        repeated_result.routes.selected_scores,
        first_result.routes.selected_scores,
    )
    assert router.training is False
    assert tuple(router.parameters()) == ()
    assert all(
        torch.equal(value, router_before[name])
        for name, value in router.state_dict().items()
    )
