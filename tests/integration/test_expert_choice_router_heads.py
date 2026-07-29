"""TRN-003 affine-router score integration for BAL-005."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    ExpertChoiceConfig,
    ExpertChoiceRouting,
)


def test_affine_router_heads_feed_training_prefill_expert_choice() -> None:
    expert_ids = ("local:0", "local:1")
    router_heads = AppendOnlyRouterHeads(feature_size=2)
    first = AffineRouterHead(
        2,
        layer_index=3,
        expert_id=expert_ids[0],
    )
    second = AffineRouterHead(
        2,
        layer_index=3,
        expert_id=expert_ids[1],
    )
    with torch.no_grad():
        first.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        second.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router_heads.append(first)
    router_heads.append(second)

    features = torch.tensor(
        [[[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]]],
        requires_grad=True,
    )
    logits = router_heads(features, layer_index=3)
    decision = ExpertChoiceRouting(
        ExpertChoiceConfig(expert_capacity=1)
    )(
        logits,
        expert_ids=router_heads.expert_ids_for_layer(3),
        token_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert torch.equal(
        decision.selected_token_indices,
        torch.tensor([[0], [2]]),
    )
    assert torch.equal(
        decision.token_assignment_counts,
        torch.tensor([[1, 0, 1, 0]]),
    )
    assert torch.equal(
        decision.anchor_fallback_mask,
        torch.tensor([[False, True, False, True]]),
    )

    decision.selected_gates.sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad).item() > 0
    assert first.linear.weight.grad is not None
    assert second.linear.weight.grad is not None
