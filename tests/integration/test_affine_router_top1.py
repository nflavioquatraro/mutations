"""Integration of affine scores with anchored hard top-1 selection."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
)


def test_affine_router_scores_feed_anchored_top1_in_append_order() -> None:
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]
    )
    features = IdentityRouterFeatureExtractor(2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)
    code = AffineRouterHead(2, layer_index=1, expert_id="code")
    math = AffineRouterHead(2, layer_index=1, expert_id="math")
    with torch.no_grad():
        code.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router.append(code)
    router.append(math)

    scores = router(features, layer_index=1)
    decision = AnchoredTop1Selection()(
        scores,
        expert_ids=router.expert_ids_for_layer(1),
    )

    assert decision.expert_ids == ("code", "math")
    assert torch.equal(decision.selected_indices, torch.tensor([[1, 0, 2]]))
    assert torch.equal(
        decision.selected_scores,
        torch.tensor([[1.0, 0.0, 1.0]]),
    )
