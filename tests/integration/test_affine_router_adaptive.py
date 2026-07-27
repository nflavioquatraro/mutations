"""Integration of affine router scores with POL-003 adaptive selection."""

from __future__ import annotations

import torch

from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
)


def test_affine_scores_choose_anchor_top1_and_top2_adaptively() -> None:
    hidden_states = torch.tensor(
        [[[1.0, 2.0], [1.0, -1.0], [-1.0, -2.0], [1.0, 0.5]]]
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

    decision = AdaptiveTop1Top2Selection(mix_threshold=0.5)(
        router(features, layer_index=1),
        expert_ids=router.expert_ids_for_layer(1),
    )

    assert decision.expert_ids == ("code", "math")
    assert torch.equal(
        decision.ranked_indices,
        torch.tensor([[[2, 1], [1, 2], [1, 2], [1, 2]]]),
    )
    assert torch.equal(
        decision.route_arity,
        torch.tensor([[2, 1, 0, 1]]),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([[False, False, True, False]]),
    )
    assert torch.equal(
        decision.top1_mask,
        torch.tensor([[False, True, False, True]]),
    )
    assert torch.equal(
        decision.top2_mask,
        torch.tensor([[True, False, False, False]]),
    )
