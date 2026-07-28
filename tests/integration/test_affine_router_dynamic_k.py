"""Integration of affine router scores with POL-006 dynamic-k selection."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    DynamicKSelection,
    IdentityRouterFeatureExtractor,
)


def test_affine_scores_choose_bounded_variable_expert_counts() -> None:
    hidden_states = torch.tensor(
        [[[1.0, 2.0], [1.0, -1.0], [-1.0, -2.0], [1.0, 0.5]]]
    )
    features = IdentityRouterFeatureExtractor(2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)
    code = AffineRouterHead(2, layer_index=1, expert_id="code")
    math = AffineRouterHead(2, layer_index=1, expert_id="math")
    science = AffineRouterHead(2, layer_index=1, expert_id="science")
    with torch.no_grad():
        code.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
        science.linear.weight.copy_(torch.tensor([[0.5, 0.5]]))
    router.append(code)
    router.append(math)
    router.append(science)

    decision = DynamicKSelection(
        max_experts=2,
        activation_threshold=0.5,
    )(
        router(features, layer_index=1),
        expert_ids=router.expert_ids_for_layer(1),
    )

    assert decision.expert_ids == ("code", "math", "science")
    assert torch.equal(
        decision.ranked_indices,
        torch.tensor([[[2, 3], [1, 3], [1, 3], [1, 3]]]),
    )
    assert torch.equal(
        decision.route_arity,
        torch.tensor([[2, 1, 0, 2]]),
    )
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([[False, False, True, False]]),
    )
    assert torch.equal(
        decision.mask_for_arity(1),
        torch.tensor([[False, True, False, False]]),
    )
    assert torch.equal(
        decision.mask_for_arity(2),
        torch.tensor([[True, False, False, True]]),
    )
