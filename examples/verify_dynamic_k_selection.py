"""Verify POL-006 confidence-gated dynamic-k selection and compute bounds."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    DynamicKSelection,
    IdentityRouterFeatureExtractor,
)


def main() -> None:
    """Select zero, one, or two specialists from deterministic affine scores."""

    hidden_states = torch.tensor(
        [[[1.0, 2.0], [1.0, -1.0], [-1.0, -2.0], [1.0, 0.5]]]
    )
    features = IdentityRouterFeatureExtractor(2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)
    code = AffineRouterHead(2, layer_index=1, expert_id="code:0")
    math = AffineRouterHead(2, layer_index=1, expert_id="math:0")
    science = AffineRouterHead(2, layer_index=1, expert_id="science:0")
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

    assert torch.equal(
        decision.route_arity,
        torch.tensor([[2, 1, 0, 2]]),
    )
    assert int(decision.anchor_mask.sum().item()) == 1
    assert int(decision.mask_for_arity(1).sum().item()) == 1
    assert int(decision.mask_for_arity(2).sum().item()) == 2
    assert decision.max_experts == 2
    print(
        "POL-006 validation passed: "
        f"ranked_indices={decision.ranked_indices.tolist()}, "
        f"route_arity={decision.route_arity.tolist()}"
    )


if __name__ == "__main__":
    main()
