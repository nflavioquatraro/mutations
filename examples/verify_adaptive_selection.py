"""Verify POL-003 adaptive selection with deterministic affine router heads."""

from __future__ import annotations

import torch

from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
)


def main() -> None:
    """Select anchor, top-1, and top-2 modes from calibrated affine scores."""

    hidden_states = torch.tensor(
        [[[1.0, 2.0], [1.0, -1.0], [-1.0, -2.0], [1.0, 0.5]]]
    )
    features = IdentityRouterFeatureExtractor(2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)
    code = AffineRouterHead(2, layer_index=1, expert_id="code:0")
    math = AffineRouterHead(2, layer_index=1, expert_id="math:0")
    with torch.no_grad():
        code.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router.append(code)
    router.append(math)

    decision = AdaptiveTop1Top2Selection(mix_threshold=0.5)(
        router(features, layer_index=1),
        expert_ids=router.expert_ids_for_layer(1),
    )

    assert torch.equal(
        decision.route_arity,
        torch.tensor([[2, 1, 0, 1]]),
    )
    assert int(decision.anchor_mask.sum().item()) == 1
    assert int(decision.top1_mask.sum().item()) == 2
    assert int(decision.top2_mask.sum().item()) == 1
    print(
        "POL-003 validation passed: "
        f"ranked_indices={decision.ranked_indices.tolist()}, "
        f"route_arity={decision.route_arity.tolist()}"
    )


if __name__ == "__main__":
    main()
