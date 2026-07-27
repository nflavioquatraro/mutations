"""Verify deterministic anchored hard top-1 selection without model downloads."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
)


def main() -> None:
    """Score synthetic pre-MLP states and verify stable anchor-first routes."""

    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, -1.0], [0.0, 1.0], [0.5, 0.5]]]
    )
    features = IdentityRouterFeatureExtractor(feature_size=2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)

    code = AffineRouterHead(2, layer_index=1, expert_id="code:0")
    math = AffineRouterHead(2, layer_index=1, expert_id="math:0")
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

    # The final token ties at a positive score, so append order selects code.
    expected_indices = torch.tensor([[1, 0, 2, 1]])
    assert torch.equal(decision.selected_indices, expected_indices)
    assert torch.equal(
        decision.anchor_mask,
        torch.tensor([[False, True, False, False]]),
    )

    print(
        "Anchored top-1 validation passed: "
        f"candidate_order=('anchor', {decision.expert_ids[0]!r}, "
        f"{decision.expert_ids[1]!r}), "
        f"selected_indices={decision.selected_indices.tolist()}, "
        f"anchor_tokens={int(decision.anchor_mask.sum().item())}"
    )


if __name__ == "__main__":
    main()
