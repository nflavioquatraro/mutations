"""Verify anchor-veto top-2 selection and score-softmax combination."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchorVetoTop2Selection,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    ScoreSoftmaxCombination,
)


def main() -> None:
    """Rank affine scores and mix only tokens approved for two specialists."""

    hidden_states = torch.tensor(
        [[[1.0, 2.0], [1.0, -1.0], [0.0, 0.0], [2.0, 1.0]]]
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

    decision = AnchorVetoTop2Selection()(
        router(features, layer_index=1),
        expert_ids=router.expert_ids_for_layer(1),
    )
    assert torch.equal(
        decision.expert_pair_mask,
        torch.tensor([[True, False, False, True]]),
    )

    # Synthetic outputs stand in for sparse expert execution, which remains a
    # separate feature. Gather them into the selection policy's ranked order.
    expert_outputs = torch.stack(
        (hidden_states + 10.0, hidden_states + 20.0),
        dim=-2,
    )
    approved_outputs = expert_outputs[decision.expert_pair_mask]
    approved_indices = decision.selected_indices[decision.expert_pair_mask] - 1
    ranked_outputs = torch.gather(
        approved_outputs,
        dim=1,
        index=approved_indices.unsqueeze(-1).expand(
            -1,
            -1,
            approved_outputs.shape[-1],
        ),
    )
    mixed = ScoreSoftmaxCombination()(
        ranked_outputs,
        selected_scores=decision.selected_scores[decision.expert_pair_mask],
    )

    assert mixed.output.shape == (2, hidden_states.shape[-1])
    assert torch.allclose(
        mixed.weights.sum(dim=-1),
        torch.ones(2),
    )
    print(
        "Anchor-veto top-2 validation passed: "
        f"selected_pairs={decision.selected_indices.tolist()}, "
        f"mixed_tokens={int(decision.expert_pair_mask.sum().item())}, "
        f"weights={mixed.weights.tolist()}"
    )


if __name__ == "__main__":
    main()
