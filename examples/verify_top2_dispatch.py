"""Verify standalone capacity-aware sparse anchor-veto top-2 execution."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop2Dispatcher,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    ScoreSoftmaxCombination,
)


def main() -> None:
    """Execute anchor and weighted specialist pairs without model downloads."""

    anchor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.eye(2))
    code = FullMLPExpert.from_anchor(anchor, trainable=False)
    math = FullMLPExpert.from_anchor(anchor, trainable=False)
    with torch.no_grad():
        code.expert_mlp.weight.copy_(torch.eye(2) * 2.0)
        math.expert_mlp.weight.copy_(torch.eye(2) * -1.0)

    router = AppendOnlyRouterHeads(feature_size=2)
    code_head = AffineRouterHead(
        2,
        layer_index=1,
        expert_id="code:0",
        trainable=False,
    )
    math_head = AffineRouterHead(
        2,
        layer_index=1,
        expert_id="math:0",
        trainable=False,
    )
    with torch.no_grad():
        code_head.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math_head.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router.append(code_head)
    router.append(math_head)

    hidden_states = torch.tensor(
        [[[2.0, 1.0], [1.0, 2.0], [1.0, -1.0]]]
    )
    scores = router(
        IdentityRouterFeatureExtractor(2)(hidden_states),
        layer_index=1,
    )
    result = AnchoredTop2Dispatcher(
        combination_policy=ScoreSoftmaxCombination(mix_temperature=1.0)
    ).dispatch(
        hidden_states,
        expert_scores=scores,
        expert_ids=router.expert_ids_for_layer(1),
        anchor_mlp=anchor,
        experts=(code, math),
        expert_availability=torch.tensor(
            [[[True, True], [True, False], [False, False]]]
        ),
    )

    assert torch.equal(
        result.selected_routes.expert_pair_mask,
        torch.tensor([[True, True, False]]),
    )
    assert torch.equal(
        result.capacity.route_arity,
        torch.tensor([[2, 1, 0]]),
    )
    assert torch.equal(
        result.capacity.effective_indices,
        torch.tensor([[[1, 2], [1, 0], [0, 0]]]),
    )
    assert torch.equal(result.output[0, 1], hidden_states[0, 1] * 2.0)
    assert torch.equal(result.output[0, 2], hidden_states[0, 2])
    assert torch.allclose(
        result.mixture_weights[result.capacity.top2_mask].sum(-1),
        torch.ones(1),
    )
    assert torch.equal(result.mixture_weights[0, 1], torch.tensor([1.0, 0.0]))
    print(
        "Capacity-aware top-2 dispatch validation passed: "
        f"selected_pairs={result.selected_routes.selected_indices.tolist()}, "
        f"effective_arity={result.capacity.route_arity.tolist()}, "
        f"weights={result.mixture_weights.tolist()}, "
        f"output={result.output.tolist()}"
    )


if __name__ == "__main__":
    main()
