"""Integration of affine routing, full-MLP experts, and INF-002 dispatch."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop2Dispatcher,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
)


def test_affine_scores_dispatch_and_mix_full_mlp_experts() -> None:
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
        expert_id="code",
        trainable=False,
    )
    math_head = AffineRouterHead(
        2,
        layer_index=1,
        expert_id="math",
        trainable=False,
    )
    with torch.no_grad():
        code_head.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math_head.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router.append(code_head)
    router.append(math_head)

    hidden = torch.tensor([[[2.0, 1.0], [1.0, 2.0], [1.0, -1.0]]])
    scores = router(
        IdentityRouterFeatureExtractor(2)(hidden),
        layer_index=1,
    )
    result = AnchoredTop2Dispatcher().dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=router.expert_ids_for_layer(1),
        anchor_mlp=anchor,
        experts=(code, math),
    )

    first_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    second_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    expected = torch.stack(
        (
            first_weights[0] * (hidden[0, 0] * 2.0)
            + first_weights[1] * (hidden[0, 0] * -1.0),
            second_weights[0] * (hidden[0, 1] * -1.0)
            + second_weights[1] * (hidden[0, 1] * 2.0),
            hidden[0, 2],
        )
    ).unsqueeze(0)
    assert torch.allclose(result.output, expected)
    assert torch.equal(
        result.selected_routes.expert_pair_mask,
        torch.tensor([[True, True, False]]),
    )
    assert torch.equal(
        result.mixture_weights[0, 2],
        torch.zeros(2),
    )
