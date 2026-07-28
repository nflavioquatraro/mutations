"""RTR-003 integration with the existing anchored top-1 runtime."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    LogicalAppendOnlyRouter,
    RoutedMLP,
)


def test_logical_router_drives_sparse_routed_mlp_execution() -> None:
    anchor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.eye(2))
    wrapper = RoutedMLP(anchor)

    code = FullMLPExpert.from_anchor(anchor, trainable=False)
    math = FullMLPExpert.from_anchor(anchor, trainable=False)
    with torch.no_grad():
        code.expert_mlp.weight.copy_(torch.eye(2) * 2.0)
        math.expert_mlp.weight.copy_(torch.eye(2) * 3.0)

    independent = AppendOnlyRouterHeads(2)
    code_head = AffineRouterHead(2, layer_index=4, expert_id="code")
    math_head = AffineRouterHead(2, layer_index=4, expert_id="math")
    with torch.no_grad():
        code_head.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math_head.linear.weight.copy_(torch.tensor([[-1.0, 0.0]]))
    independent.append(code_head)
    independent.append(math_head)
    logical = LogicalAppendOnlyRouter.from_independent_heads(independent)

    wrapper.configure_anchored_top1(
        layer_index=4,
        router_feature_extractor=IdentityRouterFeatureExtractor(2),
        router_heads=logical,
        experts={"code": code, "math": math},
    )
    hidden_states = torch.tensor([[2.0, 1.0], [-2.0, 1.0], [0.0, 1.0]])

    output = wrapper(hidden_states)

    assert wrapper.router_heads is logical
    assert wrapper.expert_ids == ("code", "math")
    assert torch.equal(
        output,
        torch.tensor([[4.0, 2.0], [-6.0, 3.0], [0.0, 1.0]]),
    )
    assert tuple(logical.parameters()) == ()
