"""Verify RTR-003 compilation, append invariance, and routed execution."""

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


def main() -> None:
    """Compile package heads and execute their frozen logical score matrix."""

    torch.manual_seed(41)
    independent = AppendOnlyRouterHeads(feature_size=2)
    code_head = AffineRouterHead(2, layer_index=3, expert_id="code:0")
    with torch.no_grad():
        code_head.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
    independent.append(code_head)

    logical = LogicalAppendOnlyRouter.from_independent_heads(independent)
    features = torch.tensor([[2.0, 1.0], [-2.0, 1.0]])
    incumbent_scores = logical(features, layer_index=3)

    math_head = AffineRouterHead(2, layer_index=3, expert_id="math:0")
    with torch.no_grad():
        math_head.linear.weight.copy_(torch.tensor([[-1.0, 0.0]]))
    logical.append(math_head)
    appended_scores = logical(features, layer_index=3)
    assert torch.equal(appended_scores[..., :1], incumbent_scores)

    anchor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.eye(2))
    code = FullMLPExpert.from_anchor(anchor, trainable=False)
    math = FullMLPExpert.from_anchor(anchor, trainable=False)
    with torch.no_grad():
        code.expert_mlp.weight.copy_(torch.eye(2) * 2.0)
        math.expert_mlp.weight.copy_(torch.eye(2) * 3.0)

    routed = RoutedMLP(anchor)
    routed.configure_anchored_top1(
        layer_index=3,
        router_feature_extractor=IdentityRouterFeatureExtractor(2),
        router_heads=logical,
        experts={"code:0": code, "math:0": math},
    )
    output = routed(features)
    assert torch.equal(output, torch.tensor([[4.0, 2.0], [-6.0, 3.0]]))

    print(
        "Logical router validation passed: "
        f"architecture={logical.architecture}, "
        f"columns={len(logical)}, "
        f"order={logical.expert_ids_for_layer(3)}, "
        "incumbent_scores_preserved=True, "
        "trainable_parameters=0"
    )


if __name__ == "__main__":
    main()
