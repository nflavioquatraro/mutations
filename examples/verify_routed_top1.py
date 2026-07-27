"""Verify routed top-1 execution, overflow fallback, and route telemetry."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    RouteTelemetry,
    RoutedMLP,
)


def main() -> None:
    """Execute synthetic tokens through anchor, expert, and overflow routes."""

    anchor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.eye(2))
    routed_mlp = RoutedMLP(anchor)

    code = FullMLPExpert.from_anchor(anchor, trainable=False)
    math = FullMLPExpert.from_anchor(anchor, trainable=False)
    with torch.no_grad():
        code.expert_mlp.weight.copy_(torch.eye(2) * 2.0)
        math.expert_mlp.weight.copy_(torch.eye(2) * -1.0)

    heads = AppendOnlyRouterHeads(feature_size=2)
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
    heads.append(code_head)
    heads.append(math_head)

    telemetry = RouteTelemetry(layer_index=1)
    routed_mlp.configure_anchored_top1(
        layer_index=1,
        router_feature_extractor=IdentityRouterFeatureExtractor(2),
        router_heads=heads,
        experts={"math:0": math, "code:0": code},
        route_telemetry=telemetry,
    )

    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, -1.0], [0.0, 1.0], [0.5, 0.5]]]
    )
    availability = torch.tensor(
        [[[True, True], [True, True], [True, False], [True, True]]]
    )
    output = routed_mlp(
        hidden_states,
        expert_availability=availability,
    )

    # Routes are code, anchor, math-overflow-to-anchor, and code (tie wins by
    # append order). Every output returns to its original token position.
    expected = torch.tensor(
        [[[2.0, 0.0], [-1.0, -1.0], [0.0, 1.0], [1.0, 1.0]]]
    )
    assert torch.equal(output, expected)
    summary = telemetry.summarize()
    assert summary.anchor_count == 2
    assert summary.expert_counts == (("code:0", 2), ("math:0", 0))
    assert summary.overflow_count == 1

    print(
        "Routed top-1 validation passed: "
        f"tokens={summary.token_count}, anchor={summary.anchor_count}, "
        f"experts={summary.expert_counts}, overflow={summary.overflow_count}"
    )


if __name__ == "__main__":
    main()
