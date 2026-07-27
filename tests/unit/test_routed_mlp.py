"""Acceptance tests for RoutedMLP top-1 runtime attachment."""

from __future__ import annotations

import pytest
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


def _top1_components() -> tuple[
    RoutedMLP,
    IdentityRouterFeatureExtractor,
    AppendOnlyRouterHeads,
    FullMLPExpert,
]:
    anchor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.eye(2))
    wrapper = RoutedMLP(anchor)
    expert = FullMLPExpert.from_anchor(anchor)
    with torch.no_grad():
        expert.expert_mlp.weight.copy_(torch.eye(2) * 2.0)
    features = IdentityRouterFeatureExtractor(2)
    heads = AppendOnlyRouterHeads(2)
    head = AffineRouterHead(2, layer_index=4, expert_id="code")
    with torch.no_grad():
        head.linear.bias.fill_(1.0)
    heads.append(head)
    return wrapper, features, heads, expert


def test_routed_mlp_attaches_once_freezes_router_and_applies_capacity() -> None:
    wrapper, features, heads, expert = _top1_components()
    telemetry = RouteTelemetry(layer_index=4)

    wrapper.configure_anchored_top1(
        layer_index=4,
        router_feature_extractor=features,
        router_heads=heads,
        experts={"code": expert},
        route_telemetry=telemetry,
    )
    output = wrapper(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        expert_availability=torch.tensor([[False], [True]]),
    )

    assert torch.equal(output, torch.tensor([[1.0, 2.0], [6.0, 8.0]]))
    assert wrapper.routing_mode == "anchored_top1"
    assert wrapper.expert_ids == ("code",)
    assert all(not parameter.requires_grad for parameter in heads.parameters())
    assert expert.trainable_parameter_count == expert.parameter_count
    assert telemetry.summarize().overflow_count == 1

    with pytest.raises(RuntimeError, match="already configured"):
        wrapper.configure_anchored_top1(
            layer_index=4,
            router_feature_extractor=features,
            router_heads=heads,
            experts={"code": expert},
        )


def test_routed_mlp_rejects_expert_and_router_ownership_mismatch() -> None:
    wrapper, features, heads, expert = _top1_components()

    with pytest.raises(ValueError, match="exactly match"):
        wrapper.configure_anchored_top1(
            layer_index=4,
            router_feature_extractor=features,
            router_heads=heads,
            experts={"math": expert},
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
