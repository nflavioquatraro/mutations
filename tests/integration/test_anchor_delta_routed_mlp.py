"""Tiny-model integration coverage for COM-004 routed delta execution."""

from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    LoRAExpertConfig,
    PEFTLoRAExpert,
    RoutedMLP,
)


def test_routed_mlp_executes_zero_initialized_lora_as_exact_zero_delta(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp.eval()
    hidden_size = tiny_qwen3_model.config.hidden_size
    hidden_states = torch.randn(2, 3, hidden_size)
    expected = anchor(hidden_states)
    wrapper = RoutedMLP(anchor)
    expert = PEFTLoRAExpert(
        anchor,
        config=LoRAExpertConfig(
            rank=2,
            alpha=4,
            target_modules=("gate_proj", "up_proj", "down_proj"),
        ),
        trainable=False,
    )
    head = AffineRouterHead(
        hidden_size,
        layer_index=2,
        expert_id="code",
        trainable=False,
    )
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(2.0)
    heads = AppendOnlyRouterHeads(hidden_size)
    heads.append(head)

    wrapper.configure_always_on_anchor_deltas(
        layer_index=2,
        router_feature_extractor=IdentityRouterFeatureExtractor(hidden_size),
        router_heads=heads,
        experts={"code": expert},
    )
    actual = wrapper(hidden_states)

    assert wrapper.routing_mode == "always_on_anchor_deltas"
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in anchor.parameters())

    wrapper.reset_to_forced_anchor()
    assert wrapper.routing_mode == "forced_anchor"
    assert torch.equal(wrapper(hidden_states), expected)
