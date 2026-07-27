"""Qwen3 integration tests for opt-in anchored top-1 dispatch."""

from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    RouteTelemetry,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import assert_tensors_equivalent


def test_identical_selected_expert_preserves_qwen3_outputs(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    input_ids = torch.tensor([[1, 7, 11, 3], [2, 5, 13, 9]])
    with torch.no_grad():
        dense_output = tiny_qwen3_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

    Qwen3Backend().convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[3]),
    )
    wrapper = tiny_qwen3_model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)

    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp, trainable=False)
    router_heads = AppendOnlyRouterHeads(feature_size=16)
    head = AffineRouterHead(
        16,
        layer_index=3,
        expert_id="code",
        trainable=False,
    )
    with torch.no_grad():
        head.linear.bias.fill_(1.0)
    router_heads.append(head)
    telemetry = RouteTelemetry(layer_index=3)
    wrapper.configure_anchored_top1(
        layer_index=3,
        router_feature_extractor=IdentityRouterFeatureExtractor(16),
        router_heads=router_heads,
        experts={"code": expert},
        route_telemetry=telemetry,
    )

    with torch.no_grad():
        routed_output = tiny_qwen3_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

    assert_tensors_equivalent(
        routed_output.logits,
        dense_output.logits,
        name="identical-expert logits",
        atol=0.0,
        rtol=0.0,
    )
    for index, (actual, expected) in enumerate(
        zip(
            routed_output.hidden_states,
            dense_output.hidden_states,
            strict=True,
        )
    ):
        assert_tensors_equivalent(
            actual,
            expected,
            name=f"identical-expert hidden_states[{index}]",
            atol=0.0,
            rtol=0.0,
        )
    summary = telemetry.summarize()
    assert summary.token_count == input_ids.numel()
    assert summary.anchor_count == 0
    assert summary.expert_counts == (("code", input_ids.numel()),)


def test_configured_empty_registry_keeps_exact_anchor_path(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    Qwen3Backend().convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[3]),
    )
    wrapper = tiny_qwen3_model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)

    with torch.no_grad():
        forced_anchor = tiny_qwen3_model(input_ids=input_ids).logits

    telemetry = RouteTelemetry(layer_index=3)
    wrapper.configure_anchored_top1(
        layer_index=3,
        router_feature_extractor=IdentityRouterFeatureExtractor(16),
        router_heads=AppendOnlyRouterHeads(feature_size=16),
        experts={},
        route_telemetry=telemetry,
    )
    with torch.no_grad():
        empty_registry = tiny_qwen3_model(input_ids=input_ids).logits

    assert_tensors_equivalent(
        empty_registry,
        forced_anchor,
        name="empty-registry logits",
        atol=0.0,
        rtol=0.0,
    )
    assert telemetry.summarize().anchor_count == input_ids.numel()
