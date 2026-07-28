"""Topology and ownership tests for TRN-003 local mini-MoE training."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    ForcedPathTrainingError,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    LocalMiniMoEExpertBinding,
    LocalMiniMoETrainingConfig,
    RoutedMLP,
    prepare_local_mini_moe_training,
)
from anchored_moe.backends import Qwen3Backend


def _components(
    model: Qwen3ForCausalLM,
) -> tuple[
    Qwen3Backend,
    RoutedMLP,
    tuple[LocalMiniMoEExpertBinding, ...],
    IdentityRouterFeatureExtractor,
    AppendOnlyRouterHeads,
]:
    """Build one complete two-expert local package for layer three."""

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    bindings = tuple(
        LocalMiniMoEExpertBinding(
            layer_index=3,
            expert_id=expert_id,
            expert=FullMLPExpert.from_anchor(wrapper.anchor_mlp),
        )
        for expert_id in ("code:0", "code:1")
    )
    features = IdentityRouterFeatureExtractor(16)
    heads = AppendOnlyRouterHeads(16)
    for expert_id in ("code:0", "code:1"):
        heads.append(
            AffineRouterHead(
                16,
                layer_index=3,
                expert_id=expert_id,
            )
        )
    return backend, wrapper, bindings, features, heads


def test_config_round_trip_is_strict() -> None:
    config = LocalMiniMoETrainingConfig(
        mix_temperature=0.75,
        router_bias_epsilon=0.02,
    )

    assert LocalMiniMoETrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="exactly two"):
        LocalMiniMoETrainingConfig.from_dict(
            {**config.to_dict(), "expert_count": 3}
        )


def test_preparation_attaches_exact_pair_and_optimizer_state(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend, wrapper, bindings, features, heads = _components(tiny_qwen3_model)

    topology = prepare_local_mini_moe_training(
        tiny_qwen3_model,
        backend=backend,
        expert_bindings=bindings[::-1],
        router_feature_extractor=features,
        router_heads=heads,
        config=LocalMiniMoETrainingConfig(router_bias_epsilon=0.025),
    )

    assert topology.layer_indices == (3,)
    assert topology.expert_ids == ("code:0", "code:1")
    assert topology.router_head_keys == ((3, "code:0"), (3, "code:1"))
    assert wrapper.routing_mode == "local_mini_moe"
    assert tuple(wrapper.experts) == tuple(
        binding.expert for binding in bindings
    )
    assert heads.heads[0].linear.bias.item() == pytest.approx(0.025)
    assert heads.heads[1].linear.bias.item() == pytest.approx(-0.025)
    assert all(
        not parameter.requires_grad
        for parameter in wrapper.anchor_mlp.parameters()
    )
    trainable_ids = {
        id(parameter)
        for parameter in tiny_qwen3_model.parameters()
        if parameter.requires_grad
    }
    expected_ids = {
        id(parameter)
        for module in [*(binding.expert for binding in bindings), heads]
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert trainable_ids == expected_ids


def test_nonzero_router_fails_before_model_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend, wrapper, bindings, features, heads = _components(tiny_qwen3_model)
    with torch.no_grad():
        heads.heads[0].linear.weight[0, 0] = 1.0

    with pytest.raises(ForcedPathTrainingError, match="exact zero"):
        prepare_local_mini_moe_training(
            tiny_qwen3_model,
            backend=backend,
            expert_bindings=bindings,
            router_feature_extractor=features,
            router_heads=heads,
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert all(
        not parameter.requires_grad
        for parameter in tiny_qwen3_model.parameters()
    )
