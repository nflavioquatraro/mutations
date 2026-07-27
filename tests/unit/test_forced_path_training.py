"""Acceptance tests for TRN-001 forced-path preparation and isolation."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    AnchoredMoEConfig,
    ForcedExpertBinding,
    ForcedPathTrainingError,
    FullMLPExpert,
    RoutedMLP,
    prepare_forced_path_training,
)
from anchored_moe.backends import Qwen3Backend


def _converted_experts(
    model: Qwen3ForCausalLM,
) -> tuple[Qwen3Backend, tuple[ForcedExpertBinding, ...]]:
    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[2, 3]))
    bindings = []
    for layer_index in (2, 3):
        wrapper = model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        bindings.append(
            ForcedExpertBinding(
                layer_index,
                "code:0",
                FullMLPExpert.from_anchor(wrapper.anchor_mlp),
            )
        )
    return backend, tuple(bindings)


def test_prepare_forced_path_freezes_model_and_trains_only_experts(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend, bindings = _converted_experts(tiny_qwen3_model)
    anchor_states = {
        layer_index: {
            name: tensor.detach().clone()
            for name, tensor in tiny_qwen3_model.model.layers[
                layer_index
            ].mlp.anchor_mlp.state_dict().items()
        }
        for layer_index in (2, 3)
    }

    report = prepare_forced_path_training(
        tiny_qwen3_model,
        backend=backend,
        expert_bindings=bindings[::-1],
    )
    output = tiny_qwen3_model(
        input_ids=torch.tensor([[1, 7, 11, 3]]),
        labels=torch.tensor([[1, 7, 11, 3]]),
    )
    output.loss.backward()

    assert report.layer_indices == (2, 3)
    assert report.expert_id == "code:0"
    assert report.trainable_parameter_count == sum(
        binding.expert.parameter_count for binding in bindings
    )
    assert all(".experts.0." in name for name in report.trainable_parameter_names)
    expected_ids = {
        id(parameter)
        for binding in bindings
        for parameter in binding.expert.parameters()
    }
    actual_ids = {
        id(parameter)
        for parameter in tiny_qwen3_model.parameters()
        if parameter.requires_grad
    }
    assert actual_ids == expected_ids
    assert all(
        parameter.grad is not None
        for binding in bindings
        for parameter in binding.expert.parameters()
    )

    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_expert"
        assert wrapper.expert_ids == ("code:0",)
        for name, tensor in wrapper.anchor_mlp.state_dict().items():
            assert torch.equal(tensor, anchor_states[layer_index][name])
    assert all(
        parameter.grad is None
        for name, parameter in tiny_qwen3_model.named_parameters()
        if ".experts.0." not in name
    )


def test_incomplete_bindings_fail_before_routed_layer_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend, bindings = _converted_experts(tiny_qwen3_model)

    with pytest.raises(ForcedPathTrainingError, match="exactly cover"):
        prepare_forced_path_training(
            tiny_qwen3_model,
            backend=backend,
            expert_bindings=bindings[:1],
        )

    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"


def test_single_expert_package_requires_one_id_across_layers(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend, bindings = _converted_experts(tiny_qwen3_model)
    mismatched = (
        bindings[0],
        ForcedExpertBinding(3, "math:0", bindings[1].expert),
    )

    with pytest.raises(ForcedPathTrainingError, match="one expert_id"):
        prepare_forced_path_training(
            tiny_qwen3_model,
            backend=backend,
            expert_bindings=mismatched,
        )


def test_cross_layer_parameter_sharing_is_rejected_before_attachment(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    layer_two = tiny_qwen3_model.model.layers[2].mlp
    assert isinstance(layer_two, RoutedMLP)
    shared = FullMLPExpert.from_anchor(layer_two.anchor_mlp)
    bindings = (
        ForcedExpertBinding(2, "code:0", shared),
        ForcedExpertBinding(3, "code:0", shared),
    )

    with pytest.raises(ForcedPathTrainingError, match="must not share"):
        prepare_forced_path_training(
            tiny_qwen3_model,
            backend=backend,
            expert_bindings=bindings,
        )

    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"
