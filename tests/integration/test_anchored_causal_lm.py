"""Integration coverage for the CORE-005 Hugging Face model facade."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedModel, Qwen3ForCausalLM

from anchored_moe import (
    AnchoredModelError,
    AnchoredMoEConfig,
    AnchoredMoEForCausalLM,
    RoutedMLP,
)
from anchored_moe.evaluation import assert_tensors_equivalent
from anchored_moe.utils import sha256_json


def test_wrapper_preserves_forward_hidden_states_and_generation(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    """The facade must preserve native Qwen3 behavior in forced-anchor mode."""

    dense_model = copy.deepcopy(tiny_qwen3_model)
    input_ids = torch.tensor([[1, 7, 11, 3], [2, 5, 13, 9]])
    with torch.no_grad():
        dense_output = dense_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        dense_generation = dense_model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
        )

    model = AnchoredMoEForCausalLM.from_anchor_model(
        tiny_qwen3_model,
        converted_layers=[2, 3],
    ).eval()
    with torch.no_grad():
        anchored_output = model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        anchored_generation = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
        )

    assert isinstance(model, PreTrainedModel)
    assert model.can_generate()
    assert model.conversion_report.converted_layers == (2, 3)
    assert model.backend.model_type == "qwen3"
    assert isinstance(model.anchor_model.model.layers[2].mlp, RoutedMLP)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert_tensors_equivalent(
        anchored_output.logits,
        dense_output.logits,
        name="CORE-005 forced-anchor logits",
        atol=0.0,
        rtol=0.0,
    )
    for index, (actual, expected) in enumerate(
        zip(
            anchored_output.hidden_states,
            dense_output.hidden_states,
            strict=True,
        )
    ):
        assert_tensors_equivalent(
            actual,
            expected,
            name=f"CORE-005 hidden_states[{index}]",
            atol=0.0,
            rtol=0.0,
        )
    assert torch.equal(anchored_generation, dense_generation)


def test_from_pretrained_loads_a_local_base_checkpoint(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """The public loader must use normal local Hugging Face checkpoint files."""

    checkpoint = tmp_path / "tiny-qwen3"
    tiny_qwen3_model.save_pretrained(checkpoint)

    model = AnchoredMoEForCausalLM.from_pretrained(
        checkpoint,
        converted_layers=[3],
        local_files_only=True,
    )

    assert model.config.base_model_name_or_path == str(checkpoint)
    assert model.config.base_config_sha256 == sha256_json(
        model.anchor_model.config.to_dict()
    )
    assert model.conversion_report.converted_layers == (3,)
    assert isinstance(model.anchor_model.model.layers[3].mlp, RoutedMLP)


def test_registry_lifecycle_requires_explicit_synchronization(
    tiny_qwen3_model: Qwen3ForCausalLM,
    expert_package_factory,
) -> None:
    """Install and unload must not silently alter routed-wrapper state."""

    model = AnchoredMoEForCausalLM.from_anchor_model(
        tiny_qwen3_model,
        converted_layers=[2, 3],
    )
    package = expert_package_factory("code", router_output_size=16)

    installed = model.install(package)

    assert installed.after.active_packages == (package.identity,)
    for layer_index in (2, 3):
        wrapper = model.anchor_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"

    unloaded = model.unload(package.identity)

    assert unloaded.after.active_packages == ()
    assert model.registry.active_packages == ()
    for layer_index in (2, 3):
        wrapper = model.anchor_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"


def test_empty_registry_synchronization_restores_forced_anchor(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    model = AnchoredMoEForCausalLM.from_anchor_model(
        tiny_qwen3_model,
        converted_layers=[2, 3],
    )

    report = model.synchronize_registry()

    assert report.active_packages == ()
    assert report.configured_layers == ()
    assert report.reset_layers == (2, 3)


def test_invalid_anchor_config_hash_fails_before_conversion(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    original_mlp = tiny_qwen3_model.model.layers[3].mlp
    config = AnchoredMoEConfig(
        converted_layers=[3],
        base_config_sha256="0" * 64,
    )

    with pytest.raises(AnchoredModelError, match="base_config_sha256"):
        AnchoredMoEForCausalLM.from_anchor_model(
            tiny_qwen3_model,
            config=config,
        )

    assert tiny_qwen3_model.model.layers[3].mlp is original_mlp


def test_composition_serialization_requires_an_external_anchor_reference(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """In-memory anchors cannot produce a reloadable reference-only artifact."""

    model = AnchoredMoEForCausalLM.from_anchor_model(
        tiny_qwen3_model,
        converted_layers=[3],
    )

    with pytest.raises(AnchoredModelError, match="base_model_name_or_path"):
        model.save_pretrained(tmp_path / "unsafe-full-model")
