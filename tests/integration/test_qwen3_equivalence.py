"""Integration tests for exact Qwen3 forced-anchor equivalence."""

from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import AnchoredMoEConfig, RoutedMLP
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import (
    assert_tensors_equivalent,
    fingerprint_module,
)


def test_forced_anchor_preserves_hidden_states_logits_and_anchor_weights(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    input_ids = torch.tensor([[1, 7, 11, 3], [2, 5, 13, 9]])
    layers = tiny_qwen3_model.model.layers
    selected = (2, 3)
    original_mlps = {index: layers[index].mlp for index in selected}
    fingerprints = {
        index: fingerprint_module(module)
        for index, module in original_mlps.items()
    }

    with torch.no_grad():
        dense_output = tiny_qwen3_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

    report = Qwen3Backend().convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers="upper_half"),
    )
    with torch.no_grad():
        converted_output = tiny_qwen3_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

    assert report.converted_layers == selected
    logits_result = assert_tensors_equivalent(
        converted_output.logits,
        dense_output.logits,
        name="logits",
        atol=0.0,
        rtol=0.0,
    )
    assert logits_result.exact

    assert len(converted_output.hidden_states) == len(dense_output.hidden_states)
    for index, (actual, expected) in enumerate(
        zip(
            converted_output.hidden_states,
            dense_output.hidden_states,
            strict=True,
        )
    ):
        result = assert_tensors_equivalent(
            actual,
            expected,
            name=f"hidden_states[{index}]",
            atol=0.0,
            rtol=0.0,
        )
        assert result.exact

    for index in selected:
        wrapper = layers[index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.anchor_mlp is original_mlps[index]
        assert fingerprint_module(wrapper.anchor_mlp) == fingerprints[index]
