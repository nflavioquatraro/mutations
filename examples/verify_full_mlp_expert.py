"""Verify full copied-MLP initialization and serialization on tiny Qwen3."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import FullMLPExpert
from anchored_moe.evaluation import (
    assert_tensors_equivalent,
    fingerprint_module,
)


def main() -> None:
    """Copy, optimize, save, and reload one deterministic local expert."""

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(config).eval()
    anchor_mlp = model.model.layers[-1].mlp
    anchor_fingerprint = fingerprint_module(anchor_mlp)
    hidden_states = torch.randn(2, 3, config.hidden_size)

    with torch.no_grad():
        anchor_output = anchor_mlp(hidden_states)
    expert = FullMLPExpert.from_anchor(anchor_mlp)
    with torch.no_grad():
        initial_expert_output = expert(hidden_states)
    initial_result = assert_tensors_equivalent(
        initial_expert_output,
        anchor_output,
        name="initial expert output",
        atol=0.0,
        rtol=0.0,
    )

    # Forced-path optimization reaches only the independent expert tensors.
    expert(hidden_states).square().mean().backward()
    assert all(parameter.grad is None for parameter in anchor_mlp.parameters())
    assert any(parameter.grad is not None for parameter in expert.parameters())
    assert fingerprint_module(anchor_mlp) == anchor_fingerprint

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-full-expert-"
    ) as directory:
        payload_path = Path(directory) / "expert.safetensors"
        expert.save_safetensors(payload_path)
        reloaded = FullMLPExpert.from_safetensors(
            anchor_mlp,
            payload_path,
            trainable=False,
        )
        with torch.no_grad():
            reloaded_result = assert_tensors_equivalent(
                reloaded(hidden_states),
                expert(hidden_states),
                name="reloaded expert output",
                atol=0.0,
                rtol=0.0,
            )

    print(
        "Full MLP expert validation passed: "
        f"exact_initial={initial_result.exact}, "
        f"exact_reload={reloaded_result.exact}, "
        f"expert_parameters={expert.parameter_count}, "
        f"trainable_parameters={expert.trainable_parameter_count}"
    )


if __name__ == "__main__":
    main()
