"""Verify EXP-003 block-sparse isolation and serialization on tiny Qwen3."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import BlockSparseDeltaConfig, BlockSparseDeltaExpert
from anchored_moe.evaluation import (
    assert_tensors_equivalent,
    fingerprint_module,
)


def main() -> None:
    """Construct, optimize, save, and reload one block-sparse expert."""

    torch.manual_seed(331)
    model_config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(model_config).eval()
    anchor_mlp = model.model.layers[-1].mlp
    reload_anchor = deepcopy(anchor_mlp)
    anchor_fingerprint = fingerprint_module(anchor_mlp)
    hidden_states = torch.randn(2, 3, model_config.hidden_size)
    sparse_config = BlockSparseDeltaConfig(
        block_shape=(4, 4),
        blocks_per_module=2,
        seed=29,
    )

    with torch.no_grad():
        anchor_output = anchor_mlp(hidden_states)
    expert = BlockSparseDeltaExpert.from_anchor(
        anchor_mlp,
        config=sparse_config,
    )
    with torch.no_grad():
        initial_result = assert_tensors_equivalent(
            expert(hidden_states),
            anchor_output,
            name="zero-update block-sparse expert output",
            atol=0.0,
            rtol=0.0,
        )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in expert.parameters()
            if parameter.requires_grad
        ),
        lr=1e-2,
    )
    optimizer.zero_grad(set_to_none=True)
    expert(hidden_states).square().mean().backward()
    optimizer.step()
    assert fingerprint_module(anchor_mlp) == anchor_fingerprint
    assert all(
        parameter.grad is None
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.delta_parameter_names
    )

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-block-sparse-"
    ) as directory:
        payload_path = Path(directory) / "sparse-delta.safetensors"
        expert.save_safetensors(payload_path)
        reloaded = BlockSparseDeltaExpert.from_safetensors(
            reload_anchor,
            payload_path,
            trainable=False,
        )
        with torch.no_grad():
            reload_result = assert_tensors_equivalent(
                reloaded(hidden_states),
                expert(hidden_states),
                name="reloaded block-sparse expert output",
                atol=0.0,
                rtol=0.0,
            )

    print(
        "Block-sparse expert validation passed: "
        f"exact_initial={initial_result.exact}, "
        f"exact_reload={reload_result.exact}, "
        f"sparse_parameters={expert.delta_parameter_count}, "
        f"dense_target_parameters={expert.dense_delta_parameter_count}, "
        f"private_base_parameters={expert.base_parameter_count}"
    )


if __name__ == "__main__":
    main()
