"""Verify EXP-004 segmentation, isolation, training, and safe reload."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import (
    FineGrainedMicroExpertConfig,
    FineGrainedMicroExpertSet,
)
from anchored_moe.evaluation import (
    assert_tensors_equivalent,
    fingerprint_module,
)


def main() -> None:
    """Exercise one atomic fine-grained set on an offline tiny Qwen3 model."""

    torch.manual_seed(373)
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
    config = FineGrainedMicroExpertConfig(segmentation_factor=4)

    with torch.no_grad():
        anchor_output = anchor_mlp(hidden_states)
    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor_mlp,
        config=config,
    )
    with torch.no_grad():
        initial_result = assert_tensors_equivalent(
            expert_set(hidden_states),
            anchor_output,
            name="initial atomic micro-expert set output",
            atol=0.0,
            rtol=0.0,
        )

    optimizer = torch.optim.AdamW(expert_set.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    expert_set(hidden_states).square().mean().backward()
    optimizer.step()
    assert fingerprint_module(anchor_mlp) == anchor_fingerprint
    assert all(parameter.grad is None for parameter in anchor_mlp.parameters())

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-micro-experts-"
    ) as directory:
        payload_path = Path(directory) / "micro-experts.safetensors"
        expert_set.save_safetensors(payload_path)
        reloaded = FineGrainedMicroExpertSet.from_safetensors(
            reload_anchor,
            payload_path,
            trainable=False,
        )
        with torch.no_grad():
            reload_result = assert_tensors_equivalent(
                reloaded(hidden_states),
                expert_set(hidden_states),
                name="reloaded atomic micro-expert set output",
                atol=0.0,
                rtol=0.0,
            )

    print(
        "Fine-grained micro-expert validation passed: "
        f"exact_initial={initial_result.exact}, "
        f"exact_reload={reload_result.exact}, "
        f"micro_experts={expert_set.expert_count}, "
        f"segment_width={expert_set.segment_width}, "
        f"parameters={expert_set.parameter_count}"
    )


if __name__ == "__main__":
    main()
