"""Verify EXP-002 PEFT/LoRA isolation and serialization on tiny Qwen3."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import LoRAExpertConfig, PEFTLoRAExpert
from anchored_moe.evaluation import (
    assert_tensors_equivalent,
    fingerprint_module,
)


def main() -> None:
    """Construct, optimize, save, and reload one adapter-only expert."""

    torch.manual_seed(233)
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
    lora_config = LoRAExpertConfig(
        rank=4,
        alpha=8,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )

    with torch.no_grad():
        anchor_output = anchor_mlp(hidden_states)
    expert = PEFTLoRAExpert.from_anchor(
        anchor_mlp,
        config=lora_config,
    )
    with torch.no_grad():
        initial_result = assert_tensors_equivalent(
            expert(hidden_states),
            anchor_output,
            name="zero-update LoRA expert output",
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
        if name not in expert.adapter_parameter_names
    )

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-lora-expert-"
    ) as directory:
        payload_path = Path(directory) / "adapter.safetensors"
        expert.save_safetensors(payload_path)
        reloaded = PEFTLoRAExpert.from_safetensors(
            reload_anchor,
            payload_path,
            trainable=False,
        )
        with torch.no_grad():
            reload_result = assert_tensors_equivalent(
                reloaded(hidden_states),
                expert(hidden_states),
                name="reloaded LoRA expert output",
                atol=0.0,
                rtol=0.0,
            )

    print(
        "PEFT/LoRA expert validation passed: "
        f"exact_initial={initial_result.exact}, "
        f"exact_reload={reload_result.exact}, "
        f"adapter_parameters={expert.adapter_parameter_count}, "
        f"private_base_parameters={expert.base_parameter_count}"
    )


if __name__ == "__main__":
    main()
