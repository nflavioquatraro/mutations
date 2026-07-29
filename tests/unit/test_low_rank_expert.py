"""Acceptance tests for the standalone PEFT/LoRA expert representation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    Expert,
    ExpertSerializationError,
    LoRAExpertConfig,
    PEFTLoRAExpert,
)


class NamedProjectionMLP(nn.Module):
    """Small gated MLP whose projection names match the Qwen3 convention."""

    def __init__(self, hidden_size: int = 4, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the same gated projection structure used by Qwen3 MLPs."""

        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _config(*, dropout: float = 0.0) -> LoRAExpertConfig:
    """Return one compact deterministic adapter configuration."""

    return LoRAExpertConfig(
        rank=2,
        alpha=4,
        dropout=dropout,
        target_modules=("up_proj", "gate_proj", "down_proj"),
    )


def test_lora_expert_is_exact_at_initialization_and_owns_private_storage() -> None:
    torch.manual_seed(211)
    anchor = NamedProjectionMLP().eval()
    hidden_states = torch.randn(2, 3, 4)
    expected = anchor(hidden_states)
    anchor_state = {
        name: tensor.detach().clone()
        for name, tensor in anchor.state_dict().items()
    }

    expert = PEFTLoRAExpert.from_anchor(anchor, config=_config())
    actual = expert(hidden_states)

    assert isinstance(expert, Expert)
    assert expert.parameterization == "peft_lora"
    assert torch.equal(actual, expected)
    assert expert.training is False
    assert expert.adapter_parameter_count == expert.trainable_parameter_count
    assert expert.base_parameter_count > expert.adapter_parameter_count
    assert all(not parameter.requires_grad for parameter in anchor.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in expert.expert_mlp.named_parameters()
        if name in expert.adapter_parameter_names
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.adapter_parameter_names
    )
    private_state = expert.expert_mlp.state_dict()
    for name, tensor in anchor_state.items():
        private_name = name.replace(
            ".weight",
            ".base_layer.weight",
        )
        assert torch.equal(private_state[private_name], tensor)
        assert private_state[private_name].data_ptr() != anchor.state_dict()[
            name
        ].data_ptr()


def test_backward_reaches_only_adapters_and_never_the_original_anchor() -> None:
    torch.manual_seed(223)
    anchor = NamedProjectionMLP()
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in anchor.state_dict().items()
    }
    expert = PEFTLoRAExpert(anchor, config=_config())
    hidden_states = torch.randn(2, 3, 4)

    expert(hidden_states).square().mean().backward()

    assert all(parameter.grad is None for parameter in anchor.parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in expert.expert_mlp.named_parameters()
        if name in expert.adapter_parameter_names
    )
    assert all(
        parameter.grad is None
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.adapter_parameter_names
    )
    for name, tensor in anchor.state_dict().items():
        assert torch.equal(tensor, anchor_before[name])


def test_lora_expert_exposes_exact_anchor_relative_delta() -> None:
    torch.manual_seed(229)
    anchor = NamedProjectionMLP().eval()
    hidden_states = torch.randn(2, 3, 4)
    expert = PEFTLoRAExpert(anchor, config=_config())
    anchor_output = anchor(hidden_states)

    zero_delta = expert.forward_delta(
        hidden_states,
        anchor_output=anchor_output,
    )
    assert torch.equal(zero_delta, torch.zeros_like(anchor_output))

    with torch.no_grad():
        for name in expert.adapter_parameter_names:
            if ".lora_B." in name:
                expert.expert_mlp.get_parameter(name).fill_(0.05)
    complete_output = expert(hidden_states)
    delta = expert.forward_delta(
        hidden_states,
        anchor_output=anchor_output,
    )
    assert torch.equal(delta, complete_output - anchor_output)
    assert torch.count_nonzero(delta).item() > 0


def test_trainability_toggle_never_unfreezes_private_base() -> None:
    expert = PEFTLoRAExpert(
        NamedProjectionMLP(),
        config=_config(),
        trainable=False,
    )

    assert expert.trainable_parameter_count == 0
    expert.set_trainable(True)
    assert expert.trainable_parameter_count == expert.adapter_parameter_count
    assert all(
        not parameter.requires_grad
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.adapter_parameter_names
    )
    expert.set_trainable(False)
    assert expert.trainable_parameter_count == 0

    with pytest.raises(TypeError, match="boolean"):
        expert.set_trainable(1)  # type: ignore[arg-type]


def test_qwen3_lora_expert_matches_anchor_at_zero_update(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp
    hidden_states = torch.randn(2, 3, tiny_qwen3_model.config.hidden_size)
    with torch.no_grad():
        expected = anchor(hidden_states)

    expert = PEFTLoRAExpert.from_anchor(anchor, config=_config())
    with torch.no_grad():
        actual = expert(hidden_states)

    assert torch.equal(actual, expected)
    assert expert.adapter_parameter_names == (
        "gate_proj.lora_A.default.weight",
        "gate_proj.lora_B.default.weight",
        "up_proj.lora_A.default.weight",
        "up_proj.lora_B.default.weight",
        "down_proj.lora_A.default.weight",
        "down_proj.lora_B.default.weight",
    )


def test_adapter_only_safetensors_round_trip_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(227)
    anchor = NamedProjectionMLP().eval()
    reload_anchor = deepcopy(anchor)
    expert = PEFTLoRAExpert(anchor, config=_config())
    with torch.no_grad():
        for index, name in enumerate(expert.adapter_parameter_names, start=1):
            expert.expert_mlp.get_parameter(name).add_(index / 100)
    hidden_states = torch.randn(2, 3, 4)
    first_path = tmp_path / "adapter-a.safetensors"
    second_path = tmp_path / "adapter-b.safetensors"

    expert.save_safetensors(first_path)
    expert.save_safetensors(second_path)
    restored = PEFTLoRAExpert.from_safetensors(
        reload_anchor,
        first_path,
        trainable=False,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert torch.equal(restored(hidden_states), expert(hidden_states))
    assert restored.config == expert.config
    assert restored.trainable_parameter_count == 0
    assert len(restored.state_dict()) > len(restored.adapter_parameter_names)
    with safe_open(first_path, framework="pt", device="cpu") as payload:
        assert tuple(sorted(payload.keys())) == (
            "down_proj.lora_A.weight",
            "down_proj.lora_B.weight",
            "gate_proj.lora_A.weight",
            "gate_proj.lora_B.weight",
            "up_proj.lora_A.weight",
            "up_proj.lora_B.weight",
        )
        metadata = json.loads(payload.metadata()["anchored_moe"])
    assert metadata["format"] == "anchored_moe.peft_lora.v1"
    assert metadata["parameterization"] == "peft_lora"
    assert metadata["config"] == expert.config.to_dict()
    assert metadata["peft_version"]


def test_load_rejects_bad_schema_without_freezing_anchor(
    tmp_path: Path,
) -> None:
    source_anchor = NamedProjectionMLP()
    expert = PEFTLoRAExpert(source_anchor, config=_config())
    valid_path = tmp_path / "valid.safetensors"
    invalid_path = tmp_path / "invalid.safetensors"
    expert.save_safetensors(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as payload:
        metadata = dict(payload.metadata())
    tensors = {
        "gate_proj.lora_A.weight": torch.zeros(2, 4),
    }
    save_file(tensors, invalid_path, metadata=metadata)
    candidate_anchor = NamedProjectionMLP()

    with pytest.raises(ExpertSerializationError, match="state keys"):
        PEFTLoRAExpert.from_safetensors(candidate_anchor, invalid_path)

    assert all(
        parameter.requires_grad for parameter in candidate_anchor.parameters()
    )


def test_invalid_target_modules_fail_without_freezing_anchor() -> None:
    anchor = NamedProjectionMLP()

    with pytest.raises(ValueError, match="target_modules"):
        PEFTLoRAExpert(
            anchor,
            config=LoRAExpertConfig(target_modules=("missing_proj",)),
        )

    assert all(parameter.requires_grad for parameter in anchor.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"alpha": 0},
        {"dropout": 1.0},
        {"target_modules": ()},
        {"target_modules": ("up_proj", "up_proj")},
        {"use_rslora": 1},
    ],
)
def test_configuration_rejects_unsupported_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        LoRAExpertConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_and_path_validation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="anchor_mlp"):
        PEFTLoRAExpert("not-a-module")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        PEFTLoRAExpert(
            NamedProjectionMLP(),
            config="not-a-config",  # type: ignore[arg-type]
        )
    expert = PEFTLoRAExpert(NamedProjectionMLP(), config=_config())
    with pytest.raises(ExpertSerializationError, match="end in"):
        expert.save_safetensors(tmp_path / "adapter.bin")
    with pytest.raises(ExpertSerializationError, match="parent"):
        expert.save_safetensors(
            tmp_path / "missing" / "adapter.safetensors"
        )
    with pytest.raises(ExpertSerializationError, match="regular file"):
        PEFTLoRAExpert.from_safetensors(
            NamedProjectionMLP(),
            tmp_path / "missing.safetensors",
        )
