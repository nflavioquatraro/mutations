"""Acceptance tests for EXP-003 fixed block-sparse delta experts."""

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
    BLOCK_SPARSE_DELTA_FORMAT_VERSION,
    AnchorRelativeExpert,
    BlockSparseDeltaConfig,
    BlockSparseDeltaExpert,
    ExpertSerializationError,
)


class NamedProjectionMLP(nn.Module):
    """Small gated MLP with Qwen3-compatible projection names."""

    def __init__(self, hidden_size: int = 4, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply a conventional gated MLP transformation."""

        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _config(*, seed: int = 17) -> BlockSparseDeltaConfig:
    """Return a compact multi-block configuration for all projections."""

    return BlockSparseDeltaConfig(
        block_shape=(2, 2),
        blocks_per_module=2,
        target_modules=("gate_proj", "up_proj", "down_proj"),
        seed=seed,
    )


def test_zero_update_is_exact_with_deterministic_disjoint_sparse_storage() -> None:
    torch.manual_seed(307)
    anchor = NamedProjectionMLP().eval()
    second_anchor = deepcopy(anchor)
    hidden_states = torch.randn(2, 3, 4)
    expected = anchor(hidden_states)
    anchor_state = {
        name: value.detach().clone()
        for name, value in anchor.state_dict().items()
    }

    expert = BlockSparseDeltaExpert.from_anchor(anchor, config=_config())
    second = BlockSparseDeltaExpert.from_anchor(
        second_anchor,
        config=_config(),
    )

    assert isinstance(expert, AnchorRelativeExpert)
    assert expert.parameterization == "block_sparse_delta"
    assert torch.equal(expert(hidden_states), expected)
    assert expert.topology == second.topology
    assert expert.delta_parameter_count == 24
    assert expert.dense_delta_parameter_count == 96
    assert expert.trainable_parameter_count == expert.delta_parameter_count
    assert expert.base_parameter_count > expert.delta_parameter_count
    assert all(not parameter.requires_grad for parameter in anchor.parameters())
    for module_name in ("gate_proj", "up_proj", "down_proj"):
        sparse_module = expert.expert_mlp.get_submodule(module_name)
        private_weight = sparse_module.base_layer.weight
        assert torch.equal(
            private_weight,
            anchor_state[f"{module_name}.weight"],
        )
        assert (
            private_weight.data_ptr()
            != anchor.get_submodule(module_name).weight.data_ptr()
        )


def test_backward_reaches_only_sparse_blocks_and_delta_contract_is_exact() -> None:
    torch.manual_seed(311)
    anchor = NamedProjectionMLP().eval()
    hidden_states = torch.randn(2, 3, 4)
    anchor_output = anchor(hidden_states)
    expert = BlockSparseDeltaExpert(anchor, config=_config())

    zero_delta = expert.forward_delta(
        hidden_states,
        anchor_output=anchor_output,
    )
    assert torch.equal(zero_delta, torch.zeros_like(anchor_output))

    with torch.no_grad():
        expert.expert_mlp.down_proj.block_values.fill_(0.05)
    complete_output = expert(hidden_states)
    delta = expert.forward_delta(
        hidden_states,
        anchor_output=anchor_output,
    )
    assert torch.equal(delta, complete_output - anchor_output)
    assert torch.count_nonzero(delta).item() > 0

    complete_output.square().mean().backward()
    delta_names = set(expert.delta_parameter_names)
    assert all(
        parameter.grad is not None
        for name, parameter in expert.expert_mlp.named_parameters()
        if name in delta_names
    )
    assert all(
        parameter.grad is None
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in delta_names
    )
    assert all(parameter.grad is None for parameter in anchor.parameters())


def test_compact_block_execution_matches_an_explicit_dense_delta() -> None:
    torch.manual_seed(312)
    anchor = NamedProjectionMLP().eval()
    dense_reference = deepcopy(anchor)
    config = BlockSparseDeltaConfig(
        block_shape=(2, 2),
        blocks_per_module=8,
        target_modules=("up_proj",),
        seed=5,
    )
    expert = BlockSparseDeltaExpert(anchor, config=config)
    sparse_up = expert.expert_mlp.up_proj
    with torch.no_grad():
        sparse_up.block_values.copy_(
            torch.randn_like(sparse_up.block_values) * 0.05
        )
        for block_values, (output_index, input_index) in zip(
            sparse_up.block_values,
            expert.topology[0].coordinates,
            strict=True,
        ):
            output_start = output_index * 2
            input_start = input_index * 2
            dense_reference.up_proj.weight[
                output_start : output_start + 2,
                input_start : input_start + 2,
            ].add_(block_values)
    hidden_states = torch.randn(2, 3, 4)

    torch.testing.assert_close(
        expert(hidden_states),
        dense_reference(hidden_states),
        rtol=1e-6,
        atol=1e-7,
    )


def test_trainability_toggle_never_unfreezes_private_base() -> None:
    expert = BlockSparseDeltaExpert(
        NamedProjectionMLP(),
        config=_config(),
        trainable=False,
    )

    assert expert.trainable_parameter_count == 0
    expert.set_trainable(True)
    assert expert.trainable_parameter_count == expert.delta_parameter_count
    assert all(
        not parameter.requires_grad
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.delta_parameter_names
    )
    expert.set_trainable(False)
    assert expert.trainable_parameter_count == 0
    with pytest.raises(TypeError, match="boolean"):
        expert.set_trainable(1)  # type: ignore[arg-type]


def test_qwen3_sparse_expert_matches_anchor_at_zero_update(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp
    hidden_states = torch.randn(2, 3, tiny_qwen3_model.config.hidden_size)
    with torch.no_grad():
        expected = anchor(hidden_states)

    expert = BlockSparseDeltaExpert.from_anchor(
        anchor,
        config=BlockSparseDeltaConfig(
            block_shape=(4, 4),
            blocks_per_module=2,
            seed=19,
        ),
    )
    with torch.no_grad():
        actual = expert(hidden_states)

    assert torch.equal(actual, expected)
    assert tuple(item.module_path for item in expert.topology) == (
        "down_proj",
        "gate_proj",
        "up_proj",
    )
    assert expert.delta_parameter_names == (
        "gate_proj.block_values",
        "up_proj.block_values",
        "down_proj.block_values",
    )


def test_sparse_only_safetensors_round_trip_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(313)
    anchor = NamedProjectionMLP().eval()
    reload_anchor = deepcopy(anchor)
    expert = BlockSparseDeltaExpert(anchor, config=_config())
    with torch.no_grad():
        for index, name in enumerate(expert.delta_parameter_names, start=1):
            expert.expert_mlp.get_parameter(name).fill_(index / 100)
    hidden_states = torch.randn(2, 3, 4)
    first_path = tmp_path / "sparse-a.safetensors"
    second_path = tmp_path / "sparse-b.safetensors"

    expert.save_safetensors(first_path)
    expert.save_safetensors(second_path)
    restored = BlockSparseDeltaExpert.from_safetensors(
        reload_anchor,
        first_path,
        trainable=False,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert torch.equal(restored(hidden_states), expert(hidden_states))
    assert restored.config == expert.config
    assert restored.topology == expert.topology
    assert restored.trainable_parameter_count == 0
    with safe_open(first_path, framework="pt", device="cpu") as payload:
        payload_keys = tuple(sorted(payload.keys()))
        assert payload_keys == (
            "modules.0.block_coordinates",
            "modules.0.block_values",
            "modules.1.block_coordinates",
            "modules.1.block_values",
            "modules.2.block_coordinates",
            "modules.2.block_values",
        )
        metadata = json.loads(payload.metadata()["anchored_moe"])
    assert BLOCK_SPARSE_DELTA_FORMAT_VERSION == 1
    assert metadata["format"] == "anchored_moe.block_sparse_delta.v1"
    assert metadata["format_version"] == 1
    assert metadata["parameterization"] == "block_sparse_delta"
    assert metadata["config"] == expert.config.to_dict()
    assert metadata["topology"] == [
        item.to_dict() for item in expert.topology
    ]
    assert not any("base_layer" in key for key in payload_keys)


def test_load_rejects_changed_coordinates_without_freezing_anchor(
    tmp_path: Path,
) -> None:
    source_anchor = NamedProjectionMLP()
    expert = BlockSparseDeltaExpert(source_anchor, config=_config())
    valid_path = tmp_path / "valid.safetensors"
    invalid_path = tmp_path / "invalid.safetensors"
    expert.save_safetensors(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as payload:
        metadata = dict(payload.metadata())
        tensors = {
            key: payload.get_tensor(key)
            for key in payload.keys()
        }
    tensors["modules.0.block_coordinates"] = torch.flip(
        tensors["modules.0.block_coordinates"],
        dims=(0,),
    )
    save_file(tensors, invalid_path, metadata=metadata)
    candidate_anchor = NamedProjectionMLP()

    with pytest.raises(ExpertSerializationError, match="coordinates changed"):
        BlockSparseDeltaExpert.from_safetensors(
            candidate_anchor,
            invalid_path,
        )

    assert all(
        parameter.requires_grad for parameter in candidate_anchor.parameters()
    )


def test_invalid_targets_and_shapes_fail_without_freezing_anchor() -> None:
    missing_target_anchor = NamedProjectionMLP()
    with pytest.raises(ValueError, match="exactly one"):
        BlockSparseDeltaExpert(
            missing_target_anchor,
            config=BlockSparseDeltaConfig(
                block_shape=(2, 2),
                target_modules=("missing_proj",),
            ),
        )
    assert all(
        parameter.requires_grad
        for parameter in missing_target_anchor.parameters()
    )

    indivisible_anchor = NamedProjectionMLP(hidden_size=5, intermediate_size=7)
    with pytest.raises(ValueError, match="divisible"):
        BlockSparseDeltaExpert(
            indivisible_anchor,
            config=BlockSparseDeltaConfig(
                block_shape=(2, 2),
                target_modules=("gate_proj",),
            ),
        )
    assert all(
        parameter.requires_grad
        for parameter in indivisible_anchor.parameters()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_shape": (0, 2)},
        {"block_shape": (2,)},
        {"blocks_per_module": 0},
        {"target_modules": ()},
        {"target_modules": ("up_proj", "up_proj")},
        {"seed": -1},
    ],
)
def test_configuration_rejects_unsupported_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        BlockSparseDeltaConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_and_path_validation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="anchor_mlp"):
        BlockSparseDeltaExpert("not-a-module")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        BlockSparseDeltaExpert(
            NamedProjectionMLP(),
            config="not-a-config",  # type: ignore[arg-type]
        )
    expert = BlockSparseDeltaExpert(NamedProjectionMLP(), config=_config())
    assert BlockSparseDeltaConfig.from_dict(
        expert.config.to_dict()
    ) == expert.config
    with pytest.raises(ExpertSerializationError, match="end in"):
        expert.save_safetensors(tmp_path / "sparse.bin")
    with pytest.raises(ExpertSerializationError, match="parent"):
        expert.save_safetensors(
            tmp_path / "missing" / "sparse.safetensors"
        )
    with pytest.raises(ExpertSerializationError, match="regular file"):
        BlockSparseDeltaExpert.from_safetensors(
            NamedProjectionMLP(),
            tmp_path / "missing.safetensors",
        )
    payload = tmp_path / "sparse.safetensors"
    expert.save_safetensors(payload)
    with pytest.raises(FileExistsError):
        expert.save_safetensors(payload)
    expert.save_safetensors(payload, overwrite=True)
