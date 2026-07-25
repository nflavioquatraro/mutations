"""Acceptance tests for the full copied-MLP expert representation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from transformers import Qwen3ForCausalLM

from anchored_moe import Expert, ExpertSerializationError, FullMLPExpert


class BufferedMLP(nn.Module):
    """Small MLP fixture with parameters, a buffer, and a keyword argument."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.register_buffer("output_scale", torch.tensor(1.25))

    def forward(self, hidden_states: Tensor, *, multiplier: float = 1.0) -> Tensor:
        """Apply a deterministic nonlinear transformation for expert tests."""

        return (
            self.down(torch.nn.functional.silu(self.up(hidden_states)))
            * self.output_scale
            * multiplier
        )


def test_full_expert_is_an_exact_independent_trainable_anchor_copy() -> None:
    torch.manual_seed(23)
    anchor = BufferedMLP(hidden_size=4).eval()
    hidden_states = torch.randn(2, 3, 4)
    expected = anchor(hidden_states, multiplier=0.5)

    expert = FullMLPExpert.from_anchor(anchor)
    actual = expert(hidden_states, multiplier=0.5)

    assert isinstance(expert, Expert)
    assert expert.parameterization == "full_mlp"
    assert torch.equal(actual, expected)
    assert expert.training == anchor.training
    assert all(not parameter.requires_grad for parameter in anchor.parameters())
    assert all(parameter.requires_grad for parameter in expert.parameters())
    assert expert.trainable_parameter_count == expert.parameter_count

    anchor_state = anchor.state_dict()
    expert_state = expert.expert_mlp.state_dict()
    assert anchor_state.keys() == expert_state.keys()
    for name in anchor_state:
        assert torch.equal(expert_state[name], anchor_state[name])
        assert expert_state[name].data_ptr() != anchor_state[name].data_ptr()


def test_forced_expert_backward_cannot_reach_or_mutate_anchor() -> None:
    torch.manual_seed(29)
    anchor = BufferedMLP(hidden_size=4)
    anchor_state = {
        name: tensor.detach().clone() for name, tensor in anchor.state_dict().items()
    }
    expert = FullMLPExpert(anchor)
    hidden_states = torch.randn(2, 3, 4)

    expert(hidden_states).square().mean().backward()

    assert all(parameter.grad is None for parameter in anchor.parameters())
    assert all(parameter.grad is not None for parameter in expert.parameters())
    for name, tensor in anchor.state_dict().items():
        assert torch.equal(tensor, anchor_state[name])


def test_inference_expert_has_no_trainable_parameters() -> None:
    anchor = BufferedMLP(hidden_size=4)

    expert = FullMLPExpert.from_anchor(anchor, trainable=False)

    assert expert.parameter_count > 0
    assert expert.trainable_parameter_count == 0
    assert all(not parameter.requires_grad for parameter in expert.parameters())


def test_qwen3_full_expert_matches_anchor_at_initialization(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp
    hidden_states = torch.randn(2, 3, tiny_qwen3_model.config.hidden_size)
    with torch.no_grad():
        expected = anchor(hidden_states)

    expert = FullMLPExpert.from_anchor(anchor)
    with torch.no_grad():
        actual = expert(hidden_states)

    assert torch.equal(actual, expected)


def test_safetensors_round_trip_is_exact_deterministic_and_inference_ready(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    first_anchor = BufferedMLP(hidden_size=4).eval()
    expert = FullMLPExpert.from_anchor(first_anchor)
    with torch.no_grad():
        for index, parameter in enumerate(expert.parameters(), start=1):
            parameter.add_(index / 100)
    hidden_states = torch.randn(2, 3, 4)
    first_path = tmp_path / "expert-a.safetensors"
    second_path = tmp_path / "expert-b.safetensors"

    expert.save_safetensors(first_path)
    expert.save_safetensors(second_path)
    second_anchor = BufferedMLP(hidden_size=4).eval()
    reloaded = FullMLPExpert.from_safetensors(
        second_anchor,
        first_path,
        trainable=False,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert torch.equal(reloaded(hidden_states), expert(hidden_states))
    assert reloaded.trainable_parameter_count == 0
    assert all(not parameter.requires_grad for parameter in second_anchor.parameters())


def test_safetensors_save_protects_existing_payload_by_default(
    tmp_path: Path,
) -> None:
    expert = FullMLPExpert.from_anchor(BufferedMLP(hidden_size=4))
    payload_path = tmp_path / "expert.safetensors"
    expert.save_safetensors(payload_path)
    original_bytes = payload_path.read_bytes()

    with pytest.raises(FileExistsError):
        expert.save_safetensors(payload_path)

    assert payload_path.read_bytes() == original_bytes
    expert.save_safetensors(payload_path, overwrite=True)
    assert payload_path.read_bytes() == original_bytes


def test_load_rejects_missing_metadata_without_freezing_anchor(
    tmp_path: Path,
) -> None:
    anchor = BufferedMLP(hidden_size=4)
    payload_path = tmp_path / "unmarked.safetensors"
    state = {
        f"expert_mlp.{name}": tensor.detach().clone()
        for name, tensor in anchor.state_dict().items()
    }
    save_file(state, payload_path)

    with pytest.raises(ExpertSerializationError, match="metadata"):
        FullMLPExpert.from_safetensors(anchor, payload_path)

    assert all(parameter.requires_grad for parameter in anchor.parameters())


def test_load_rejects_incompatible_tensor_schema_without_freezing_anchor(
    tmp_path: Path,
) -> None:
    source_anchor = BufferedMLP(hidden_size=4)
    expert = FullMLPExpert.from_anchor(source_anchor)
    payload_path = tmp_path / "expert.safetensors"
    expert.save_safetensors(payload_path)
    incompatible_anchor = BufferedMLP(hidden_size=5)

    with pytest.raises(ExpertSerializationError, match="schema"):
        FullMLPExpert.from_safetensors(incompatible_anchor, payload_path)

    assert all(
        parameter.requires_grad for parameter in incompatible_anchor.parameters()
    )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "expert.bin",
        "missing.safetensors",
    ],
)
def test_load_rejects_unsafe_or_missing_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    anchor = BufferedMLP(hidden_size=4)

    with pytest.raises(ExpertSerializationError, match="payload"):
        FullMLPExpert.from_safetensors(anchor, tmp_path / invalid_path)

    assert all(parameter.requires_grad for parameter in anchor.parameters())


def test_save_requires_existing_parent_and_safetensors_suffix(
    tmp_path: Path,
) -> None:
    expert = FullMLPExpert.from_anchor(BufferedMLP(hidden_size=4))

    with pytest.raises(ExpertSerializationError, match="end in"):
        expert.save_safetensors(tmp_path / "expert.bin")
    with pytest.raises(ExpertSerializationError, match="parent"):
        expert.save_safetensors(tmp_path / "missing" / "expert.safetensors")


def test_constructor_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="anchor_mlp"):
        FullMLPExpert("not-a-module")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trainable"):
        FullMLPExpert(BufferedMLP(hidden_size=4), trainable=1)  # type: ignore[arg-type]
