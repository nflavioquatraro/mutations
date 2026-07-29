"""Acceptance tests for EXP-004 fine-grained micro-expert sets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    MICRO_EXPERT_FORMAT_VERSION,
    Expert,
    ExpertSerializationError,
    FineGrainedMicroExpert,
    FineGrainedMicroExpertConfig,
    FineGrainedMicroExpertSet,
)


class GatedMLP(nn.Module):
    """Small bias-free gated MLP matching the EXP-004 structural contract."""

    def __init__(self, hidden_size: int = 4, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the conventional gated MLP equation."""

        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _config(*, factor: int = 4) -> FineGrainedMicroExpertConfig:
    """Return the standard contiguous test configuration."""

    return FineGrainedMicroExpertConfig(segmentation_factor=factor)


def test_all_segments_exactly_reconstruct_an_independent_frozen_anchor() -> None:
    torch.manual_seed(347)
    anchor = GatedMLP().eval()
    hidden_states = torch.randn(2, 3, 4)
    expected = anchor(hidden_states)
    anchor_parameters = tuple(anchor.parameters())
    anchor_parameter_count = sum(
        parameter.numel() for parameter in anchor_parameters
    )

    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor,
        config=_config(),
    )
    actual = expert_set(hidden_states)

    assert isinstance(expert_set, Expert)
    assert expert_set.parameterization == "fine_grained_micro"
    assert expert_set.expert_count == 4
    assert expert_set.segment_width == 2
    assert expert_set.parameter_count == anchor_parameter_count
    assert expert_set.trainable_parameter_count == anchor_parameter_count
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in anchor_parameters)
    assert all(
        isinstance(expert, FineGrainedMicroExpert)
        for expert in expert_set.micro_experts
    )
    assert tuple(
        (expert.index, expert.channel_start, expert.channel_end)
        for expert in expert_set.micro_experts
    ) == (
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 6),
        (3, 6, 8),
    )

    anchor_storage = {parameter.data_ptr() for parameter in anchor_parameters}
    expert_storage = {
        parameter.data_ptr() for parameter in expert_set.parameters()
    }
    assert len(expert_storage) == len(tuple(expert_set.parameters()))
    assert expert_storage.isdisjoint(anchor_storage)


def test_individual_micro_experts_are_narrow_partial_contributions() -> None:
    torch.manual_seed(349)
    anchor = GatedMLP().eval()
    hidden_states = torch.randn(2, 3, 4)
    expert_set = FineGrainedMicroExpertSet(anchor, config=_config())

    contributions = tuple(
        expert(hidden_states) for expert in expert_set.micro_experts
    )

    assert all(output.shape == hidden_states.shape for output in contributions)
    torch.testing.assert_close(
        torch.stack(contributions).sum(dim=0),
        expert_set(hidden_states),
        rtol=1e-6,
        atol=1e-7,
    )
    assert all(
        expert.gate_proj.out_features == expert_set.segment_width
        and expert.up_proj.out_features == expert_set.segment_width
        and expert.down_proj.in_features == expert_set.segment_width
        for expert in expert_set.micro_experts
    )


def test_backward_reaches_every_segment_but_not_the_anchor() -> None:
    torch.manual_seed(353)
    anchor = GatedMLP()
    anchor_state = {
        name: tensor.detach().clone()
        for name, tensor in anchor.state_dict().items()
    }
    expert_set = FineGrainedMicroExpertSet(anchor, config=_config())
    hidden_states = torch.randn(2, 3, 4)

    expert_set(hidden_states).square().mean().backward()

    assert all(parameter.grad is None for parameter in anchor.parameters())
    assert all(
        parameter.grad is not None for parameter in expert_set.parameters()
    )
    for name, tensor in anchor.state_dict().items():
        assert torch.equal(tensor, anchor_state[name])


def test_trainability_is_atomic_across_the_complete_set() -> None:
    expert_set = FineGrainedMicroExpertSet(
        GatedMLP(),
        config=_config(),
        trainable=False,
    )

    assert expert_set.trainable_parameter_count == 0
    expert_set.set_trainable(True)
    assert expert_set.trainable_parameter_count == expert_set.parameter_count
    expert_set.set_trainable(False)
    assert expert_set.trainable_parameter_count == 0
    with pytest.raises(TypeError, match="boolean"):
        expert_set.set_trainable(1)  # type: ignore[arg-type]


def test_qwen3_micro_expert_set_is_exact_at_initialization(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp
    hidden_states = torch.randn(
        2,
        3,
        tiny_qwen3_model.config.hidden_size,
    )
    with torch.no_grad():
        expected = anchor(hidden_states)

    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor,
        config=_config(),
    )
    with torch.no_grad():
        actual = expert_set(hidden_states)

    assert torch.equal(actual, expected)
    assert (
        expert_set.topology.source_module_class
        == "transformers.models.qwen3.modeling_qwen3.Qwen3MLP"
    )


def test_safetensors_round_trip_is_exact_deterministic_and_atomic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(359)
    source_anchor = GatedMLP().eval()
    reload_anchor = deepcopy(source_anchor)
    expert_set = FineGrainedMicroExpertSet(
        source_anchor,
        config=_config(),
    )
    with torch.no_grad():
        for index, parameter in enumerate(expert_set.parameters(), start=1):
            parameter.add_(index / 1000)
    hidden_states = torch.randn(2, 3, 4)
    first_path = tmp_path / "micro-a.safetensors"
    second_path = tmp_path / "micro-b.safetensors"

    expert_set.save_safetensors(first_path)
    expert_set.save_safetensors(second_path)
    reloaded = FineGrainedMicroExpertSet.from_safetensors(
        reload_anchor,
        first_path,
        trainable=False,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert torch.equal(reloaded(hidden_states), expert_set(hidden_states))
    assert reloaded.config == expert_set.config
    assert reloaded.topology == expert_set.topology
    assert reloaded.trainable_parameter_count == 0
    with safe_open(first_path, framework="pt", device="cpu") as payload:
        metadata = json.loads(payload.metadata()["anchored_moe"])
        keys = tuple(payload.keys())
    assert metadata["format"] == "anchored_moe.fine_grained_micro.v1"
    assert metadata["parameterization"] == "fine_grained_micro"
    assert metadata["topology"] == expert_set.topology.to_dict()
    assert len(keys) == expert_set.expert_count * 3
    assert all(key.startswith("micro_experts.") for key in keys)


def test_save_protects_destinations_and_validates_paths(tmp_path: Path) -> None:
    expert_set = FineGrainedMicroExpertSet(GatedMLP(), config=_config())
    payload_path = tmp_path / "micro.safetensors"
    expert_set.save_safetensors(payload_path)
    original = payload_path.read_bytes()

    with pytest.raises(FileExistsError):
        expert_set.save_safetensors(payload_path)
    expert_set.save_safetensors(payload_path, overwrite=True)
    assert payload_path.read_bytes() == original
    with pytest.raises(ExpertSerializationError, match="end in"):
        expert_set.save_safetensors(tmp_path / "micro.bin")
    with pytest.raises(ExpertSerializationError, match="parent"):
        expert_set.save_safetensors(
            tmp_path / "missing" / "micro.safetensors"
        )


def test_invalid_structure_fails_without_freezing_the_anchor() -> None:
    nondivisible = GatedMLP(intermediate_size=10)
    biased = GatedMLP()
    biased.down_proj = nn.Linear(8, 4, bias=True)
    extra_state = GatedMLP()
    extra_state.extra = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="divisible"):
        FineGrainedMicroExpertSet(nondivisible, config=_config())
    with pytest.raises(ValueError, match="bias-free"):
        FineGrainedMicroExpertSet(biased, config=_config())
    with pytest.raises(ValueError, match="exactly"):
        FineGrainedMicroExpertSet(extra_state, config=_config())

    assert all(
        parameter.requires_grad
        for anchor in (nondivisible, biased, extra_state)
        for parameter in anchor.parameters()
    )


@pytest.mark.parametrize(
    "values, message",
    [
        ({"segmentation_factor": 1}, "integer >= 2"),
        ({"segmentation_factor": True}, "integer >= 2"),
        (
            {"segmentation_factor": 2, "partition": "hashed"},
            "contiguous",
        ),
        (
            {"segmentation_factor": 2, "gate_projection": "_private"},
            "public Python identifiers",
        ),
    ],
)
def test_config_rejects_ambiguous_or_unsupported_values(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FineGrainedMicroExpertConfig(**values)  # type: ignore[arg-type]


def test_load_rejects_tampering_before_freezing_the_anchor(
    tmp_path: Path,
) -> None:
    expert_set = FineGrainedMicroExpertSet(GatedMLP(), config=_config())
    valid_path = tmp_path / "valid.safetensors"
    expert_set.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as payload:
        metadata = json.loads(payload.metadata()["anchored_moe"])
    metadata["topology"]["hidden_size"] = 99
    tampered_path = tmp_path / "tampered.safetensors"
    save_file(
        tensors,
        tampered_path,
        metadata={
            "anchored_moe": json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    reload_anchor = GatedMLP()

    with pytest.raises(ExpertSerializationError, match="topology"):
        FineGrainedMicroExpertSet.from_safetensors(
            reload_anchor,
            tampered_path,
        )

    assert all(
        parameter.requires_grad for parameter in reload_anchor.parameters()
    )


def test_load_rejects_nonfinite_or_incomplete_state_before_freezing_anchor(
    tmp_path: Path,
) -> None:
    expert_set = FineGrainedMicroExpertSet(GatedMLP(), config=_config())
    valid_path = tmp_path / "valid.safetensors"
    expert_set.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as payload:
        metadata = payload.metadata()
    first_key = sorted(tensors)[0]
    tensors[first_key] = tensors[first_key].clone()
    tensors[first_key].view(-1)[0] = float("nan")
    nonfinite_path = tmp_path / "nonfinite.safetensors"
    save_file(tensors, nonfinite_path, metadata=metadata)
    incomplete_path = tmp_path / "incomplete.safetensors"
    tensors.pop(first_key)
    save_file(tensors, incomplete_path, metadata=metadata)

    for path, message in (
        (nonfinite_path, "not finite"),
        (incomplete_path, "keys"),
    ):
        reload_anchor = GatedMLP()
        with pytest.raises(ExpertSerializationError, match=message):
            FineGrainedMicroExpertSet.from_safetensors(
                reload_anchor,
                path,
            )
        assert all(
            parameter.requires_grad for parameter in reload_anchor.parameters()
        )


def test_public_format_version_and_forward_contract_are_strict() -> None:
    assert MICRO_EXPERT_FORMAT_VERSION == 1
    expert_set = FineGrainedMicroExpertSet(GatedMLP(), config=_config())

    with pytest.raises(ValueError, match="last dimension"):
        expert_set(torch.randn(2, 3, 5))
    with pytest.raises(TypeError, match="only hidden_states"):
        expert_set(torch.randn(2, 3, 4), multiplier=0.5)
