"""Acceptance tests for COM-005 effective LoRA parameter merging."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn

from anchored_moe import (
    LearnedMixtureWeightHead,
    LoRAExpertConfig,
    LoRAParameterSpaceMerge,
    PEFTLoRAExpert,
    ParameterSpaceMergeError,
    ParameterSpaceMergeResult,
)


class NamedProjectionMLP(nn.Module):
    """Small nonlinear gated MLP with Qwen3-compatible projection names."""

    def __init__(self, hidden_size: int = 4, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the Qwen3-style gated projection order."""

        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _config() -> LoRAExpertConfig:
    """Return one deterministic all-projection LoRA topology."""

    return LoRAExpertConfig(
        rank=2,
        alpha=4,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )


def _experts(anchor: nn.Module) -> tuple[PEFTLoRAExpert, PEFTLoRAExpert]:
    """Construct two compatible experts with distinct non-zero updates."""

    first = PEFTLoRAExpert.from_anchor(anchor, config=_config())
    second = PEFTLoRAExpert.from_anchor(anchor, config=_config())
    with torch.no_grad():
        for expert_index, expert in enumerate((first, second), start=1):
            for parameter_index, name in enumerate(
                expert.adapter_parameter_names,
                start=1,
            ):
                value = (expert_index * 10 + parameter_index) / 100
                expert.expert_mlp.get_parameter(name).fill_(value)
    return first, second


def _manual_merged_outputs(
    anchor: NamedProjectionMLP,
    experts: tuple[PEFTLoRAExpert, PEFTLoRAExpert],
    hidden_states: Tensor,
    weights: Tensor,
) -> Tensor:
    """Materialize the effective dense equation independently per token."""

    expert_deltas = [expert.projection_deltas() for expert in experts]
    outputs: list[Tensor] = []
    for token_index in range(hidden_states.shape[0]):
        merged = deepcopy(anchor)
        with torch.no_grad():
            for projection_index, projection_name in enumerate(
                item.module_name for item in expert_deltas[0]
            ):
                live_projection = merged.get_submodule(projection_name)
                live_projection.weight.copy_(
                    anchor.get_submodule(projection_name).weight
                    + weights[token_index, 0]
                    * expert_deltas[0][projection_index].delta_weight
                    + weights[token_index, 1]
                    * expert_deltas[1][projection_index].delta_weight
                )
        outputs.append(merged(hidden_states[token_index : token_index + 1]))
    return torch.cat(outputs, dim=0)


def test_parameter_merge_matches_explicit_effective_weight_materialization() -> None:
    torch.manual_seed(601)
    anchor = NamedProjectionMLP().eval()
    experts = _experts(anchor)
    hidden_states = torch.randn(3, 4)
    weights = torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.5, 0.5]])
    policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(weight=torch.zeros(2), bias=torch.zeros(()))
    )

    result = policy.merge(
        hidden_states,
        anchor_mlp=anchor,
        experts=experts,
        weights=weights,
    )
    expected = _manual_merged_outputs(anchor, experts, hidden_states, weights)

    assert policy.policy_name == "parameter_space_lora_merge"
    assert result.projection_names == ("down_proj", "gate_proj", "up_proj")
    assert torch.equal(result.weights, weights)
    assert torch.allclose(result.output, expected, atol=1e-6, rtol=1e-6)


def test_zero_updates_are_exact_anchor_and_endpoint_weights_select_one_delta() -> None:
    torch.manual_seed(607)
    anchor = NamedProjectionMLP().eval()
    zero_experts = (
        PEFTLoRAExpert.from_anchor(anchor, config=_config()),
        PEFTLoRAExpert.from_anchor(anchor, config=_config()),
    )
    policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(weight=torch.zeros(2), bias=torch.zeros(()))
    )
    hidden_states = torch.randn(2, 4, requires_grad=True)
    zero_result = policy.merge(
        hidden_states,
        anchor_mlp=anchor,
        experts=zero_experts,
        weights=torch.tensor([[0.5, 0.5], [0.2, 0.8]]),
    )
    assert torch.equal(zero_result.output, anchor(hidden_states))
    zero_result.output.square().sum().backward()
    assert hidden_states.grad is not None
    assert any(
        expert.expert_mlp.get_parameter(name).grad is not None
        and torch.count_nonzero(
            expert.expert_mlp.get_parameter(name).grad
        ).item()
        > 0
        for expert in zero_experts
        for name in expert.adapter_parameter_names
        if ".lora_B." in name
    )

    experts = _experts(anchor)
    endpoint_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    endpoint_result = policy.merge(
        hidden_states,
        anchor_mlp=anchor,
        experts=experts,
        weights=endpoint_weights,
    )
    expected = _manual_merged_outputs(
        anchor,
        experts,
        hidden_states,
        endpoint_weights,
    )
    assert torch.allclose(endpoint_result.output, expected, atol=1e-6, rtol=1e-6)


def test_learned_weights_and_lora_factors_remain_differentiable() -> None:
    torch.manual_seed(613)
    anchor = NamedProjectionMLP().eval()
    experts = _experts(anchor)
    head = LearnedMixtureWeightHead(
        weight=torch.tensor([0.7, -0.4]),
        bias=torch.tensor(0.2),
    )
    policy = LoRAParameterSpaceMerge(head)
    hidden_states = torch.randn(2, 4, requires_grad=True)
    selected_scores = torch.tensor(
        [[2.0, 1.0], [1.5, 0.5]],
        requires_grad=True,
    )
    weights = policy.pair_weights(selected_scores)

    result = policy.merge(
        hidden_states,
        anchor_mlp=anchor,
        experts=experts,
        weights=weights,
    )
    result.output.square().mean().backward()

    assert selected_scores.grad is not None
    assert hidden_states.grad is not None
    assert torch.isfinite(selected_scores.grad).all()
    assert all(
        expert.expert_mlp.get_parameter(name).grad is not None
        for expert in experts
        for name in expert.adapter_parameter_names
    )
    assert all(parameter.grad is None for parameter in anchor.parameters())


def test_parameter_merge_rejects_private_base_drift_and_invalid_weights() -> None:
    torch.manual_seed(617)
    anchor = NamedProjectionMLP().eval()
    experts = _experts(anchor)
    hidden_states = torch.randn(1, 4)
    policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(weight=torch.zeros(2), bias=torch.zeros(()))
    )
    with torch.no_grad():
        experts[1].expert_mlp.gate_proj.base_layer.weight.add_(0.01)

    with pytest.raises(ParameterSpaceMergeError, match="private base differs"):
        policy.merge(
            hidden_states,
            anchor_mlp=anchor,
            experts=experts,
            weights=torch.tensor([[0.5, 0.5]]),
        )
    with pytest.raises(ValueError, match="sum to one"):
        policy.merge(
            hidden_states,
            anchor_mlp=anchor,
            experts=(experts[0],),
            weights=torch.tensor([[0.5]]),
        )
    with pytest.raises(ValueError, match="sum to one"):
        ParameterSpaceMergeResult(
            output=torch.ones(1, 4),
            weights=torch.tensor([[0.2, 0.2]]),
            projection_names=("gate_proj",),
        )
