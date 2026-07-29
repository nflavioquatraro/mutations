"""Acceptance tests for pair-only COM-005 routed execution."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from anchored_moe import (
    AnchoredLoRAParameterMergeDispatcher,
    LearnedMixtureWeightHead,
    LoRAExpertConfig,
    LoRAParameterSpaceMerge,
    PEFTLoRAExpert,
    ParameterSpaceMergeError,
)


class NamedProjectionMLP(nn.Module):
    """Small nonlinear MLP using the supported projection names."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.gate_proj = nn.Linear(3, 5, bias=False)
        self.up_proj = nn.Linear(3, 5, bias=False)
        self.down_proj = nn.Linear(5, 3, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply a gated MLP transformation."""

        self.calls += 1
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _experts(anchor: nn.Module) -> tuple[PEFTLoRAExpert, PEFTLoRAExpert]:
    """Return two compatible experts with distinct effective updates."""

    config = LoRAExpertConfig(
        rank=2,
        alpha=4,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    experts = (
        PEFTLoRAExpert(anchor, config=config),
        PEFTLoRAExpert(anchor, config=config),
    )
    with torch.no_grad():
        for expert_index, expert in enumerate(experts, start=1):
            for name in expert.adapter_parameter_names:
                if ".lora_A." in name:
                    expert.expert_mlp.get_parameter(name).fill_(0.1)
                else:
                    expert.expert_mlp.get_parameter(name).fill_(
                        0.05 * expert_index
                    )
    return experts


def test_dispatch_preserves_anchor_and_applies_top1_and_ranked_top2_weights() -> None:
    torch.manual_seed(631)
    anchor = NamedProjectionMLP().eval()
    experts = _experts(anchor)
    hidden_states = torch.randn(3, 3)
    scores = torch.tensor(
        [[-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]],
        requires_grad=True,
    )
    availability = torch.tensor(
        [[True, True], [True, False], [True, True]]
    )
    merge_policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(
            weight=torch.zeros(2),
            bias=torch.tensor(math.log(3.0)),
        )
    )
    dispatcher = AnchoredLoRAParameterMergeDispatcher(
        merge_policy=merge_policy
    )

    result = dispatcher.dispatch(
        hidden_states,
        expert_scores=scores,
        expert_ids=("code-a", "code-b"),
        anchor_mlp=anchor,
        experts=experts,
        expert_availability=availability,
    )
    expected_top1 = merge_policy.merge(
        hidden_states[1:2],
        anchor_mlp=anchor,
        experts=(experts[0],),
        weights=torch.ones(1, 1),
    ).output
    # Token two ranks expert B first, so the learned 0.75 coefficient applies
    # to expert B and the second coefficient applies to expert A.
    expected_top2 = merge_policy.merge(
        hidden_states[2:3],
        anchor_mlp=anchor,
        experts=(experts[1], experts[0]),
        weights=torch.tensor([[0.75, 0.25]]),
    ).output

    assert torch.equal(result.output[0], anchor(hidden_states[0:1])[0])
    assert torch.allclose(result.output[1:2], expected_top1)
    assert torch.allclose(result.output[2:3], expected_top2)
    assert torch.equal(result.capacity.route_arity, torch.tensor([0, 1, 2]))
    assert torch.equal(
        result.capacity.effective_indices,
        torch.tensor([[0, 0], [1, 0], [2, 1]]),
    )
    assert torch.allclose(
        result.merge_weights,
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.75, 0.25]]),
    )
    with pytest.raises(ValueError, match="Anchor rows"):
        replace(
            result,
            merge_weights=result.merge_weights.new_ones(
                result.merge_weights.shape
            ),
        )

    result.output.square().sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_dispatch_rejects_non_lora_or_non_pair_topology_before_execution() -> None:
    anchor = NamedProjectionMLP().eval()
    experts = _experts(anchor)
    policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(weight=torch.zeros(2), bias=torch.zeros(()))
    )
    dispatcher = AnchoredLoRAParameterMergeDispatcher(merge_policy=policy)
    hidden_states = torch.randn(1, 3)

    with pytest.raises(ValueError, match="exactly two"):
        dispatcher.dispatch(
            hidden_states,
            expert_scores=torch.ones(1, 1),
            expert_ids=("only",),
            anchor_mlp=anchor,
            experts=(experts[0],),
        )
    with pytest.raises(TypeError, match="PEFTLoRAExpert"):
        dispatcher.dispatch(
            hidden_states,
            expert_scores=torch.ones(1, 2),
            expert_ids=("a", "b"),
            anchor_mlp=anchor,
            experts=(experts[0], nn.Identity()),  # type: ignore[arg-type]
        )

    with torch.no_grad():
        experts[1].expert_mlp.up_proj.base_layer.weight.add_(0.01)
    with pytest.raises(ParameterSpaceMergeError, match="private base differs"):
        dispatcher.dispatch(
            torch.randn(2, 3),
            expert_scores=torch.tensor([[-1.0, -2.0], [2.0, 1.0]]),
            expert_ids=("a", "b"),
            anchor_mlp=anchor,
            experts=experts,
        )
    assert anchor.calls == 0
