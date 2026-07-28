"""COM-003 integration with pair-shaped sparse top-2 dispatch."""

from __future__ import annotations

import math

import torch
from torch import nn

from anchored_moe import (
    AnchoredTop2Dispatcher,
    LearnedMixtureWeightCombination,
    LearnedMixtureWeightHead,
    ScoreSoftmaxCombination,
)


def _scaled_identity(scale: float) -> nn.Linear:
    module = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2) * scale)
    return module


def test_learned_weights_change_combination_without_changing_selected_routes() -> None:
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expert_scores = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    expert_ids = ("code:0", "math:0")
    anchor = _scaled_identity(1.0)
    experts = (_scaled_identity(2.0), _scaled_identity(-1.0))
    learned_head = LearnedMixtureWeightHead(
        weight=torch.zeros(2),
        bias=torch.tensor(math.log(3.0)),
    )

    learned = AnchoredTop2Dispatcher(
        combination_policy=LearnedMixtureWeightCombination(learned_head),
    ).dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=experts,
    )
    score_softmax = AnchoredTop2Dispatcher(
        combination_policy=ScoreSoftmaxCombination(),
    ).dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=experts,
    )

    assert torch.equal(
        learned.selected_routes.selected_indices,
        score_softmax.selected_routes.selected_indices,
    )
    assert torch.equal(
        learned.selected_routes.selected_scores,
        score_softmax.selected_routes.selected_scores,
    )
    assert torch.allclose(
        learned.mixture_weights,
        torch.tensor([[0.75, 0.25], [0.75, 0.25]]),
    )
    assert not torch.equal(
        learned.mixture_weights,
        score_softmax.mixture_weights,
    )
    assert torch.allclose(
        learned.output,
        torch.stack((hidden_states[0] * 1.25, hidden_states[1] * -0.25)),
    )
