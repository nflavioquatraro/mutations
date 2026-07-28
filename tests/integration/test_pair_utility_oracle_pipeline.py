"""Integration of COM-001 losses with CAL-003, CAL-007, and EVAL-004."""

from __future__ import annotations

import torch

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    ScoreSoftmaxCombination,
    compare_top1_top2_to_oracle,
    compute_pair_utilities,
    plan_expert_pairs,
)


def test_pair_mixture_losses_flow_into_oracle_comparison() -> None:
    expert_ids = ("code", "math")
    scores = torch.tensor([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
    expert_outputs = torch.tensor(
        [
            [[0.0], [2.0]],
            [[2.0], [2.0]],
            [[4.0], [0.0]],
        ]
    )
    targets = torch.tensor([[0.0], [2.0], [1.0]])
    anchor_outputs = torch.tensor([[1.0], [0.0], [2.0]])

    combination = ScoreSoftmaxCombination()(
        expert_outputs,
        selected_scores=scores,
    )
    anchor_losses = (anchor_outputs - targets).square().mean(dim=-1)
    single_losses = (expert_outputs - targets.unsqueeze(-2)).square().mean(dim=-1)
    pair_losses = (
        (combination.output - targets).square().mean(dim=-1).unsqueeze(-1)
    )
    plan = plan_expert_pairs(expert_ids, max_pairs=1, seed=7)
    pair_utility = compute_pair_utilities(
        plan,
        anchor_losses=anchor_losses,
        single_losses=single_losses,
        pair_losses=pair_losses,
        loss_name="mean_squared_error",
    )
    comparison = compare_top1_top2_to_oracle(
        pair_utility.to_counterfactual_route_losses(),
        AnchoredTop1Selection()(scores, expert_ids=expert_ids),
        AnchorVetoTop2Selection()(scores, expert_ids=expert_ids),
    )

    assert plan.pairs[0].expert_ids == ("code", "math")
    assert torch.allclose(pair_utility.best_pair_losses, pair_losses.squeeze(-1))
    assert torch.allclose(comparison.best_pair_losses, pair_losses.squeeze(-1))
    assert comparison.top2.selected_candidate_indices.tolist() == [3, 3, 3]
    assert 0.0 <= comparison.destructive_mixture_fraction <= 1.0
