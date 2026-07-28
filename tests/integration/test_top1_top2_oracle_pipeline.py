"""Integration of routing policies with CAL-007 and EVAL-004."""

from __future__ import annotations

import torch

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    compare_top1_top2_to_oracle,
)


def test_router_decisions_map_to_counterfactual_oracle_candidates() -> None:
    scores = torch.tensor(
        [
            [-1.0, -2.0],
            [2.0, 1.0],
            [0.5, 1.5],
        ]
    )
    expert_ids = ("code", "math")
    top1 = AnchoredTop1Selection()(scores, expert_ids=expert_ids)
    top2 = AnchorVetoTop2Selection()(scores, expert_ids=expert_ids)
    losses = CounterfactualRouteLosses(
        torch.tensor(
            [
                [0.5, 0.9, 1.0, 0.8],
                [1.0, 0.4, 0.7, 0.3],
                [1.0, 0.8, 0.5, 0.4],
            ]
        ),
        expert_ids=expert_ids,
        routes=(
            CounterfactualRoute.anchor(),
            CounterfactualRoute.single("code"),
            CounterfactualRoute.single("math"),
            CounterfactualRoute.pair("code", "math"),
        ),
        loss_name="cross_entropy",
    )

    result = compare_top1_top2_to_oracle(losses, top1, top2)

    assert torch.equal(top1.selected_indices, torch.tensor([0, 1, 2]))
    assert torch.equal(
        top2.selected_indices,
        torch.tensor([[0, 1], [1, 2], [2, 1]]),
    )
    assert torch.equal(
        result.top2.selected_candidate_indices,
        torch.tensor([0, 3, 3]),
    )
    assert result.mean_top1_regret > result.mean_top2_regret
    assert result.destructive_mixture_fraction == 0.0
