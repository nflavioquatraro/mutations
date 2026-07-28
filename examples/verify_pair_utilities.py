"""Verify CAL-003 bounded pair planning and utility construction."""

from __future__ import annotations

import json

import torch

from anchored_moe import (
    compute_pair_utilities,
    evaluate_counterfactual_oracle,
    plan_expert_pairs,
)


def main() -> None:
    """Plan sparse pairs, compute utilities, and build a CAL-007 table."""

    expert_ids = ("code", "math", "bio", "legal", "finance")
    plan = plan_expert_pairs(
        expert_ids,
        max_pairs=3,
        seed=17,
    )
    pair_utility = compute_pair_utilities(
        plan,
        anchor_losses=torch.tensor([1.0, 1.0, 1.0]),
        single_losses=torch.tensor(
            [
                [0.8, 0.7, 0.9, 1.1, 1.0],
                [0.6, 0.9, 0.8, 0.7, 1.0],
                [0.9, 0.8, 0.7, 0.6, 0.5],
            ]
        ),
        pair_losses=torch.tensor(
            [
                [0.5, 0.8, 0.6],
                [0.7, 0.5, 0.8],
                [0.6, 0.7, 0.4],
            ]
        ),
        loss_name="cross_entropy",
    )
    counterfactual = pair_utility.to_counterfactual_route_losses()
    selected_pair_columns = (
        pair_utility.best_pair_indices + len(expert_ids) + 1
    )
    oracle = evaluate_counterfactual_oracle(
        counterfactual,
        selected_pair_columns,
    )

    assert [pair.expert_ids for pair in plan.pairs] == [
        ("code", "bio"),
        ("math", "finance"),
        ("bio", "legal"),
    ]
    assert plan.pair_count == 3
    assert plan.available_pair_count == 10
    assert counterfactual.pair_count == 3
    assert torch.equal(
        pair_utility.best_pair_indices,
        torch.tensor([0, 1, 2]),
    )

    summary = {
        "available_pairs": plan.available_pair_count,
        "best_pair_indices": pair_utility.best_pair_indices.tolist(),
        "mean_best_pair_gain": float(
            pair_utility.best_pair_improvements_over_best_single.mean().item()
        ),
        "oracle_mean_regret": oracle.mean_regret,
        "sampled_pairs": [list(pair.expert_ids) for pair in plan.pairs],
    }
    print(
        "CAL-003 pair-utility validation passed: "
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
