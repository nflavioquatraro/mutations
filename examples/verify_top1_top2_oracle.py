"""Verify CAL-007 and EVAL-004 with deterministic local losses."""

from __future__ import annotations

import json

import torch

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    compare_top1_top2_to_oracle,
)


def main() -> None:
    """Compare routed top-1 and top-2 decisions with a sparse route oracle."""

    expert_ids = ("code", "math")
    router_scores = torch.tensor(
        [
            [-1.0, -2.0],
            [2.0, 1.0],
            [0.5, 1.5],
            [1.5, 1.0],
        ]
    )
    top1 = AnchoredTop1Selection()(router_scores, expert_ids=expert_ids)
    top2 = AnchorVetoTop2Selection()(router_scores, expert_ids=expert_ids)

    # Columns are anchor, code, math, and the sampled code/math mixture. The
    # final row deliberately makes mixing worse than selected top-1.
    losses = CounterfactualRouteLosses(
        torch.tensor(
            [
                [0.5, 0.9, 1.0, 0.8],
                [1.0, 0.8, 0.9, 0.2],
                [1.0, 0.9, 0.8, 0.2],
                [1.0, 0.4, 0.6, 0.7],
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
    comparison = compare_top1_top2_to_oracle(losses, top1, top2)
    summary = {
        "destructive_mixture_fraction": comparison.destructive_mixture_fraction,
        "mean_oracle_pair_gain": comparison.mean_oracle_pair_gain,
        "mean_second_expert_gain": comparison.mean_second_expert_gain,
        "mean_top1_regret": comparison.mean_top1_regret,
        "mean_top2_regret": comparison.mean_top2_regret,
    }

    assert comparison.mean_top2_regret < comparison.mean_top1_regret
    assert comparison.destructive_mixture_fraction == 1.0 / 3.0
    print(
        "Counterfactual top-1/top-2 oracle validation passed: "
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
