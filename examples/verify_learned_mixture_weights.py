"""Fit and deploy COM-003 learned pair-mixture weights without model downloads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from anchored_moe import (
    LearnedMixtureWeightCombination,
    LearnedMixtureWeightHead,
    MixtureWeightFittingConfig,
    fit_learned_mixture_head,
    plan_expert_pairs,
)


def main() -> None:
    """Verify deterministic fitting, combination, and safe reload."""

    plan = plan_expert_pairs(("code:0", "code:1", "math:0"), max_pairs=None)
    selected_scores = torch.tensor(
        [
            [-2.0, 1.0],
            [-1.0, 0.5],
            [0.0, 0.0],
            [0.5, -1.0],
            [1.0, 0.5],
            [2.0, -1.0],
        ]
    )
    oracle_first_weight = torch.sigmoid(
        1.5 * selected_scores[:, 0]
        - 0.75 * selected_scores[:, 1]
        + 0.3
    )
    oracle_weights = torch.stack(
        (oracle_first_weight, 1.0 - oracle_first_weight),
        dim=-1,
    )
    pair_indices = torch.tensor([0, 1, 2, 0, 1, 2])

    head, report = fit_learned_mixture_head(
        plan,
        selected_scores=selected_scores,
        target_weights=oracle_weights,
        pair_indices=pair_indices,
        config=MixtureWeightFittingConfig(
            learning_rate=0.1,
            optimization_steps=300,
            l2_weight=0.0,
        ),
    )
    pair_outputs = torch.stack(
        (
            torch.tensor([[2.0, 4.0]]).expand(selected_scores.shape[0], -1),
            torch.tensor([[-1.0, -2.0]]).expand(selected_scores.shape[0], -1),
        ),
        dim=1,
    )
    combined = LearnedMixtureWeightCombination(head)(
        pair_outputs,
        selected_scores=selected_scores,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        payload = Path(temporary_directory) / "mixture-head.safetensors"
        head.save_safetensors(payload)
        restored = LearnedMixtureWeightHead.from_safetensors(payload)
        assert torch.equal(restored(selected_scores), head(selected_scores))

    assert report.final_cross_entropy < report.initial_cross_entropy
    assert report.final_mean_absolute_error < 0.02
    assert torch.allclose(combined.weights, oracle_weights, atol=0.03, rtol=0.0)
    assert combined.output.shape == (selected_scores.shape[0], 2)
    print(
        "COM-003 validation passed: "
        f"rows={report.row_count}, "
        f"pair_counts={report.pair_counts}, "
        f"final_mae={report.final_mean_absolute_error:.6f}"
    )


if __name__ == "__main__":
    main()
