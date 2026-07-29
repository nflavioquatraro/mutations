"""Sparse anchored-selection integration for BAL-004."""

from __future__ import annotations

import torch
import pytest

from anchored_moe import (
    AnchoredTop1Selection,
    LossFreeBiasBalancer,
    LossFreeBiasConfig,
)


def _raw_scores() -> torch.Tensor:
    """Return heterogeneous evidence with an initial nine-to-one collapse."""

    first = torch.tensor(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.1]
    )
    second = torch.full_like(first, 0.1)
    return torch.stack((first, second), dim=-1)


def test_historical_bias_updates_balance_sparse_anchored_top1_selection(
    tmp_path,
) -> None:
    expert_ids = ("code:0", "code:1")
    selector = AnchoredTop1Selection()
    balancer = LossFreeBiasBalancer(
        expert_ids,
        config=LossFreeBiasConfig(update_rate=0.05),
    )
    raw_scores = _raw_scores()
    raw_snapshot = raw_scores.clone()
    reports = []

    for _ in range(5):
        score_views = balancer.apply_for_selection(raw_scores)
        routes = selector(
            score_views.selection_scores,
            expert_ids=score_views.expert_ids,
        )
        assert not routes.anchor_mask.any().item()
        reports.append(balancer.update(routes.selected_indices))

    assert reports[0].assignment_loads == (9.0, 1.0)
    assert reports[0].maximum_load_violation == pytest.approx(0.8)
    assert reports[-1].assignment_loads == (5.0, 5.0)
    assert reports[-1].maximum_load_violation == 0.0
    assert torch.equal(raw_scores, raw_snapshot)
    assert tuple(balancer.parameters()) == ()

    checkpoint = tmp_path / "balancer.safetensors"
    balancer.save_safetensors(checkpoint)
    resumed = LossFreeBiasBalancer.from_safetensors(checkpoint)
    uninterrupted_view = balancer.apply_for_selection(raw_scores)
    resumed_view = resumed.apply_for_selection(raw_scores)
    assert torch.equal(
        resumed_view.selection_scores,
        uninterrupted_view.selection_scores,
    )
    assert torch.equal(resumed_view.raw_scores, raw_scores)
    assert resumed.update_step == 5
