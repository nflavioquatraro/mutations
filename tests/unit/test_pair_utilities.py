"""Acceptance tests for CAL-003 pair planning and utility construction."""

from __future__ import annotations

import json

import pytest
import torch

from anchored_moe import (
    CounterfactualRoute,
    compute_pair_utilities,
    plan_expert_pairs,
)


def test_bounded_pair_sampling_is_seeded_stable_and_canonical() -> None:
    expert_ids = ("code", "math", "bio", "legal", "finance")

    first = plan_expert_pairs(expert_ids, max_pairs=3, seed=17)
    repeated = plan_expert_pairs(expert_ids, max_pairs=3, seed=17)
    changed_seed = plan_expert_pairs(expert_ids, max_pairs=3, seed=18)

    assert first == repeated
    assert first.pairs == (
        CounterfactualRoute.pair("code", "bio"),
        CounterfactualRoute.pair("math", "finance"),
        CounterfactualRoute.pair("bio", "legal"),
    )
    assert first.pairs != changed_seed.pairs
    assert first.pair_count == 3
    assert first.available_pair_count == 10
    assert first.is_exhaustive is False
    assert first.sampling_mode == "bounded"
    json.dumps(first.to_dict(), allow_nan=False)
    assert first.to_dict()["seed"] == 17


def test_explicit_exhaustive_pair_plan_uses_registry_order() -> None:
    plan = plan_expert_pairs(
        ("code", "math", "bio"),
        max_pairs=None,
        seed=999,
    )

    assert plan.pairs == (
        CounterfactualRoute.pair("code", "math"),
        CounterfactualRoute.pair("code", "bio"),
        CounterfactualRoute.pair("math", "bio"),
    )
    assert plan.pair_count == 3
    assert plan.is_exhaustive
    assert plan.sampling_mode == "exhaustive"
    assert plan.index_for_pair(CounterfactualRoute.pair("code", "bio")) == 1


def test_default_pair_budget_avoids_exhaustive_quadratic_coverage() -> None:
    plan = plan_expert_pairs(tuple(f"expert-{index}" for index in range(12)))
    large_plan = plan_expert_pairs(
        tuple(f"large-{index}" for index in range(10_000)),
        max_pairs=2,
        seed=5,
    )

    assert plan.max_pairs == 64
    assert plan.pair_count == 64
    assert plan.available_pair_count == 66
    assert plan.is_exhaustive is False
    assert large_plan.pair_count == 2
    assert large_plan.available_pair_count == 49_995_000


def test_pair_utilities_report_anchor_advantage_and_best_single_gain() -> None:
    plan = plan_expert_pairs(
        ("code", "math", "bio"),
        max_pairs=None,
    )
    result = compute_pair_utilities(
        plan,
        anchor_losses=torch.tensor([1.0, 2.0]),
        single_losses=torch.tensor(
            [[0.8, 0.7, 0.9], [1.5, 1.4, 1.3]]
        ),
        pair_losses=torch.tensor(
            [[0.6, 0.7, 0.8], [1.3, 1.1, 1.1]]
        ),
        loss_name="cross_entropy",
    )

    assert torch.allclose(
        result.pair_utilities,
        torch.tensor([[0.4, 0.3, 0.2], [0.7, 0.9, 0.9]]),
    )
    assert torch.allclose(
        result.pair_improvements_over_best_single,
        torch.tensor([[0.1, 0.0, -0.1], [0.0, 0.2, 0.2]]),
    )
    assert torch.equal(result.best_pair_indices, torch.tensor([0, 1]))
    assert torch.allclose(result.best_pair_losses, torch.tensor([0.6, 1.1]))
    assert torch.allclose(result.best_pair_utilities, torch.tensor([0.4, 0.9]))
    assert torch.allclose(
        result.best_pair_improvements_over_best_single,
        torch.tensor([0.1, 0.2]),
    )
    assert torch.equal(
        result.pair_beats_best_single(),
        torch.tensor([[True, False, False], [False, True, True]]),
    )
    assert torch.equal(
        result.pair_beats_best_single(minimum_margin=0.15),
        torch.tensor([[False, False, False], [False, True, True]]),
    )


def test_pair_utilities_build_canonical_cal007_table_without_mutation() -> None:
    plan = plan_expert_pairs(("code", "math"), max_pairs=1, seed=4)
    anchor = torch.tensor([1.0], requires_grad=True)
    singles = torch.tensor([[0.8, 0.9]], requires_grad=True)
    pairs = torch.tensor([[0.7]], requires_grad=True)
    result = compute_pair_utilities(
        plan,
        anchor_losses=anchor,
        single_losses=singles,
        pair_losses=pairs,
        loss_name="cross_entropy",
    )

    anchor.detach()[0] = 99.0
    singles.detach()[0, 0] = 88.0
    pairs.detach()[0, 0] = 77.0
    first_read = result.pair_losses
    first_read[0, 0] = 66.0
    table = result.to_counterfactual_route_losses()

    assert result.anchor_losses.requires_grad is False
    assert torch.equal(result.anchor_losses, torch.tensor([1.0]))
    assert torch.equal(result.single_losses, torch.tensor([[0.8, 0.9]]))
    assert torch.equal(result.pair_losses, torch.tensor([[0.7]]))
    assert table.routes == (
        CounterfactualRoute.anchor(),
        CounterfactualRoute.single("code"),
        CounterfactualRoute.single("math"),
        CounterfactualRoute.pair("code", "math"),
    )
    assert torch.equal(table.losses, torch.tensor([[1.0, 0.8, 0.9, 0.7]]))


@pytest.mark.parametrize(
    ("expert_ids", "max_pairs", "seed", "error_type", "message"),
    [
        (("code",), 1, 0, ValueError, "at least two"),
        (("code", "code"), 1, 0, ValueError, "unique"),
        (("code", "math"), 0, 0, ValueError, "greater than zero"),
        (("code", "math"), True, 0, TypeError, "positive integer"),
        (("code", "math"), 1, -1, ValueError, "non-negative"),
    ],
)
def test_pair_sampling_rejects_invalid_configuration(
    expert_ids: tuple[str, ...],
    max_pairs: int | None,
    seed: int,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        plan_expert_pairs(expert_ids, max_pairs=max_pairs, seed=seed)


def test_pair_utilities_reject_misaligned_or_invalid_losses() -> None:
    plan = plan_expert_pairs(("code", "math"), max_pairs=1)
    with pytest.raises(ValueError, match="expert_count"):
        compute_pair_utilities(
            plan,
            anchor_losses=torch.ones(2),
            single_losses=torch.ones(2, 3),
            pair_losses=torch.ones(2, 1),
            loss_name="cross_entropy",
        )
    with pytest.raises(ValueError, match="finite"):
        compute_pair_utilities(
            plan,
            anchor_losses=torch.ones(2),
            single_losses=torch.ones(2, 2),
            pair_losses=torch.tensor([[1.0], [float("nan")]]),
            loss_name="cross_entropy",
        )
    with pytest.raises(ValueError, match="one dtype"):
        compute_pair_utilities(
            plan,
            anchor_losses=torch.ones(2, dtype=torch.float64),
            single_losses=torch.ones(2, 2),
            pair_losses=torch.ones(2, 1),
            loss_name="cross_entropy",
        )
    result = compute_pair_utilities(
        plan,
        anchor_losses=torch.ones(2),
        single_losses=torch.ones(2, 2),
        pair_losses=torch.ones(2, 1),
        loss_name="cross_entropy",
    )
    with pytest.raises(ValueError, match="non-negative"):
        result.pair_beats_best_single(minimum_margin=-0.1)
