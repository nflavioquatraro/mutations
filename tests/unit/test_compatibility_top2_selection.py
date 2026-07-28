"""Acceptance tests for compatibility-constrained global top-2 selection."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    CompatibilityConstrainedTop2Selection,
    MixtureCompatibilityIndex,
)


def _policy() -> CompatibilityConstrainedTop2Selection:
    """Return one policy with two explicit groups and one non-mixable expert."""

    return CompatibilityConstrainedTop2Selection(
        MixtureCompatibilityIndex(
            expert_ids=("a", "b", "c", "d", "solo"),
            group_ids=("alpha", "beta", "beta", "alpha", None),
        )
    )


def test_policy_selects_best_compatible_pair_not_incompatible_top_scores() -> None:
    scores = torch.tensor(
        [
            [0.90, 0.80, 0.70, -0.10, 1.00],
            [0.50, 0.50, 0.50, 0.50, -1.00],
            [0.90, 0.20, 0.00, 0.00, 0.10],
            [-0.50, -0.20, -0.10, -0.40, -0.30],
        ],
        dtype=torch.float64,
    )
    original = scores.clone()

    decision = _policy()(scores, expert_ids=("a", "b", "c", "d", "solo"))

    # Row 0 ignores the strongest non-mixable expert and the incompatible
    # highest specialist, selecting the positive beta pair instead.
    # Row 1 has two equal pair score tuples; append-order pair alpha wins.
    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[2, 3], [1, 4], [1, 0], [0, 3]]),
    )
    assert torch.equal(
        decision.selected_scores,
        torch.tensor(
            [
                [0.80, 0.70],
                [0.50, 0.50],
                [0.90, 0.00],
                [0.00, -0.10],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        decision.expert_pair_mask,
        torch.tensor([True, True, False, False]),
    )
    assert torch.equal(scores, original)
    assert decision.selected_scores.dtype == scores.dtype
    assert decision.selected_scores.device == scores.device


def test_policy_realigns_groups_for_candidate_order() -> None:
    index = MixtureCompatibilityIndex(
        expert_ids=("early", "late", "other"),
        group_ids=("shared", "shared", None),
    )
    policy = CompatibilityConstrainedTop2Selection(index)

    decision = policy(
        torch.tensor([[0.7, 0.9, 1.2]]),
        expert_ids=("late", "early", "other"),
    )

    assert decision.expert_ids == ("late", "early", "other")
    assert torch.equal(decision.selected_indices, torch.tensor([[2, 1]]))
    assert torch.equal(decision.selected_scores, torch.tensor([[0.9, 0.7]]))


def test_empty_and_single_nonmixable_candidates_fall_back_to_anchor() -> None:
    empty = CompatibilityConstrainedTop2Selection(
        MixtureCompatibilityIndex((), ())
    )(torch.empty(2, 0), expert_ids=())
    assert torch.equal(empty.selected_indices, torch.zeros(2, 2, dtype=torch.long))
    assert empty.anchor_mask.all()

    single = CompatibilityConstrainedTop2Selection(
        MixtureCompatibilityIndex(("only",), (None,))
    )(torch.tensor([[0.4], [0.0]]), expert_ids=("only",))
    assert torch.equal(single.selected_indices, torch.tensor([[1, 0], [0, 1]]))
    assert single.anchor_mask.all()


def test_policy_rejects_invalid_or_stale_inputs() -> None:
    policy = _policy()

    with pytest.raises(TypeError, match="MixtureCompatibilityIndex"):
        CompatibilityConstrainedTop2Selection(  # type: ignore[arg-type]
            {"a": "group"}
        )
    with pytest.raises(ValueError, match="exactly match"):
        policy(torch.ones(1, 4), expert_ids=("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="finite"):
        policy(
            torch.tensor([[0.1, 0.2, float("nan"), 0.3, 0.4]]),
            expert_ids=("a", "b", "c", "d", "solo"),
        )
    with pytest.raises(TypeError, match="floating-point"):
        policy(
            torch.ones(1, 5, dtype=torch.long),
            expert_ids=("a", "b", "c", "d", "solo"),
        )
