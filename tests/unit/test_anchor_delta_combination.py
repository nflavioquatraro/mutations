"""Acceptance tests for COM-004 anchor-plus-delta combination."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import AlwaysOnAnchorDeltaCombination


def test_combination_preserves_anchor_and_applies_top1_and_top2_deltas() -> None:
    anchor = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    deltas = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, -1.0], [0.0, 0.0]],
            [[1.0, 2.0], [-2.0, 1.0]],
        ],
        requires_grad=True,
    )
    scores = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]],
        requires_grad=True,
    )
    arity = torch.tensor([0, 1, 2])

    result = AlwaysOnAnchorDeltaCombination()(
        anchor,
        deltas,
        selected_scores=scores,
        route_arity=arity,
    )

    pair_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    expected = torch.stack(
        (
            anchor[0],
            anchor[1] + deltas[1, 0],
            anchor[2]
            + pair_weights[0] * deltas[2, 0]
            + pair_weights[1] * deltas[2, 1],
        )
    )
    assert torch.allclose(result.output, expected)
    assert torch.equal(result.output[0], anchor[0])
    assert torch.equal(
        result.weights,
        torch.stack(
            (
                torch.tensor([0.0, 0.0]),
                torch.tensor([1.0, 0.0]),
                pair_weights,
            )
        ),
    )

    result.output.square().sum().backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert deltas.grad is not None and torch.isfinite(deltas.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad[:2], torch.zeros_like(scores.grad[:2]))


def test_combination_rejects_nonzero_inactive_delta_slots() -> None:
    anchor = torch.ones(2, 3)
    deltas = torch.zeros(2, 2, 3)
    deltas[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="inactive"):
        AlwaysOnAnchorDeltaCombination().combine(
            anchor,
            deltas,
            selected_scores=torch.zeros(2, 2),
            route_arity=torch.zeros(2, dtype=torch.long),
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), True])
def test_combination_rejects_invalid_temperature(temperature: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AlwaysOnAnchorDeltaCombination(temperature)  # type: ignore[arg-type]
