"""Acceptance tests for sparse anchored top-1 dispatch."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anchored_moe import AnchoredTop1Dispatcher


class _CountingLinear(nn.Linear):
    """Linear test double recording selected token counts."""

    def __init__(self, scale: float) -> None:
        super().__init__(2, 2, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(2) * scale)
        self.calls: list[int] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append(hidden_states.shape[0])
        return super().forward(hidden_states)


def test_dispatch_executes_only_selected_candidates_and_restores_order() -> None:
    anchor = _CountingLinear(1.0)
    code = _CountingLinear(2.0)
    math = _CountingLinear(-1.0)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    scores = torch.tensor(
        [[[-0.1, -0.2], [0.7, 0.1], [0.2, 0.8], [0.6, 0.5]]]
    )

    result = AnchoredTop1Dispatcher().dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=("code", "math"),
        anchor_mlp=anchor,
        experts=(code, math),
    )

    expected = torch.stack(
        (
            hidden[0, 0],
            hidden[0, 1] * 2.0,
            hidden[0, 2] * -1.0,
            hidden[0, 3] * 2.0,
        )
    ).unsqueeze(0)
    assert torch.equal(result.output, expected)
    assert anchor.calls == [1]
    assert code.calls == [2]
    assert math.calls == [1]
    assert torch.equal(
        result.capacity.routes.selected_indices,
        torch.tensor([[0, 1, 2, 1]]),
    )


def test_dispatch_overflow_executes_anchor_and_never_drops_tokens() -> None:
    anchor = _CountingLinear(1.0)
    expert = _CountingLinear(3.0)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = AnchoredTop1Dispatcher().dispatch(
        hidden,
        expert_scores=torch.tensor([[0.5], [0.7]]),
        expert_ids=("code",),
        anchor_mlp=anchor,
        experts=(expert,),
        expert_availability=torch.tensor([[False], [True]]),
    )

    assert torch.equal(
        result.output,
        torch.tensor([[1.0, 2.0], [9.0, 12.0]]),
    )
    assert anchor.calls == [1]
    assert expert.calls == [1]
    assert torch.equal(
        result.capacity.overflow_mask,
        torch.tensor([True, False]),
    )


def test_empty_registry_is_exact_anchor_execution() -> None:
    anchor = _CountingLinear(1.0)
    hidden = torch.randn(2, 3, 2)

    result = AnchoredTop1Dispatcher().dispatch(
        hidden,
        expert_scores=torch.empty(2, 3, 0),
        expert_ids=(),
        anchor_mlp=anchor,
        experts=(),
    )

    assert torch.equal(result.output, hidden)
    assert result.capacity.routes.anchor_mask.all()
    assert anchor.calls == [6]


def test_dispatch_preserves_expert_and_hidden_gradients() -> None:
    anchor = nn.Linear(2, 2, bias=False)
    expert = nn.Linear(2, 2, bias=False)
    anchor.requires_grad_(False)
    hidden = torch.randn(3, 2, requires_grad=True)

    result = AnchoredTop1Dispatcher().dispatch(
        hidden,
        expert_scores=torch.ones(3, 1),
        expert_ids=("code",),
        anchor_mlp=anchor,
        experts=(expert,),
    )
    result.output.square().sum().backward()

    assert hidden.grad is not None
    assert expert.weight.grad is not None
    assert anchor.weight.grad is None


def test_dispatch_validates_score_shape_before_module_execution() -> None:
    anchor = _CountingLinear(1.0)

    with pytest.raises(ValueError, match="token shape"):
        AnchoredTop1Dispatcher().dispatch(
            torch.randn(2, 2),
            expert_scores=torch.randn(3, 0),
            expert_ids=(),
            anchor_mlp=anchor,
            experts=(),
        )

    assert anchor.calls == []
