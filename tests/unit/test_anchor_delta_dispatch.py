"""Acceptance tests for sparse always-on-anchor delta dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AnchorRelativeExpert,
    AlwaysOnAnchorDeltaDispatcher,
    AnchoredTop1Selection,
)


class _CountingIdentity(nn.Module):
    """Identity anchor recording the number of executed tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def forward(self, hidden_states: Tensor) -> Tensor:
        self.calls.append(hidden_states.shape[0])
        return hidden_states


class _ScalingDeltaExpert(AnchorRelativeExpert):
    """Complete identity-plus-scale expert with an explicit delta path."""

    parameterization = "test_scaling_delta"

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.delta_calls: list[int] = []

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        return hidden_states + self.scale * hidden_states

    def forward_delta(
        self,
        hidden_states: Tensor,
        *args: Any,
        anchor_output: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        self.delta_calls.append(hidden_states.shape[0])
        assert torch.equal(anchor_output, hidden_states)
        return self.scale * hidden_states

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        raise NotImplementedError


def test_adaptive_delta_dispatch_is_sparse_and_preserves_exact_fallback() -> None:
    anchor = _CountingIdentity()
    code = _ScalingDeltaExpert(1.0)
    math = _ScalingDeltaExpert(-0.5)
    hidden = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        requires_grad=True,
    )
    scores = torch.tensor(
        [[-1.0, -2.0], [2.0, -1.0], [2.0, 1.0], [2.0, 1.0]],
        requires_grad=True,
    )
    availability = torch.tensor(
        [[True, True], [True, True], [True, True], [False, True]]
    )
    dispatcher = AlwaysOnAnchorDeltaDispatcher(
        selection_policy=AdaptiveTop1Top2Selection(mix_threshold=0.5),
    )

    result = dispatcher.dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=("code", "math"),
        anchor_mlp=anchor,
        experts=(code, math),
        expert_availability=availability,
    )

    pair_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    expected = torch.stack(
        (
            hidden[0],
            hidden[1] + hidden[1],
            hidden[2]
            + pair_weights[0] * hidden[2]
            + pair_weights[1] * (-0.5 * hidden[2]),
            hidden[3] - 0.5 * hidden[3],
        )
    )
    assert torch.allclose(result.output, expected)
    assert torch.equal(result.output[0], hidden[0])
    assert anchor.calls == [4]
    assert code.delta_calls == [2]
    assert math.delta_calls == [2]
    assert torch.equal(
        result.capacity.route_arity,
        torch.tensor([0, 1, 2, 1]),
    )
    assert torch.equal(
        result.capacity.effective_indices,
        torch.tensor([[0, 0], [1, 0], [1, 2], [2, 0]]),
    )
    assert torch.equal(
        result.capacity.overflow_mask,
        torch.tensor(
            [[False, False], [False, False], [False, False], [True, False]]
        ),
    )
    assert torch.allclose(result.delta_weights[2], pair_weights)
    assert torch.equal(result.delta_weights[3], torch.tensor([1.0, 0.0]))

    result.output.square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert code.scale.grad is not None
    assert math.scale.grad is not None


def test_top1_delta_dispatch_overflow_returns_to_exact_anchor() -> None:
    anchor = _CountingIdentity()
    expert = _ScalingDeltaExpert(2.0)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scores = torch.tensor([[3.0], [3.0]])

    result = AlwaysOnAnchorDeltaDispatcher(
        selection_policy=AnchoredTop1Selection(),
    ).dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=("code",),
        anchor_mlp=anchor,
        experts=(expert,),
        expert_availability=torch.tensor([[True], [False]]),
    )

    assert torch.equal(result.output[0], hidden[0] * 3.0)
    assert torch.equal(result.output[1], hidden[1])
    assert anchor.calls == [2]
    assert expert.delta_calls == [1]
    assert torch.equal(result.capacity.route_arity, torch.tensor([1, 0]))
