"""Acceptance tests for sparse anchor-veto top-2 dispatch."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    AnchoredTop2Dispatcher,
    ScoreSoftmaxCombination,
    Top2DispatchResult,
)


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


def test_top2_dispatch_executes_sparse_pairs_and_restores_order() -> None:
    anchor = _CountingLinear(1.0)
    code = _CountingLinear(2.0)
    math = _CountingLinear(-1.0)
    inactive = _CountingLinear(4.0)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    scores = torch.tensor(
        [[[-0.1, -0.2, -0.3], [2.0, 1.0, -1.0], [0.5, 1.5, -2.0], [1.0, -0.2, -0.5]]]
    )

    result = AnchoredTop2Dispatcher().dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=("code", "math", "inactive"),
        anchor_mlp=anchor,
        experts=(code, math, inactive),
    )

    first_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    second_weights = torch.softmax(torch.tensor([1.5, 0.5]), dim=0)
    expected = torch.stack(
        (
            hidden[0, 0],
            first_weights[0] * (hidden[0, 1] * 2.0)
            + first_weights[1] * (hidden[0, 1] * -1.0),
            second_weights[0] * (hidden[0, 2] * -1.0)
            + second_weights[1] * (hidden[0, 2] * 2.0),
            hidden[0, 3],
        )
    ).unsqueeze(0)
    assert torch.allclose(result.output, expected)
    assert anchor.calls == [2]
    assert code.calls == [2]
    assert math.calls == [2]
    assert inactive.calls == []
    assert torch.equal(
        result.selected_routes.selected_indices,
        torch.tensor([[[0, 1], [1, 2], [2, 1], [1, 0]]]),
    )
    assert torch.equal(
        result.mixture_weights[result.selected_routes.anchor_mask],
        torch.zeros(2, 2),
    )
    assert torch.allclose(
        result.mixture_weights[result.selected_routes.expert_pair_mask],
        torch.stack((first_weights, second_weights)),
    )
    assert result.output.shape == hidden.shape


def test_empty_and_single_expert_registries_execute_only_anchor() -> None:
    hidden = torch.randn(2, 3, 2)
    empty_anchor = _CountingLinear(1.0)

    empty = AnchoredTop2Dispatcher().dispatch(
        hidden,
        expert_scores=torch.empty(2, 3, 0),
        expert_ids=(),
        anchor_mlp=empty_anchor,
        experts=(),
    )

    assert torch.equal(empty.output, hidden)
    assert empty.selected_routes.anchor_mask.all()
    assert torch.equal(empty.mixture_weights, torch.zeros(2, 3, 2))
    assert empty_anchor.calls == [6]

    single_anchor = _CountingLinear(1.0)
    single_expert = _CountingLinear(3.0)
    single = AnchoredTop2Dispatcher().dispatch(
        hidden,
        expert_scores=torch.full((2, 3, 1), 4.0),
        expert_ids=("only",),
        anchor_mlp=single_anchor,
        experts=(single_expert,),
    )

    assert torch.equal(single.output, hidden)
    assert single.selected_routes.anchor_mask.all()
    assert single_anchor.calls == [6]
    assert single_expert.calls == []


def test_top2_dispatch_preserves_score_expert_and_hidden_gradients() -> None:
    anchor = nn.Linear(2, 2, bias=False)
    code = nn.Linear(2, 2, bias=False)
    math = nn.Linear(2, 2, bias=False)
    anchor.requires_grad_(False)
    hidden = torch.randn(3, 2, requires_grad=True)
    scores = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0], [0.7, 0.8]],
        requires_grad=True,
    )

    result = AnchoredTop2Dispatcher().dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=("code", "math"),
        anchor_mlp=anchor,
        experts=(code, math),
    )
    result.output.square().sum().backward()

    assert hidden.grad is not None
    assert scores.grad is not None
    assert code.weight.grad is not None
    assert math.weight.grad is not None
    assert anchor.weight.grad is None
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(scores.grad).all()


def test_top2_dispatch_uses_configured_mixture_temperature() -> None:
    hidden = torch.tensor([[1.0, 2.0]])
    anchor = _CountingLinear(1.0)
    code = _CountingLinear(2.0)
    math = _CountingLinear(-1.0)
    dispatcher = AnchoredTop2Dispatcher(
        combination_policy=ScoreSoftmaxCombination(mix_temperature=2.0)
    )

    result = dispatcher.dispatch(
        hidden,
        expert_scores=torch.tensor([[2.0, 1.0]]),
        expert_ids=("code", "math"),
        anchor_mlp=anchor,
        experts=(code, math),
    )

    expected_weights = torch.softmax(torch.tensor([1.0, 0.5]), dim=0)
    assert torch.allclose(result.mixture_weights[0], expected_weights)
    assert anchor.calls == []


def test_top2_dispatch_validates_inputs_before_module_execution() -> None:
    anchor = _CountingLinear(1.0)
    code = _CountingLinear(2.0)
    math = _CountingLinear(-1.0)
    dispatcher = AnchoredTop2Dispatcher()

    with pytest.raises(ValueError, match="token shape"):
        dispatcher.dispatch(
            torch.randn(2, 2),
            expert_scores=torch.randn(3, 2),
            expert_ids=("code", "math"),
            anchor_mlp=anchor,
            experts=(code, math),
        )
    with pytest.raises(ValueError, match="share one dtype"):
        dispatcher.dispatch(
            torch.randn(2, 2, dtype=torch.float32),
            expert_scores=torch.randn(2, 2, dtype=torch.float64),
            expert_ids=("code", "math"),
            anchor_mlp=anchor,
            experts=(code, math),
        )
    with pytest.raises(TypeError, match="floating-point"):
        dispatcher.dispatch(
            torch.randn(2, 2),
            expert_scores=torch.ones(2, 2, dtype=torch.int64),
            expert_ids=("code", "math"),
            anchor_mlp=anchor,
            experts=(code, math),
        )

    assert anchor.calls == []
    assert code.calls == []
    assert math.calls == []


def test_top2_dispatch_rejects_incompatible_policy_types() -> None:
    with pytest.raises(TypeError, match="selection_policy"):
        AnchoredTop2Dispatcher(
            selection_policy=AnchoredTop1Selection(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="combination_policy"):
        AnchoredTop2Dispatcher(
            combination_policy=object(),  # type: ignore[arg-type]
        )


class _WrongWidth(nn.Module):
    """Return an incompatible trailing output shape."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden_states, hidden_states[:, :1]), dim=-1)


def test_top2_dispatch_rejects_inconsistent_candidate_outputs() -> None:
    with pytest.raises(ValueError, match="trailing shape"):
        AnchoredTop2Dispatcher().dispatch(
            torch.randn(2, 2),
            expert_scores=torch.ones(2, 2),
            expert_ids=("code", "math"),
            anchor_mlp=_CountingLinear(1.0),
            experts=(_WrongWidth(), _CountingLinear(2.0)),
        )


def test_top2_dispatch_result_rejects_invalid_public_construction() -> None:
    routes = AnchorVetoTop2Selection()(
        torch.tensor([[-0.2, -0.1], [0.8, 0.7]]),
        expert_ids=("code", "math"),
    )

    with pytest.raises(ValueError, match="pair shape"):
        Top2DispatchResult(
            output=torch.ones(2, 2),
            selected_routes=routes,
            mixture_weights=torch.ones(2, 3),
        )
    with pytest.raises(ValueError, match="zero specialist"):
        Top2DispatchResult(
            output=torch.ones(2, 2),
            selected_routes=routes,
            mixture_weights=torch.full((2, 2), 0.5),
        )
    with pytest.raises(ValueError, match="sum to one"):
        Top2DispatchResult(
            output=torch.ones(2, 2),
            selected_routes=routes,
            mixture_weights=torch.tensor([[0.0, 0.0], [0.25, 0.25]]),
        )
