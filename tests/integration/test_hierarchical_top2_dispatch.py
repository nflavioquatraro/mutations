"""End-to-end staged RTR-005/POL-004 execution through INF-002."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop2Dispatcher,
    AppendOnlyRouterHeads,
    HierarchicalPackageTop1LocalTop2Selection,
    HierarchicalRouterScoreBank,
    LogicalAppendOnlyRouter,
    Top2RouteDecision,
)


class _TrackedScale(nn.Module):
    """Apply one scalar while recording sparse token execution."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.token_count = 0

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Scale selected rows without changing their shape."""

        self.token_count += hidden_states.shape[0]
        return hidden_states * self.scale


def _logical_router(
    rows: tuple[tuple[str, tuple[float, ...]], ...],
) -> LogicalAppendOnlyRouter:
    """Compile explicit layer-zero affine rows."""

    bank = AppendOnlyRouterHeads(len(rows[0][1]))
    for identifier, weight in rows:
        head = AffineRouterHead(
            len(weight),
            layer_index=0,
            expert_id=identifier,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
        bank.append(head)
    return LogicalAppendOnlyRouter.from_independent_heads(bank)


def test_hierarchical_routes_reuse_sparse_top2_execution_without_cross_pairs() -> None:
    package_router = _logical_router(
        (("code-package", (1.0, 0.0)), ("math-package", (0.0, 1.0)))
    )
    hierarchy = HierarchicalRouterScoreBank(
        package_router,
        {
            "code-package": _logical_router(
                (("code-a", (1.0, 0.0)), ("code-b", (0.0, 1.0)))
            ),
            "math-package": _logical_router(
                (("math-a", (-1.0, 0.0)), ("math-b", (0.0, -1.0)))
            ),
        },
    )
    policy = HierarchicalPackageTop1LocalTop2Selection()
    hidden_states = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0], [1.0, 1.0]]
    )

    package_scores = hierarchy.score_packages(hidden_states, layer_index=0)
    package_routes = policy.select_packages(
        package_scores,
        package_ids=hierarchy.package_ids_for_layer(0),
    )
    local_scores = hierarchy.score_selected_locals(
        hidden_states,
        package_indices=package_routes.selected_indices,
        package_ids=package_routes.expert_ids,
        layer_index=0,
    )
    decision = policy.select_locals(
        package_routes,
        local_scores,
        expert_ids_by_package={
            package_id: hierarchy.expert_ids_for_package(
                package_id,
                layer_index=0,
            )
            for package_id in hierarchy.package_ids
        },
    )

    anchor = _TrackedScale(1.0)
    experts = (
        _TrackedScale(2.0),
        _TrackedScale(4.0),
        _TrackedScale(8.0),
        _TrackedScale(16.0),
    )
    result = AnchoredTop2Dispatcher().dispatch_preselected(
        hidden_states,
        selected_routes=decision.expert_routes,
        anchor_mlp=anchor,
        experts=experts,
    )

    assert result.selected_routes is decision.expert_routes
    assert torch.equal(result.output[2], hidden_states[2])
    assert anchor.token_count == 1
    assert tuple(expert.token_count for expert in experts) == (2, 2, 1, 1)
    for token_index, package_index in enumerate(
        package_routes.selected_indices.tolist()
    ):
        pair = result.selected_routes.selected_indices[token_index].tolist()
        if package_index == 0:
            assert pair == [0, 0]
            continue
        owners = {
            decision.expert_package_ids[candidate - 1]
            for candidate in pair
        }
        assert owners == {package_routes.expert_ids[package_index - 1]}

    expected = torch.empty_like(hidden_states)
    expected[2] = hidden_states[2]
    for token_index in (0, 1, 3):
        pair = result.selected_routes.selected_indices[token_index]
        scores = result.selected_routes.selected_scores[token_index]
        weights = torch.softmax(scores, dim=-1)
        first = experts[int(pair[0].item()) - 1].scale * hidden_states[token_index]
        second = experts[int(pair[1].item()) - 1].scale * hidden_states[token_index]
        expected[token_index] = weights[0] * first + weights[1] * second
    assert torch.allclose(result.output, expected)


def test_preselected_shape_is_rejected_before_candidate_execution() -> None:
    hidden_states = torch.ones((2, 2))
    anchor = _TrackedScale(1.0)
    experts = (_TrackedScale(2.0), _TrackedScale(3.0))
    wrong_shape = Top2RouteDecision(
        selected_indices=torch.tensor([[1, 2]]),
        selected_scores=torch.tensor([[1.0, 0.5]]),
        expert_ids=("a", "b"),
    )

    with pytest.raises(ValueError, match="score token shape"):
        AnchoredTop2Dispatcher().dispatch_preselected(
            hidden_states,
            selected_routes=wrong_shape,
            anchor_mlp=anchor,
            experts=experts,
        )

    assert anchor.token_count == 0
    assert all(expert.token_count == 0 for expert in experts)
