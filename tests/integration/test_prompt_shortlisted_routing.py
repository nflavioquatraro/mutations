"""RTR-006 integration with existing anchored top-1 sparse dispatch."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Dispatcher,
    AppendOnlyRouterHeads,
    PromptShortlistedLogicalRouter,
    PromptTopMShortlister,
    compile_logical_router,
)


class _CountingScale(nn.Module):
    """Multiply routed rows by a fixed scale and count sparse invocations."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return scaled rows while recording one module invocation."""

        self.calls += 1
        return hidden_states * self.scale


def test_prompt_shortlist_flows_through_top1_selection_and_sparse_dispatch() -> None:
    heads = AppendOnlyRouterHeads(feature_size=2)
    for expert_id, weight in (
        ("code:0", (1.0, 0.0)),
        ("math:0", (0.0, 1.0)),
        ("science:0", (-1.0, -1.0)),
    ):
        head = AffineRouterHead(2, layer_index=0, expert_id=expert_id)
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
            head.linear.bias.zero_()
        heads.append(head)

    shortlist = PromptTopMShortlister(1)(
        torch.tensor(
            [
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        candidate_ids=("code:0", "math:0", "science:0"),
    )
    hidden_states = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    scores = PromptShortlistedLogicalRouter(compile_logical_router(heads))(
        hidden_states,
        shortlist=shortlist,
        layer_index=0,
    )
    anchor = _CountingScale(1.0)
    experts = (
        _CountingScale(2.0),
        _CountingScale(3.0),
        _CountingScale(4.0),
    )

    result = AnchoredTop1Dispatcher().dispatch(
        hidden_states,
        expert_scores=scores.scores,
        expert_ids=scores.expert_ids,
        anchor_mlp=anchor,
        experts=experts,
    )

    expected = torch.stack(
        (
            hidden_states[0] * 2.0,
            hidden_states[1] * 3.0,
            hidden_states[2],
        )
    )
    assert torch.equal(result.output, expected)
    assert result.selected_routes.selected_indices.tolist() == [
        [1, 1],
        [2, 2],
        [0, 0],
    ]
    assert anchor.calls == 1
    assert tuple(expert.calls for expert in experts) == (1, 1, 0)
