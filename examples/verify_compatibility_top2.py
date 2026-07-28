"""Verify registry-style compatibility metadata and constrained top-2 routing."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AnchoredTop2Dispatcher,
    CompatibilityConstrainedTop2Selection,
    MixtureCompatibilityIndex,
)


class _ScaleMLP(nn.Module):
    """Small deterministic MLP stand-in for sparse-dispatch verification."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states * self.scale


def main() -> None:
    """Execute the strongest compatible pair with exact anchor fallback."""

    compatibility = MixtureCompatibilityIndex(
        expert_ids=("code:0", "math:0", "math:1", "general:0"),
        group_ids=("code-v1", "math-v1", "math-v1", None),
    )
    scores = torch.tensor(
        [
            [0.95, 0.80, 0.70, 1.10],
            [0.60, 0.20, -0.10, 0.90],
        ]
    )

    anchor = _ScaleMLP(1.0)
    code = _ScaleMLP(8.0)
    math_zero = _ScaleMLP(2.0)
    math_one = _ScaleMLP(-1.0)
    general = _ScaleMLP(4.0)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = AnchoredTop2Dispatcher(
        selection_policy=CompatibilityConstrainedTop2Selection(compatibility),
    ).dispatch(
        hidden,
        expert_scores=scores,
        expert_ids=compatibility.expert_ids,
        anchor_mlp=anchor,
        experts=(code, math_zero, math_one, general),
    )

    # The first token mixes the two math experts even though incompatible and
    # non-mixable experts have higher individual scores. The second token has
    # no compatible positive pair and therefore includes the anchor.
    assert torch.equal(
        result.selected_routes.selected_indices,
        torch.tensor([[2, 3], [4, 0]]),
    )
    assert torch.equal(
        result.selected_routes.expert_pair_mask,
        torch.tensor([True, False]),
    )
    weights = torch.softmax(torch.tensor([0.80, 0.70]), dim=0)
    expected_pair = weights[0] * hidden[0] * 2.0 + weights[1] * hidden[0] * -1.0
    assert torch.allclose(result.output[0], expected_pair)
    assert torch.equal(result.output[1], hidden[1])
    assert anchor.calls == 1
    assert code.calls == general.calls == 0
    assert math_zero.calls == math_one.calls == 1
    print(
        "Compatibility-aware top-2 dispatch validation passed: "
        f"pairs={result.selected_routes.selected_indices.tolist()}, "
        f"anchor_fallback={result.selected_routes.anchor_mask.tolist()}"
    )


if __name__ == "__main__":
    main()
