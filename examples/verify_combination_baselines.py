"""Compare COM-002 baselines through standalone sparse top-2 dispatch."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AnchoredTop2Dispatcher,
    EqualWeightCombination,
    NormalizedPositiveEvidenceCombination,
)


def _scaled_identity(scale: float) -> nn.Linear:
    """Return a deterministic two-dimensional linear expert."""

    module = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2) * scale)
    return module


def main() -> None:
    """Verify both COM-002 policies under identical selection and execution."""

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expert_scores = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    expert_ids = ("code:0", "math:0")
    anchor = _scaled_identity(1.0)
    experts = (_scaled_identity(2.0), _scaled_identity(-1.0))

    equal = AnchoredTop2Dispatcher(
        combination_policy=EqualWeightCombination(),
    ).dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=experts,
    )
    evidence = AnchoredTop2Dispatcher(
        combination_policy=NormalizedPositiveEvidenceCombination(),
    ).dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=experts,
    )

    assert torch.equal(
        equal.mixture_weights,
        torch.full((2, 2), 0.5),
    )
    assert torch.allclose(
        evidence.mixture_weights,
        torch.tensor([[0.75, 0.25], [0.75, 0.25]]),
    )
    assert torch.allclose(equal.output, hidden_states * 0.5)
    assert torch.allclose(
        evidence.output,
        torch.stack((hidden_states[0] * 1.25, hidden_states[1] * -0.25)),
    )
    print(
        "COM-002 validation passed: "
        f"equal_weights={equal.mixture_weights.tolist()}, "
        f"evidence_weights={evidence.mixture_weights.tolist()}"
    )


if __name__ == "__main__":
    main()
