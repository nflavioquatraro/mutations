"""Verify deterministic BAL-005 training/prefill assignment planning."""

from __future__ import annotations

import torch

from anchored_moe import (
    ExpertChoiceConfig,
    ExpertChoiceRouting,
)


def main() -> None:
    """Prove fixed expert load, variable token arity, and anchor fallback."""

    logits = torch.tensor(
        [
            [4.0, 4.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 3.0],
            [9.0, 9.0, 9.0],
        ],
        requires_grad=True,
    )
    token_mask = torch.tensor([True, True, True, True, True, False])
    decision = ExpertChoiceRouting(
        ExpertChoiceConfig(expert_capacity=1)
    )(
        logits,
        expert_ids=("code:0", "code:1", "code:2"),
        token_mask=token_mask,
    )

    assert torch.equal(
        decision.selected_token_indices,
        torch.tensor([[0], [0], [3]]),
    )
    assert torch.equal(
        decision.token_assignment_counts,
        torch.tensor([2, 0, 0, 1, 0, 0]),
    )
    assert torch.equal(
        decision.anchor_fallback_mask,
        torch.tensor([False, True, True, False, True, False]),
    )
    decision.selected_gates.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0

    print(
        "BAL-005 validation passed: "
        f"expert_capacity={decision.expert_capacity}, "
        f"selected_indices={decision.selected_token_indices.tolist()}, "
        f"token_arities={decision.token_assignment_counts.tolist()}, "
        f"anchor_fallback={decision.anchor_fallback_mask.tolist()}, "
        "padding_selected=false, autoregressive_decode=unsupported"
    )


if __name__ == "__main__":
    main()
