"""Verify registry-style compatibility metadata and constrained top-2 routing."""

from __future__ import annotations

import torch

from anchored_moe import (
    CompatibilityConstrainedTop2Selection,
    MixtureCompatibilityIndex,
)


def main() -> None:
    """Select the strongest compatible pair with deterministic anchor fallback."""

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

    decision = CompatibilityConstrainedTop2Selection(compatibility)(
        scores,
        expert_ids=compatibility.expert_ids,
    )

    # The first token mixes the two math experts even though incompatible and
    # non-mixable experts have higher individual scores. The second token has
    # no compatible positive pair and therefore includes the anchor.
    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[2, 3], [4, 0]]),
    )
    assert torch.equal(
        decision.expert_pair_mask,
        torch.tensor([True, False]),
    )
    print(
        "Compatibility-constrained top-2 validation passed: "
        f"pairs={decision.selected_indices.tolist()}, "
        f"anchor_fallback={decision.anchor_mask.tolist()}"
    )


if __name__ == "__main__":
    main()
