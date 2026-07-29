"""Verify BAL-004 on deterministic sparse anchored top-1 assignments."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    LossFreeBiasBalancer,
    LossFreeBiasConfig,
)


def main() -> None:
    """Rebalance historical loads without changing raw scores or gradients."""

    expert_ids = ("code:0", "code:1")
    first = torch.tensor(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.1]
    )
    raw_scores = torch.stack((first, torch.full_like(first, 0.1)), dim=-1)
    raw_snapshot = raw_scores.clone()
    selector = AnchoredTop1Selection()
    balancer = LossFreeBiasBalancer(
        expert_ids,
        config=LossFreeBiasConfig(update_rate=0.05),
    )
    reports = []

    for _ in range(5):
        score_views = balancer.apply_for_selection(raw_scores)
        routes = selector(
            score_views.selection_scores,
            expert_ids=score_views.expert_ids,
        )
        reports.append(balancer.update(routes.selected_indices))

    assert reports[0].assignment_loads == (9.0, 1.0)
    assert reports[-1].assignment_loads == (5.0, 5.0)
    assert reports[-1].maximum_load_violation == 0.0
    assert torch.equal(raw_scores, raw_snapshot)
    assert tuple(balancer.parameters()) == ()

    with TemporaryDirectory() as directory:
        checkpoint = f"{directory}/loss-free-bias.safetensors"
        balancer.save_safetensors(checkpoint)
        restored = LossFreeBiasBalancer.from_safetensors(checkpoint)
        assert torch.equal(restored.biases, balancer.biases)
        assert restored.update_step == balancer.update_step

    print(
        "BAL-004 validation passed: "
        f"initial_loads={reports[0].assignment_loads}, "
        f"final_loads={reports[-1].assignment_loads}, "
        f"final_max_violation={reports[-1].maximum_load_violation:.3f}, "
        f"biases={reports[-1].updated_biases}, "
        f"updates={balancer.update_step}, "
        "anchor_bias=absent, trainable_parameters=0"
    )


if __name__ == "__main__":
    main()
