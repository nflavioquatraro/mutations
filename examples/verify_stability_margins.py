"""Verify CAL-006 sequential-installation margin refinement offline."""

from __future__ import annotations

import json

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    RouterFeatureBank,
    UtilityLabelBank,
    refine_stability_margins,
)


def main() -> None:
    """Protect incumbent rows while making a useful candidate win."""

    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    bank = RouterFeatureBank(
        {2: features},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 1,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "synthetic-local-stability-refinement",
            "revision": "1",
            "split": "validation",
            "filters": [],
        },
        seed=89,
        row_ids=torch.arange(10, 14),
    )
    utilities = UtilityLabelBank(
        {2: torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])},
        expert_ids_by_layer={2: ("code:new",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="registry:old",
    )

    incumbent = AppendOnlyRouterHeads(1)
    incumbent_head = AffineRouterHead(
        1,
        layer_index=2,
        expert_id="code:old",
        trainable=False,
    )
    with torch.no_grad():
        incumbent_head.linear.bias.fill_(0.2)
    incumbent.append(incumbent_head)

    candidate = AppendOnlyRouterHeads(1)
    candidate.append(
        AffineRouterHead(
            1,
            layer_index=2,
            expert_id="code:new",
            trainable=False,
        )
    )
    incumbent_before = {
        name: tensor.detach().clone()
        for name, tensor in incumbent.state_dict().items()
    }
    candidate_before = {
        name: tensor.detach().clone()
        for name, tensor in candidate.state_dict().items()
    }

    refined, report = refine_stability_margins(
        candidate,
        incumbent,
        bank,
        utilities,
        stability_row_ids=[11, 10],
        win_row_ids=[13, 12],
        stability_margin=0.3,
        win_margin=0.3,
        optimization_steps=128,
    )
    old_scores = incumbent(features, layer_index=2)
    new_scores = refined(features, layer_index=2)
    decision = AnchoredTop1Selection()(
        torch.cat((old_scores, new_scores), dim=-1),
        expert_ids=("code:old", "code:new"),
    )

    assert torch.equal(
        decision.selected_indices,
        torch.tensor([1, 1, 2, 2]),
    )
    assert all(
        torch.equal(incumbent.state_dict()[name], tensor)
        for name, tensor in incumbent_before.items()
    )
    assert all(
        torch.equal(candidate.state_dict()[name], tensor)
        for name, tensor in candidate_before.items()
    )
    assert all(not parameter.requires_grad for parameter in refined.parameters())
    json.dumps(report.to_dict(), allow_nan=False)

    head_report = report.head_reports[0]
    print(
        "Stability-margin validation passed: "
        f"status={head_report.status}, "
        "stability_violations="
        f"{head_report.stability_violation_count_before}->"
        f"{head_report.stability_violation_count_after}, "
        "win_violations="
        f"{head_report.win_violation_count_before}->"
        f"{head_report.win_violation_count_after}, "
        f"routes={decision.selected_indices.tolist()}, "
        "immutable_inputs=True"
    )


if __name__ == "__main__":
    main()
