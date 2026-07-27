"""Integration of CAL-002 labels with CAL-001 features and router columns."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    DomainLabelBank,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
    UtilityLabelBank,
)


def test_feature_domain_and_utility_rows_share_router_expert_order() -> None:
    extractor = IdentityRouterFeatureExtractor(2)
    feature_bank = RouterFeatureBank.from_feature_batches(
        {
            1: [
                extractor(
                    torch.tensor(
                        [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]
                    )
                )
            ]
        },
        feature_specification=extractor.feature_specification,
        dataset_provenance={
            "dataset_id": "synthetic-label-integration",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=31,
    )
    domains = DomainLabelBank.from_domain_names(
        feature_bank,
        ["code", "math", "code"],
        label_source="dataset:domain",
    )

    router = AppendOnlyRouterHeads(feature_size=2)
    expert_ids = ("code:0", "math:0")
    for expert_id in expert_ids:
        router.append(
            AffineRouterHead(
                2,
                layer_index=1,
                expert_id=expert_id,
                trainable=False,
            )
        )
    utilities = UtilityLabelBank.from_losses(
        feature_bank,
        incumbent_losses_by_layer={1: torch.tensor([2.0, 2.0, 2.0])},
        expert_losses_by_layer={
            1: torch.tensor([[1.0, 2.5], [2.5, 1.0], [1.5, 2.0]])
        },
        expert_ids_by_layer={1: expert_ids},
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )

    domains.assert_aligned(feature_bank)
    utilities.assert_aligned(feature_bank)
    assert utilities.expert_ids_for_layer(1) == expert_ids
    assert utilities.utilities_for_layer(1).shape == (
        feature_bank.row_count,
        len(expert_ids),
    )
    assert router.expert_ids_for_layer(1) == (
        utilities.expert_ids_for_layer(1)
    )
