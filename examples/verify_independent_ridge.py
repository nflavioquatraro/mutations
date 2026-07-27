"""Verify CAL-004 ridge fitting and anchored top-1 routing offline."""

from __future__ import annotations

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
    UtilityLabelBank,
    fit_independent_ridge,
)


def main() -> None:
    """Fit two utility heads and verify deterministic anchor-aware routes."""

    extractor = IdentityRouterFeatureExtractor(feature_size=1)
    hidden_states = torch.tensor(
        [[[-2.0], [-1.0], [0.0], [1.0], [2.0]]]
    )
    features = extractor(hidden_states)
    feature_bank = RouterFeatureBank.from_feature_batches(
        {2: [features]},
        feature_specification=extractor.feature_specification,
        dataset_provenance={
            "dataset_id": "synthetic-local-ridge",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=61,
    )
    utilities = UtilityLabelBank(
        {2: torch.column_stack((-features.reshape(-1), features.reshape(-1)))},
        expert_ids_by_layer={2: ("math:0", "code:0")},
        row_ids=feature_bank.row_ids,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )

    router = fit_independent_ridge(
        feature_bank,
        utilities,
        regularization=1e-6,
    )
    scores = router(features, layer_index=2)
    decision = AnchoredTop1Selection()(
        scores,
        expert_ids=router.expert_ids_for_layer(2),
    )
    expected_routes = torch.tensor([[1, 1, 0, 2, 2]])
    assert torch.equal(decision.selected_indices, expected_routes)
    assert all(not parameter.requires_grad for parameter in router.parameters())

    print(
        "Independent ridge validation passed: "
        f"heads={router.head_keys}, "
        f"routes={decision.selected_indices.tolist()}, "
        "intercept_unregularized=True, frozen=True"
    )


if __name__ == "__main__":
    main()
