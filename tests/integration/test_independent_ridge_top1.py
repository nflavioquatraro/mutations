"""End-to-end CAL-004 fitting through anchored top-1 selection."""

from __future__ import annotations

from pathlib import Path

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
    UtilityLabelBank,
    fit_independent_ridge,
)


def test_fitted_utility_heads_feed_anchored_top1_in_label_order(
    tmp_path: Path,
) -> None:
    extractor = IdentityRouterFeatureExtractor(feature_size=1)
    features = extractor(torch.tensor([[[-2.0], [-1.0], [0.0], [1.0], [2.0]]]))
    feature_bank = RouterFeatureBank.from_feature_batches(
        {2: [features]},
        feature_specification=extractor.feature_specification,
        dataset_provenance={
            "dataset_id": "synthetic-ridge-top1",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=59,
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

    assert router.expert_ids_for_layer(2) == ("math:0", "code:0")
    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[1, 1, 0, 2, 2]]),
    )
    assert decision.expert_ids == ("math:0", "code:0")

    payload_path = tmp_path / "fitted-router.safetensors"
    router.save_safetensors(payload_path)
    restored = AppendOnlyRouterHeads.from_safetensors(payload_path)
    assert restored.head_keys == router.head_keys
    assert torch.equal(restored(features, layer_index=2), scores)
