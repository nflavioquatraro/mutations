"""Integration of CAL-001 feature banks with RTR-001 and RTR-002."""

from __future__ import annotations

from pathlib import Path

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
)


def test_cached_identity_features_preserve_affine_router_scores(
    tmp_path: Path,
) -> None:
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]
    )
    extractor = IdentityRouterFeatureExtractor(2)
    features = extractor(hidden_states)
    bank = RouterFeatureBank.from_feature_batches(
        {1: [features]},
        feature_specification=extractor.feature_specification,
        dataset_provenance={
            "dataset_id": "synthetic-router-integration",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=29,
    )

    payload_path = tmp_path / "features.safetensors"
    bank.save_safetensors(payload_path)
    restored = RouterFeatureBank.from_safetensors(payload_path)

    router = AppendOnlyRouterHeads(feature_size=2)
    head = AffineRouterHead(2, layer_index=1, expert_id="code")
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[1.0, -0.5]]))
    router.append(head)

    expected_scores = router(features.reshape(-1, 2), layer_index=1)
    restored_scores = router(
        restored.features_for_layer(1),
        layer_index=1,
    )

    assert torch.equal(restored_scores, expected_scores)
