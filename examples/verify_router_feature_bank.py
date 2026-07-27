"""Verify deterministic CAL-001 router feature caching and serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from anchored_moe import IdentityRouterFeatureExtractor, RouterFeatureBank


def main() -> None:
    """Cache ordered synthetic features and verify an exact safe reload."""

    extractor = IdentityRouterFeatureExtractor(feature_size=3)
    layer_two_batches = [
        extractor(torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]])),
        extractor(torch.tensor([[[0.5, 0.5, 1.0]]])),
    ]
    layer_three_batches = [
        extractor(torch.tensor([[[1.5, 0.0, 0.5], [0.0, 1.5, -0.5]]])),
        extractor(torch.tensor([[[0.5, 0.5, 1.5]]])),
    ]
    provenance = {
        "dataset_id": "synthetic-local-reference",
        "revision": "1",
        "split": "train",
        "filters": [],
    }
    bank = RouterFeatureBank.from_feature_batches(
        {2: layer_two_batches, 3: layer_three_batches},
        feature_specification=dict(extractor.feature_specification),
        dataset_provenance=provenance,
        seed=41,
    )

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-router-feature-bank-"
    ) as directory:
        payload_path = Path(directory) / "reference-bank.safetensors"
        bank.save_safetensors(payload_path)
        restored = RouterFeatureBank.from_safetensors(payload_path)

    assert restored.metadata_sha256 == bank.metadata_sha256
    assert torch.equal(
        restored.features_for_layer(2),
        bank.features_for_layer(2),
    )
    restored.assert_compatible(extractor.feature_specification)

    print(
        "Router feature-bank validation passed: "
        f"layers={restored.layer_indices}, "
        f"rows={restored.row_count}, "
        f"feature_size={restored.feature_size}, "
        f"metadata_sha256={restored.metadata_sha256}, "
        "exact_reload=True"
    )


if __name__ == "__main__":
    main()
