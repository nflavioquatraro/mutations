"""Verify CAL-002 domain and individual-utility label datasets."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from anchored_moe import (
    DomainLabelBank,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
    UtilityLabelBank,
)


def main() -> None:
    """Construct aligned labels, serialize them, and verify exact reload."""

    extractor = IdentityRouterFeatureExtractor(feature_size=2)
    features = extractor(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    )
    feature_bank = RouterFeatureBank.from_feature_batches(
        {2: [features]},
        feature_specification=extractor.feature_specification,
        dataset_provenance={
            "dataset_id": "synthetic-local-labels",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=43,
    )
    domains = DomainLabelBank.from_domain_names(
        feature_bank,
        ["code", "math", "code"],
        label_source="dataset:domain",
    )
    utilities = UtilityLabelBank.from_losses(
        feature_bank,
        incumbent_losses_by_layer={2: torch.tensor([2.0, 2.0, 2.0])},
        expert_losses_by_layer={
            2: torch.tensor([[1.0, 2.5], [2.5, 1.0], [1.5, 2.0]])
        },
        expert_ids_by_layer={2: ["code:0", "math:0"]},
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-router-labels-"
    ) as directory:
        root = Path(directory)
        domain_path = root / "domains.safetensors"
        utility_path = root / "utilities.safetensors"
        domains.save_safetensors(domain_path)
        utilities.save_safetensors(utility_path)
        restored_domains = DomainLabelBank.from_safetensors(domain_path)
        restored_utilities = UtilityLabelBank.from_safetensors(utility_path)

    restored_domains.assert_aligned(feature_bank)
    restored_utilities.assert_aligned(feature_bank)
    assert restored_domains.metadata_sha256 == domains.metadata_sha256
    assert restored_utilities.metadata_sha256 == utilities.metadata_sha256

    print(
        "Router label-bank validation passed: "
        f"classes={restored_domains.class_names}, "
        f"layers={restored_utilities.layer_indices}, "
        f"experts={restored_utilities.expert_ids_for_layer(2)}, "
        f"rows={restored_utilities.row_count}, "
        "exact_reload=True"
    )


if __name__ == "__main__":
    main()
