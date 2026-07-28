"""Verify RTR-004 recalibration and PKG-005 exact registry restoration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from anchored_moe import (
    REGISTRY_ROUTER_VERSION_FORMAT_VERSION,
    AffineRouterHead,
    AppendOnlyRouterHeads,
    PackageIdentity,
    RegistryRouterVersion,
    RegistrySnapshot,
    RouterFeatureBank,
    UtilityLabelBank,
    compile_logical_router,
    recalibrate_global_router,
)
from anchored_moe.utils import sha256_json


def main() -> None:
    """Fit, persist, reload, and explicitly activate one router version."""

    feature_specification = {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": 2,
        "version": 1,
    }
    features = RouterFeatureBank(
        {
            2: torch.tensor(
                [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            )
        },
        feature_specification=feature_specification,
        dataset_provenance={
            "dataset_id": "examples/offline-router-replay",
            "revision": "v1",
            "split": "validation",
            "filters": ["synthetic"],
        },
        seed=19,
    )
    utilities = UtilityLabelBank(
        {2: torch.tensor([[-1.0], [-1.0], [-1.0], [1.0], [1.0]])},
        expert_ids_by_layer={2: ("code:0",)},
        row_ids=features.row_ids,
        feature_bank_metadata_sha256=features.metadata_sha256,
        loss_name="token_nll",
        incumbent_route="anchor",
    )
    heads = AppendOnlyRouterHeads(2)
    heads.append(AffineRouterHead(2, layer_index=2, expert_id="code:0"))
    source = compile_logical_router(heads)

    package = PackageIdentity("code", "1.0.0", "1" * 64)
    registry_hash = sha256_json(
        {
            "active_packages": [package.to_dict()],
            "registry_format_version": 1,
        }
    )
    snapshot = RegistrySnapshot(
        registry_format_version=1,
        registry_sha256=registry_hash,
        compatibility_sha256="2" * 64,
        active_packages=(package,),
    )
    recalibrated, report = recalibrate_global_router(
        source,
        features,
        utilities,
        registry_sha256=snapshot.registry_sha256,
        compatibility_sha256=snapshot.compatibility_sha256,
        learning_rate=0.1,
        optimization_steps=64,
    )
    version = RegistryRouterVersion(
        recalibrated,
        registry_snapshot=snapshot,
        feature_specification=feature_specification,
        recalibration_report=report,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "router-version.safetensors"
        version.save_safetensors(path)
        loaded = RegistryRouterVersion.from_safetensors(path)
        restored = loaded.activate(snapshot)

    replay = features.features_for_layer(2)
    assert torch.equal(
        restored(replay, layer_index=2),
        recalibrated(replay, layer_index=2),
    )
    assert report.layer_reports[0].final_cross_entropy < (
        report.layer_reports[0].initial_cross_entropy
    )
    print(
        "RTR-004/PKG-005 validation passed: "
        f"format={REGISTRY_ROUTER_VERSION_FORMAT_VERSION}, "
        f"registry_sha256={snapshot.registry_sha256}, "
        f"metadata_sha256={version.metadata_sha256}, "
        "hidden_recalibration=False"
    )


if __name__ == "__main__":
    main()
