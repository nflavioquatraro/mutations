"""End-to-end offline RTR-004 and PKG-005 lifecycle acceptance test."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    ExpertPackage,
    ExpertRegistry,
    RegistryRouterVersion,
    RegistryRouterVersionError,
    RouterFeatureBank,
    UtilityLabelBank,
    compile_logical_router,
    recalibrate_global_router,
)


def test_recalibrate_persist_remove_and_restore_exact_registry_router(
    tmp_path: Path,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code", router_output_size=2)
    registry = ExpertRegistry()
    registry.install(package)
    installed = registry.snapshot()
    assert installed.compatibility_sha256 is not None

    feature_specification = package.manifest.router_feature_specification
    features = RouterFeatureBank(
        {
            2: torch.tensor(
                [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            )
        },
        feature_specification=feature_specification,
        dataset_provenance={
            "dataset_id": "tests/registry-replay",
            "revision": "v1",
            "split": "validation",
            "filters": ["offline"],
        },
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

    recalibrated, report = recalibrate_global_router(
        source,
        features,
        utilities,
        registry_sha256=installed.registry_sha256,
        compatibility_sha256=installed.compatibility_sha256,
        learning_rate=0.1,
        optimization_steps=64,
    )
    artifact = RegistryRouterVersion(
        recalibrated,
        registry_snapshot=installed,
        feature_specification=feature_specification,
        recalibration_report=report,
    )
    path = tmp_path / "code-math-router.safetensors"
    artifact.save_safetensors(path)
    loaded = RegistryRouterVersion.from_safetensors(path)
    replay = features.features_for_layer(2)
    expected_scores = recalibrated(replay, layer_index=2)

    assert torch.equal(
        loaded.activate(installed)(replay, layer_index=2),
        expected_scores,
    )
    registry.unload(package.identity)
    with pytest.raises(RegistryRouterVersionError, match="does not match"):
        loaded.activate(registry.snapshot())

    registry.install(package)
    assert registry.snapshot() == installed
    assert torch.equal(
        loaded.activate(registry.snapshot())(replay, layer_index=2),
        expected_scores,
    )
