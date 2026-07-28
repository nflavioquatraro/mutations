"""Acceptance tests for PKG-005 registry-specific router versions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    ExpertPackage,
    ExpertRegistry,
    GlobalRouterRecalibrationReport,
    LayerGlobalRouterRecalibrationReport,
    RegistryRouterVersion,
    RegistryRouterVersionError,
    compile_logical_router,
)


def _report(registry: ExpertRegistry) -> GlobalRouterRecalibrationReport:
    snapshot = registry.snapshot()
    assert snapshot.compatibility_sha256 is not None
    return GlobalRouterRecalibrationReport(
        registry_sha256=snapshot.registry_sha256,
        compatibility_sha256=snapshot.compatibility_sha256,
        feature_bank_metadata_sha256="1" * 64,
        utility_label_metadata_sha256="2" * 64,
        loss_name="token_nll",
        learning_rate=0.05,
        optimization_steps=32,
        drift_weight=1e-3,
        layer_reports=(
            LayerGlobalRouterRecalibrationReport(
                layer_index=2,
                expert_ids=("code:0",),
                row_count=4,
                anchor_target_count=2,
                initial_cross_entropy=0.7,
                final_cross_entropy=0.4,
                initial_total_loss=0.7,
                final_total_loss=0.41,
                initial_correct_count=2,
                final_correct_count=4,
                changed_route_count=2,
            ),
        ),
    )


def _router():
    heads = AppendOnlyRouterHeads(4)
    head = AffineRouterHead(
        4,
        layer_index=2,
        expert_id="code:0",
        temperature=1.5,
        threshold=0.2,
    )
    with torch.no_grad():
        head.linear.weight.copy_(
            torch.tensor([[0.5, -0.25, 0.75, 0.1]])
        )
        head.linear.bias.fill_(0.3)
    heads.append(head)
    return compile_logical_router(heads)


def _feature_specification() -> dict[str, object]:
    return {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": 4,
        "version": 1,
    }


def test_registry_router_version_round_trip_and_exact_activation(
    tmp_path: Path,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code")
    registry = ExpertRegistry()
    registry.install(package)
    artifact = RegistryRouterVersion(
        _router(),
        registry_snapshot=registry.snapshot(),
        feature_specification=_feature_specification(),
        recalibration_report=_report(registry),
    )
    path = tmp_path / "registry-router.safetensors"

    artifact.save_safetensors(path)
    loaded = RegistryRouterVersion.from_safetensors(path)
    features = torch.randn(3, 4)

    assert loaded.metadata == artifact.metadata
    assert loaded.metadata_sha256 == artifact.metadata_sha256
    assert torch.equal(
        loaded.router(features, layer_index=2),
        artifact.router(features, layer_index=2),
    )
    first = loaded.activate(registry.snapshot())
    second = loaded.activate(registry.snapshot())
    assert {
        value.data_ptr() for value in first.state_dict().values()
    }.isdisjoint(value.data_ptr() for value in second.state_dict().values())

    registry.unload(package.identity)
    with pytest.raises(RegistryRouterVersionError, match="does not match"):
        loaded.activate(registry.snapshot())


def test_registry_router_version_detects_tensor_tampering(
    tmp_path: Path,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code")
    registry = ExpertRegistry()
    registry.install(package)
    artifact = RegistryRouterVersion(
        _router(),
        registry_snapshot=registry.snapshot(),
        feature_specification=_feature_specification(),
        recalibration_report=_report(registry),
    )
    source = tmp_path / "source.safetensors"
    tampered = tmp_path / "tampered.safetensors"
    artifact.save_safetensors(source)

    tensors = load_file(source)
    tensors["heads.0.weight"][0, 0] += 1.0
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata is not None
    save_file(tensors, tampered, metadata=metadata)

    with pytest.raises(RegistryRouterVersionError, match="digest mismatch"):
        RegistryRouterVersion.from_safetensors(tampered)


def test_registry_router_version_rejects_incompatible_construction(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code")
    registry = ExpertRegistry()
    registry.install(package)

    with pytest.raises(ValueError, match="output_size"):
        RegistryRouterVersion(
            _router(),
            registry_snapshot=registry.snapshot(),
            feature_specification={
                **_feature_specification(),
                "output_size": 8,
            },
            recalibration_report=_report(registry),
        )

    registry.unload(package.identity)
    with pytest.raises(ValueError, match="active package"):
        RegistryRouterVersion(
            _router(),
            registry_snapshot=registry.snapshot(),
            feature_specification=_feature_specification(),
            recalibration_report=_report_for_snapshot(registry),
        )


def _report_for_snapshot(
    registry: ExpertRegistry,
) -> GlobalRouterRecalibrationReport:
    snapshot = registry.snapshot()
    assert snapshot.compatibility_sha256 is not None
    return GlobalRouterRecalibrationReport(
        registry_sha256=snapshot.registry_sha256,
        compatibility_sha256=snapshot.compatibility_sha256,
        feature_bank_metadata_sha256="1" * 64,
        utility_label_metadata_sha256="2" * 64,
        loss_name="token_nll",
        learning_rate=0.05,
        optimization_steps=1,
        drift_weight=0.0,
        layer_reports=(
            LayerGlobalRouterRecalibrationReport(
                layer_index=2,
                expert_ids=("code:0",),
                row_count=1,
                anchor_target_count=1,
                initial_cross_entropy=0.7,
                final_cross_entropy=0.7,
                initial_total_loss=0.7,
                final_total_loss=0.7,
                initial_correct_count=1,
                final_correct_count=1,
                changed_route_count=0,
            ),
        ),
    )
