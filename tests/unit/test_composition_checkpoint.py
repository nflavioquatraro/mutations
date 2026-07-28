"""Acceptance tests for the CORE-006 local composition format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import GenerationConfig

from anchored_moe import (
    COMPOSITION_FORMAT_VERSION,
    COMPOSITION_MANIFEST_FILENAME,
    AnchoredMoEConfig,
    CompositionCheckpointError,
    ExpertPackage,
    ExpertRegistry,
)
from anchored_moe.registry.composition import (
    export_composition_checkpoint,
    load_composition_checkpoint,
)
from anchored_moe.utils import sha256_json


def _configuration() -> AnchoredMoEConfig:
    """Return a pinned configuration compatible with package fixtures."""

    return AnchoredMoEConfig(
        base_model_name_or_path="Qwen/Qwen3-4B",
        base_model_revision="base-revision",
        base_config_sha256=sha256_json({"model_type": "qwen3"}),
        converted_layers=[2, 3],
    )


def test_export_and_load_preserve_ordered_active_registry(
    expert_package_factory,
    tmp_path: Path,
) -> None:
    """The artifact must reconstruct active order and copied identities."""

    code = expert_package_factory("code")
    math = expert_package_factory("math")
    registry = ExpertRegistry()
    registry.install(code)
    registry.install(math)
    destination = tmp_path / "composition"

    exported = export_composition_checkpoint(
        destination,
        config=_configuration(),
        generation_config=GenerationConfig(max_new_tokens=3),
        registry=registry,
    )
    loaded = load_composition_checkpoint(destination)

    assert exported.composition_format_version == COMPOSITION_FORMAT_VERSION
    assert loaded.manifest == exported
    assert tuple(package.identity for package in loaded.packages) == (
        code.identity,
        math.identity,
    )
    assert loaded.manifest.registry_sha256 == registry.active_registry_sha256
    assert loaded.generation_config is not None
    assert loaded.generation_config.max_new_tokens == 3
    assert all(
        package.package_directory.is_relative_to(destination.resolve())
        for package in loaded.packages
    )


def test_load_rejects_config_tampering_and_untracked_files(
    expert_package_factory,
    tmp_path: Path,
) -> None:
    """Every non-package file must be tracked and hash-validated."""

    registry = ExpertRegistry()
    registry.install(expert_package_factory("code"))
    tampered = tmp_path / "tampered"
    export_composition_checkpoint(
        tampered,
        config=_configuration(),
        generation_config=None,
        registry=registry,
    )
    config_path = tampered / "config.json"
    config_path.write_bytes(config_path.read_bytes() + b"\n")

    with pytest.raises(CompositionCheckpointError, match="config.json"):
        load_composition_checkpoint(tampered)

    untracked = tmp_path / "untracked"
    export_composition_checkpoint(
        untracked,
        config=_configuration(),
        generation_config=None,
        registry=registry,
    )
    (untracked / "notes.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(CompositionCheckpointError, match="untracked"):
        load_composition_checkpoint(untracked)


def test_load_rejects_reordered_package_entries(
    expert_package_factory,
    tmp_path: Path,
) -> None:
    """Package order is authenticated by the active-registry digest."""

    registry = ExpertRegistry()
    registry.install(expert_package_factory("code"))
    registry.install(expert_package_factory("math"))
    destination = tmp_path / "composition"
    export_composition_checkpoint(
        destination,
        config=_configuration(),
        generation_config=None,
        registry=registry,
    )
    manifest_path = destination / COMPOSITION_MANIFEST_FILENAME
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["packages"].reverse()
    manifest_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        CompositionCheckpointError,
        match="registry_sha256",
    ):
        load_composition_checkpoint(destination)


def test_export_rejects_package_from_another_base_configuration(
    expert_package_factory,
    tmp_path: Path,
) -> None:
    """A synchronized artifact cannot cross anchor configuration identities."""

    registry = ExpertRegistry()
    registry.install(expert_package_factory("code"))
    incompatible = AnchoredMoEConfig(
        base_model_name_or_path="Qwen/Qwen3-4B",
        base_model_revision="base-revision",
        base_config_sha256="0" * 64,
        converted_layers=[2, 3],
    )

    with pytest.raises(
        CompositionCheckpointError,
        match="base_config_sha256",
    ):
        export_composition_checkpoint(
            tmp_path / "composition",
            config=incompatible,
            generation_config=None,
            registry=registry,
        )


def test_registry_snapshot_round_trip_preserves_empty_compatibility_binding(
    expert_package_factory,
) -> None:
    """An empty post-unload registry keeps its permanent compatibility pin."""

    package: ExpertPackage = expert_package_factory("code")
    registry = ExpertRegistry()
    registry.install(package)
    registry.unload(package.identity)
    snapshot = registry.snapshot()

    restored = ExpertRegistry.from_snapshot(
        (),
        registry_format_version=snapshot.registry_format_version,
        registry_sha256=snapshot.registry_sha256,
        compatibility_sha256=snapshot.compatibility_sha256,
    )

    assert restored.snapshot() == snapshot
