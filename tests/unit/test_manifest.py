"""Acceptance tests for versioned expert-package manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anchored_moe import ExpertPackageManifest, PackageValidationError
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _create_weight_file(package_root: Path) -> tuple[str, Path]:
    relative_path = "experts/model.safetensors"
    weight_path = package_root / relative_path
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"expert.weight": torch.arange(4).reshape(2, 2)}, weight_path)
    return relative_path, weight_path


def _manifest_values(
    relative_weight_path: str,
    weight_digest: str,
) -> dict[str, object]:
    return {
        "package_format_version": PACKAGE_FORMAT_VERSION,
        "specialization_name": "code",
        "specialization_version": "1.0.0",
        "package_structure": "single_expert",
        "base_model_repository": "Qwen/Qwen3-4B",
        "base_model_revision": "0123456789abcdef",
        "base_config_sha256": sha256_json({"model_type": "qwen3"}),
        "base_weight_sha256": {"model-00001-of-00002.safetensors": "1" * 64},
        "tokenizer_repository": "Qwen/Qwen3-4B",
        "tokenizer_revision": "0123456789abcdef",
        "base_model_dtype": "float32",
        "converted_layers": [3, 2],
        "expert_parameterization": "full_mlp",
        "expert_parameterization_config": {"initialization": "anchor_copy"},
        "expert_count": 1,
        "weight_dtype": "float32",
        "expert_weight_files": [relative_weight_path],
        "runtime_versions": {
            "accelerate": "not-installed",
            "kernels": "torch-default-cpu",
            "peft": "not-installed",
            "python": "3.12.10",
            "safetensors": "0.8.0",
            "torch": "2.13.0",
            "transformers": "5.14.1",
        },
        "training_dataset": {
            "dataset_id": "organization/code-data",
            "revision": "dataset-revision",
            "split": "train",
            "filters": ["quality"],
        },
        "package_hashes": {relative_weight_path: weight_digest},
    }


def test_manifest_round_trip_and_package_validation_are_deterministic(
    tmp_path,
) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    manifest = ExpertPackageManifest(
        **_manifest_values(relative_path, sha256_file(weight_path))
    )
    manifest_path = tmp_path / "manifest.json"

    manifest.save_json(manifest_path)
    reloaded = ExpertPackageManifest.from_json(manifest_path)
    report = reloaded.validate_package_files(tmp_path)

    assert reloaded.to_dict() == manifest.to_dict()
    assert reloaded.to_json_bytes() == manifest.to_json_bytes()
    assert reloaded.converted_layers == (2, 3)
    assert report.manifest_sha256 == manifest.manifest_sha256
    assert report.verified_files == (relative_path,)
    assert report.verified_bytes == weight_path.stat().st_size


def test_manifest_nested_metadata_is_defensively_copied(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    values = _manifest_values(relative_path, sha256_file(weight_path))
    runtime_versions = values["runtime_versions"]
    manifest = ExpertPackageManifest(**values)

    assert isinstance(runtime_versions, dict)
    runtime_versions["python"] = "mutated"  # type: ignore[index]

    assert manifest.runtime_versions["python"] == "3.12.10"


def test_payload_tampering_is_detected_before_installation(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    manifest = ExpertPackageManifest(
        **_manifest_values(relative_path, sha256_file(weight_path))
    )
    weight_path.write_bytes(weight_path.read_bytes() + b"tampered")

    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        manifest.validate_package_files(tmp_path)


def test_strict_validation_rejects_untracked_payload(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    manifest = ExpertPackageManifest(
        **_manifest_values(relative_path, sha256_file(weight_path))
    )
    (tmp_path / "untracked.txt").write_text("not declared", encoding="utf-8")

    with pytest.raises(PackageValidationError, match="untracked payload"):
        manifest.validate_package_files(tmp_path)


def test_invalid_safetensors_payload_is_rejected(tmp_path) -> None:
    relative_path = "experts/model.safetensors"
    weight_path = tmp_path / relative_path
    weight_path.parent.mkdir()
    weight_path.write_bytes(b"not a safetensors file")
    manifest = ExpertPackageManifest(
        **_manifest_values(relative_path, sha256_file(weight_path))
    )

    with pytest.raises(PackageValidationError, match="Invalid safetensors"):
        manifest.validate_package_files(tmp_path)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.safetensors",
        "/absolute/model.safetensors",
        "experts\\model.safetensors",
        "experts//model.safetensors",
        "C:/model.safetensors",
        ".",
        "experts/\nmodel.safetensors",
    ],
)
def test_manifest_rejects_unsafe_or_noncanonical_paths(unsafe_path: str) -> None:
    values = _manifest_values(unsafe_path, "a" * 64)

    with pytest.raises(PackageValidationError, match="path|POSIX"):
        ExpertPackageManifest(**values)


def test_missing_and_unknown_manifest_fields_are_rejected(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    values = _manifest_values(relative_path, sha256_file(weight_path))
    del values["base_model_revision"]

    with pytest.raises(PackageValidationError, match="Missing required"):
        ExpertPackageManifest.from_dict(values)

    values["base_model_revision"] = "pinned"
    values["unexpected"] = "value"
    with pytest.raises(PackageValidationError, match="Unknown manifest"):
        ExpertPackageManifest.from_dict(values)


def test_unsupported_format_version_is_rejected(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    values = _manifest_values(relative_path, sha256_file(weight_path))
    values["package_format_version"] = 2

    with pytest.raises(PackageValidationError, match="Unsupported"):
        ExpertPackageManifest(**values)


def test_manifest_requires_checkpoint_and_provenance_hashes(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    values = _manifest_values(relative_path, sha256_file(weight_path))
    values["base_config_sha256"] = "not-a-digest"

    with pytest.raises(PackageValidationError, match="base_config_sha256"):
        ExpertPackageManifest(**values)


@pytest.mark.parametrize(
    ("mapping_name", "missing_key"),
    [
        ("runtime_versions", "accelerate"),
        ("training_dataset", "filters"),
    ],
)
def test_manifest_requires_complete_reproducibility_metadata(
    tmp_path,
    mapping_name: str,
    missing_key: str,
) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    values = _manifest_values(relative_path, sha256_file(weight_path))
    mapping = values[mapping_name]
    assert isinstance(mapping, dict)
    del mapping[missing_key]

    with pytest.raises(PackageValidationError, match="missing required keys"):
        ExpertPackageManifest(**values)


def test_manifest_write_does_not_overwrite_by_default(tmp_path) -> None:
    relative_path, weight_path = _create_weight_file(tmp_path)
    manifest = ExpertPackageManifest(
        **_manifest_values(relative_path, sha256_file(weight_path))
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save_json(manifest_path)

    with pytest.raises(FileExistsError):
        manifest.save_json(manifest_path)
