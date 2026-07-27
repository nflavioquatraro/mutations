"""Acceptance tests for explicit expert-payload ownership metadata."""

from __future__ import annotations

import pytest

from anchored_moe import (
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    ExpertPackageManifest,
    PackageValidationError,
    parse_expert_payload_bindings,
)
from anchored_moe.registry import PACKAGE_FORMAT_VERSION


def _manifest(
    *,
    parameterization_config: dict[str, object],
    weight_files: list[str] | None = None,
) -> ExpertPackageManifest:
    files = weight_files or [
        "experts/layer-2-code.safetensors",
        "experts/layer-3-code.safetensors",
    ]
    return ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="code",
        specialization_version="1.0.0",
        package_structure="single_expert",
        base_model_repository="local-qwen3",
        base_model_revision="seed-13",
        base_config_sha256="0" * 64,
        base_weight_sha256={"model.safetensors": "1" * 64},
        tokenizer_repository="local-qwen3",
        tokenizer_revision="seed-13",
        base_model_dtype="float32",
        converted_layers=[2, 3],
        expert_parameterization="full_mlp",
        expert_parameterization_config=parameterization_config,
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=files,
        runtime_versions={
            "accelerate": "not-installed",
            "kernels": "torch-default-cpu",
            "peft": "not-installed",
            "python": "3.12",
            "safetensors": "0.8",
            "torch": "2.13",
            "transformers": "5.14",
        },
        training_dataset={
            "dataset_id": "local/code",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        package_hashes={path: "2" * 64 for path in files},
    )


def _binding_config() -> dict[str, object]:
    return {
        "initialization": "anchor_copy",
        "payload_binding_format_version": (
            EXPERT_PAYLOAD_BINDING_FORMAT_VERSION
        ),
        "payload_bindings": [
            {
                "layer_index": 3,
                "expert_id": "code:0",
                "weight_file": "experts/layer-3-code.safetensors",
            },
            {
                "layer_index": 2,
                "expert_id": "code:0",
                "weight_file": "experts/layer-2-code.safetensors",
            },
        ],
    }


def test_payload_bindings_are_explicit_complete_and_runtime_sorted() -> None:
    manifest = _manifest(parameterization_config=_binding_config())

    bindings = parse_expert_payload_bindings(manifest, require=True)

    assert [binding.key for binding in bindings] == [
        (2, "code:0"),
        (3, "code:0"),
    ]
    assert bindings[0].to_dict() == {
        "layer_index": 2,
        "expert_id": "code:0",
        "weight_file": "experts/layer-2-code.safetensors",
    }


def test_legacy_v1_manifest_remains_valid_but_is_not_attachable() -> None:
    manifest = _manifest(
        parameterization_config={"initialization": "anchor_copy"}
    )

    assert parse_expert_payload_bindings(manifest) == ()
    with pytest.raises(PackageValidationError, match="requires explicit"):
        parse_expert_payload_bindings(manifest, require=True)


def test_binding_block_requires_exact_file_and_layer_coverage() -> None:
    config = _binding_config()
    config["payload_bindings"] = list(config["payload_bindings"])[:1]

    with pytest.raises(PackageValidationError, match="exactly cover"):
        _manifest(parameterization_config=config)


def test_binding_block_rejects_router_style_ownership_ambiguity() -> None:
    config = _binding_config()
    bindings = list(config["payload_bindings"])
    bindings[1] = dict(bindings[0])
    config["payload_bindings"] = bindings

    with pytest.raises(PackageValidationError, match="repeats layer/expert"):
        _manifest(parameterization_config=config)


def test_binding_block_rejects_unknown_fields() -> None:
    config = _binding_config()
    bindings = [dict(value) for value in config["payload_bindings"]]
    bindings[0]["inferred_name"] = True
    config["payload_bindings"] = bindings

    with pytest.raises(PackageValidationError, match="unknown"):
        _manifest(parameterization_config=config)
