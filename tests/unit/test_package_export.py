"""Acceptance tests for the NB-003 expert-only package exporter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anchored_moe import (
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ExpertPackageExportError,
    FullMLPExpert,
    export_full_mlp_expert_package,
)
from anchored_moe.evaluation import fingerprint_module
from anchored_moe.utils import sha256_json


def _config(
    model: torch.nn.Module,
    *,
    base_config_sha256: str | None = None,
) -> ExpertPackageExportConfig:
    """Return complete deterministic provenance for a local smoke model."""

    return ExpertPackageExportConfig(
        specialization_name="code",
        specialization_version="1.0.0",
        base_model_repository="local-random-qwen3",
        base_model_revision="synthetic-seed-13",
        base_config_sha256=(
            base_config_sha256 or sha256_json(model.config.to_dict())
        ),
        base_weight_sha256={
            "in-memory-anchor-state": fingerprint_module(model)
        },
        tokenizer_repository="local-random-qwen3",
        tokenizer_revision="synthetic-seed-13",
        base_model_dtype="float32",
        runtime_versions={
            "accelerate": "test",
            "kernels": "torch-default-cpu",
            "peft": "not-installed",
            "python": "test",
            "safetensors": "test",
            "torch": torch.__version__,
            "transformers": "test",
        },
        training_dataset={
            "dataset_id": "synthetic-local-code",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        router_feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 16,
            "version": 1,
        },
        expert_parameterization_config={
            "initialization": "anchor_copy",
            "training_strategy": "forced_path",
        },
    )


def _bindings(
    model: torch.nn.Module,
) -> tuple[ExpertPackageExportBinding, ...]:
    """Copy one independently owned expert for each selected layer."""

    return tuple(
        ExpertPackageExportBinding(
            layer_index=layer_index,
            expert_id="code:0",
            expert=FullMLPExpert.from_anchor(
                model.model.layers[layer_index].mlp
            ),
        )
        for layer_index in (2, 3)
    )


def test_export_is_deterministic_bound_and_excludes_anchor(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Export exact payload bindings and reproduce portable package identity."""

    first_bindings = _bindings(tiny_qwen3_model)
    first = export_full_mlp_expert_package(
        tmp_path / "first",
        bindings=first_bindings,
        config=_config(tiny_qwen3_model),
    )
    second = export_full_mlp_expert_package(
        tmp_path / "second",
        bindings=_bindings(tiny_qwen3_model),
        config=_config(tiny_qwen3_model),
    )

    assert first.identity.manifest_sha256 == second.identity.manifest_sha256
    assert first.manifest.package_hashes == second.manifest.package_hashes
    assert first.manifest.package_structure == "single_expert"
    assert first.manifest.expert_parameterization == "full_mlp"
    assert first.manifest.expert_count == 1
    assert first.manifest.weight_dtype == "float32"
    assert first.manifest.router_architecture == "none"
    assert first.manifest.router_files == ()
    assert first.manifest.routing_policies_supported == ()
    assert first.validation_report.verified_files == (
        "experts/layer-0002-expert-0000.safetensors",
        "experts/layer-0003-expert-0000.safetensors",
    )
    assert [binding.key for binding in first.payload_bindings] == [
        (2, "code:0"),
        (3, "code:0"),
    ]
    assert {path.name for path in first.package_directory.rglob("*") if path.is_file()} == {
        "layer-0002-expert-0000.safetensors",
        "layer-0003-expert-0000.safetensors",
        "manifest.json",
    }
    assert not any(
        "anchor" in path.name
        for path in first.package_directory.rglob("*")
        if path.is_file()
    )

    restored = FullMLPExpert.from_safetensors(
        tiny_qwen3_model.model.layers[2].mlp,
        first.package_directory
        / "experts/layer-0002-expert-0000.safetensors",
        trainable=False,
    )
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in first_bindings[0].expert.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_export_rejects_existing_incomplete_and_aliased_topologies(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Fail before publication when ownership is ambiguous or incomplete."""

    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_full_mlp_expert_package(
            destination,
            bindings=_bindings(tiny_qwen3_model),
            config=_config(tiny_qwen3_model),
        )

    code = _bindings(tiny_qwen3_model)
    incomplete = (
        *code,
        ExpertPackageExportBinding(
            2,
            "math:0",
            FullMLPExpert.from_anchor(
                tiny_qwen3_model.model.layers[2].mlp
            ),
        ),
    )
    with pytest.raises(ValueError, match="complete layer-by-expert rectangle"):
        export_full_mlp_expert_package(
            tmp_path / "incomplete",
            bindings=incomplete,
            config=_config(tiny_qwen3_model),
        )

    shared = FullMLPExpert.from_anchor(
        tiny_qwen3_model.model.layers[2].mlp
    )
    aliased = (
        ExpertPackageExportBinding(2, "code:0", shared),
        ExpertPackageExportBinding(3, "code:0", shared),
    )
    with pytest.raises(ValueError, match="independent expert object"):
        export_full_mlp_expert_package(
            tmp_path / "aliased",
            bindings=aliased,
            config=_config(tiny_qwen3_model),
        )


def test_failed_manifest_validation_removes_staging_directory(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Never publish or retain staged payloads after manifest rejection."""

    destination = tmp_path / "invalid"
    with pytest.raises(ExpertPackageExportError, match="base_config_sha256"):
        export_full_mlp_expert_package(
            destination,
            bindings=_bindings(tiny_qwen3_model),
            config=_config(
                tiny_qwen3_model,
                base_config_sha256="not-a-sha256",
            ),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".invalid.staging-*"))


def test_export_config_defensively_copies_and_reserves_binding_metadata(
    tiny_qwen3_model: torch.nn.Module,
) -> None:
    """Keep caller mutations and manual binding injection out of export state."""

    runtime_versions = {
        "accelerate": "test",
        "kernels": "torch-default-cpu",
        "peft": "not-installed",
        "python": "test",
        "safetensors": "test",
        "torch": "test",
        "transformers": "test",
    }
    config = ExpertPackageExportConfig(
        specialization_name="code",
        specialization_version="1.0.0",
        base_model_repository="local-random-qwen3",
        base_model_revision="seed-13",
        base_config_sha256="0" * 64,
        base_weight_sha256={"model.safetensors": "1" * 64},
        tokenizer_repository="local-random-qwen3",
        tokenizer_revision="seed-13",
        base_model_dtype="float32",
        runtime_versions=runtime_versions,
        training_dataset={
            "dataset_id": "local/code",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        router_feature_specification={"output_size": 16},
    )
    runtime_versions["torch"] = "mutated"

    assert config.runtime_versions["torch"] == "test"
    with pytest.raises(ValueError, match="exporter-owned binding keys"):
        ExpertPackageExportConfig(
            specialization_name="code",
            specialization_version="1.0.0",
            base_model_repository="local-random-qwen3",
            base_model_revision="seed-13",
            base_config_sha256="0" * 64,
            base_weight_sha256={"model.safetensors": "1" * 64},
            tokenizer_repository="local-random-qwen3",
            tokenizer_revision="seed-13",
            base_model_dtype="float32",
            runtime_versions=runtime_versions,
            training_dataset={
                "dataset_id": "local/code",
                "revision": "1",
                "split": "train",
                "filters": [],
            },
            router_feature_specification={"output_size": 16},
            expert_parameterization_config={
                "payload_bindings": [],
            },
        )
