"""Registry-to-policy integration for PKG-003 and POL-005."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anchored_moe import (
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    CompatibilityConstrainedTop2Selection,
    ExpertPackage,
    ExpertPackageManifest,
    ExpertRegistry,
    MixtureCompatibilityError,
)
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _bound_package(
    root: Path,
    *,
    specialization: str,
    expert_id: str,
    group_id: str | None,
) -> ExpertPackage:
    """Create one strict single-layer package with explicit expert ownership."""

    root.mkdir()
    weight_file = f"experts/{specialization}.safetensors"
    weight_path = root / weight_file
    weight_path.parent.mkdir()
    save_file({"expert.weight": torch.ones(1)}, weight_path)
    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name=specialization,
        specialization_version="1.0.0",
        package_structure="single_expert",
        base_model_repository="local-qwen3",
        base_model_revision="revision-1",
        base_config_sha256=sha256_json({"model_type": "qwen3"}),
        base_weight_sha256={"model.safetensors": "1" * 64},
        tokenizer_repository="local-qwen3",
        tokenizer_revision="revision-1",
        base_model_dtype="float32",
        converted_layers=[2],
        expert_parameterization="full_mlp",
        expert_parameterization_config={
            "initialization": "anchor_copy",
            "payload_binding_format_version": (
                EXPERT_PAYLOAD_BINDING_FORMAT_VERSION
            ),
            "payload_bindings": [
                {
                    "layer_index": 2,
                    "expert_id": expert_id,
                    "weight_file": weight_file,
                }
            ],
        },
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=[weight_file],
        router_architecture="none",
        routing_policies_supported=[],
        router_feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 4,
            "version": 1,
        },
        mixture_compatibility_group=group_id,
        runtime_versions={
            "accelerate": "1",
            "kernels": "torch-default-cpu",
            "peft": "none",
            "python": "3.12",
            "safetensors": "0.8",
            "torch": "2",
            "transformers": "5",
        },
        training_dataset={
            "dataset_id": f"local/{specialization}",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        package_hashes={weight_file: sha256_file(weight_path)},
    )
    manifest.save_json(root / "manifest.json")
    return ExpertPackage.from_directory(root)


def test_registry_index_drives_compatible_selection_and_tracks_unload(
    tmp_path: Path,
) -> None:
    code = _bound_package(
        tmp_path / "code",
        specialization="code",
        expert_id="code:0",
        group_id="reasoning-v1",
    )
    math = _bound_package(
        tmp_path / "math",
        specialization="math",
        expert_id="math:0",
        group_id="reasoning-v1",
    )
    general = _bound_package(
        tmp_path / "general",
        specialization="general",
        expert_id="general:0",
        group_id=None,
    )
    registry = ExpertRegistry()
    registry.install(code)
    registry.install(math)
    registry.install(general)

    index = registry.mixture_compatibility(layer_index=2)
    assert index.expert_ids == ("code:0", "math:0", "general:0")
    assert index.group_ids == ("reasoning-v1", "reasoning-v1", None)

    decision = CompatibilityConstrainedTop2Selection(index)(
        torch.tensor([[0.9, 0.7, 1.2]]),
        expert_ids=index.expert_ids,
    )
    assert torch.equal(decision.selected_indices, torch.tensor([[1, 2]]))
    assert decision.expert_pair_mask.item()

    registry.unload(math.identity)
    reduced = registry.mixture_compatibility(layer_index=2)
    fallback = CompatibilityConstrainedTop2Selection(reduced)(
        torch.tensor([[0.9, 1.2]]),
        expert_ids=reduced.expert_ids,
    )
    assert reduced.expert_ids == ("code:0", "general:0")
    assert torch.equal(fallback.selected_indices, torch.tensor([[2, 0]]))
    assert fallback.anchor_mask.item()


def test_registry_compatibility_requires_explicit_payload_bindings(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    registry = ExpertRegistry()
    registry.install(expert_package_factory("legacy"))

    with pytest.raises(MixtureCompatibilityError, match="explicit payload"):
        registry.mixture_compatibility(layer_index=2)


def test_registry_compatibility_rejects_layer_and_expert_ambiguity(
    tmp_path: Path,
) -> None:
    first = _bound_package(
        tmp_path / "first",
        specialization="first",
        expert_id="shared:0",
        group_id="shared-group",
    )
    second = _bound_package(
        tmp_path / "second",
        specialization="second",
        expert_id="shared:0",
        group_id="shared-group",
    )
    registry = ExpertRegistry()
    registry.install(first)

    with pytest.raises(MixtureCompatibilityError, match="does not cover"):
        registry.mixture_compatibility(layer_index=3)

    registry.install(second)
    with pytest.raises(MixtureCompatibilityError, match="repeat expert"):
        registry.mixture_compatibility(layer_index=2)
