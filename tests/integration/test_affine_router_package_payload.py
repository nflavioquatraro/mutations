"""Integration of RTR-001/RTR-002 payloads with the package manifest."""

from __future__ import annotations

import platform
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    ExpertPackageManifest,
    IdentityRouterFeatureExtractor,
)
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def test_affine_router_payload_is_accepted_by_expert_package_manifest(
    tmp_path: Path,
) -> None:
    """Export router heads and validate them beside an expert payload."""

    expert_relative_path = "experts/layer-2-expert-0.safetensors"
    expert_path = tmp_path / expert_relative_path
    expert_path.parent.mkdir()
    save_file({"expert.weight": torch.zeros(4, 4)}, expert_path)

    router_relative_path = "routers/affine-heads.safetensors"
    router_path = tmp_path / router_relative_path
    router_path.parent.mkdir()
    router = AppendOnlyRouterHeads(feature_size=4)
    router.append(
        AffineRouterHead(
            4,
            layer_index=2,
            expert_id="synthetic-full-mlp:0",
            trainable=False,
        )
    )
    router.save_safetensors(router_path)
    feature_extractor = IdentityRouterFeatureExtractor(feature_size=4)

    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="synthetic-affine-router",
        specialization_version="0.1.0",
        package_structure="single_expert",
        base_model_repository="local-random-model",
        base_model_revision="synthetic-seed-13",
        base_config_sha256=sha256_json({"hidden_size": 4}),
        base_weight_sha256={"model.safetensors": "0" * 64},
        tokenizer_repository="local-random-model",
        tokenizer_revision="synthetic-seed-13",
        base_model_dtype="float32",
        converted_layers=[2],
        expert_parameterization="full_mlp",
        expert_parameterization_config={"initialization": "anchor_copy"},
        expert_count=1,
        router_architecture="independent_affine_heads",
        router_feature_specification=dict(
            feature_extractor.feature_specification
        ),
        routing_policies_supported=["anchored_top1"],
        weight_dtype="float32",
        expert_weight_files=[expert_relative_path],
        router_files=[router_relative_path],
        runtime_versions={
            "accelerate": "not-installed",
            "kernels": "torch-default-cpu",
            "peft": "not-installed",
            "python": platform.python_version(),
            "safetensors": safetensors.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        training_dataset={
            "dataset_id": "synthetic-local-fixture",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        package_hashes={
            expert_relative_path: sha256_file(expert_path),
            router_relative_path: sha256_file(router_path),
        },
    )

    report = manifest.validate_package_files(tmp_path)

    assert report.verified_files == (
        expert_relative_path,
        router_relative_path,
    )
