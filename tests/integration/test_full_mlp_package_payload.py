"""Integration of EXP-001 safetensors with the PKG-001 manifest."""

from __future__ import annotations

import platform
from pathlib import Path

import safetensors
import torch
import transformers
from transformers import Qwen3ForCausalLM

from anchored_moe import ExpertPackageManifest, FullMLPExpert
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def test_full_mlp_payload_is_accepted_by_expert_package_manifest(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """Export one expert and validate it as a complete package payload."""

    relative_path = "experts/layer-2-expert-0.safetensors"
    payload_path = tmp_path / relative_path
    payload_path.parent.mkdir()
    expert = FullMLPExpert.from_anchor(tiny_qwen3_model.model.layers[2].mlp)
    expert.save_safetensors(payload_path)

    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="synthetic-full-mlp",
        specialization_version="0.1.0",
        package_structure="single_expert",
        base_model_repository="local-random-qwen3",
        base_model_revision="synthetic-seed-13",
        base_config_sha256=sha256_json(tiny_qwen3_model.config.to_dict()),
        base_weight_sha256={"model.safetensors": "0" * 64},
        tokenizer_repository="local-random-qwen3",
        tokenizer_revision="synthetic-seed-13",
        base_model_dtype="float32",
        converted_layers=[2],
        expert_parameterization="full_mlp",
        expert_parameterization_config={
            "initialization": "anchor_copy",
            "serialization_format": "anchored_moe.full_mlp.v1",
        },
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=[relative_path],
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
        package_hashes={relative_path: sha256_file(payload_path)},
    )

    report = manifest.validate_package_files(tmp_path)

    assert report.verified_files == (relative_path,)
    assert report.verified_bytes == payload_path.stat().st_size
