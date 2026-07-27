"""Deterministic test fixtures for the foundation implementation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe.registry import (
    ExpertPackage,
    ExpertPackageManifest,
    PACKAGE_FORMAT_VERSION,
)
from anchored_moe.utils import sha256_file, sha256_json


@pytest.fixture
def tiny_qwen3_model() -> Qwen3ForCausalLM:
    """Return a deterministic Qwen3 model small enough for CPU unit tests."""

    torch.manual_seed(13)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def expert_package_factory(
    tmp_path: Path,
) -> Callable[..., ExpertPackage]:
    """Create independently validated local package fixtures."""

    package_number = 0

    def create(
        specialization_name: str,
        *,
        specialization_version: str = "1.0.0",
        base_model_revision: str = "base-revision",
        router_output_size: int = 4,
        tensor_offset: float = 0.0,
        strict: bool = True,
        add_untracked_file: bool = False,
    ) -> ExpertPackage:
        nonlocal package_number
        package_number += 1
        package_root = (
            tmp_path
            / f"{specialization_name}-{specialization_version}-{package_number}"
        )
        weight_relative = "experts/model.safetensors"
        weight_path = package_root / weight_relative
        weight_path.parent.mkdir(parents=True)
        save_file(
            {
                "expert.weight": (
                    torch.arange(4, dtype=torch.float32).reshape(2, 2)
                    + tensor_offset
                )
            },
            weight_path,
        )
        manifest = ExpertPackageManifest(
            package_format_version=PACKAGE_FORMAT_VERSION,
            specialization_name=specialization_name,
            specialization_version=specialization_version,
            package_structure="single_expert",
            base_model_repository="Qwen/Qwen3-4B",
            base_model_revision=base_model_revision,
            base_config_sha256=sha256_json({"model_type": "qwen3"}),
            base_weight_sha256={
                "model-00001-of-00002.safetensors": "1" * 64
            },
            tokenizer_repository="Qwen/Qwen3-4B",
            tokenizer_revision="tokenizer-revision",
            base_model_dtype="float32",
            converted_layers=[2, 3],
            expert_parameterization="full_mlp",
            expert_parameterization_config={"initialization": "anchor_copy"},
            expert_count=1,
            weight_dtype="float32",
            expert_weight_files=[weight_relative],
            runtime_versions={
                "accelerate": "not-installed",
                "kernels": "torch-default-cpu",
                "peft": "not-installed",
                "python": "3.12.10",
                "safetensors": "0.8.0",
                "torch": "2.13.0",
                "transformers": "5.14.1",
            },
            training_dataset={
                "dataset_id": f"organization/{specialization_name}-data",
                "revision": "dataset-revision",
                "split": "train",
                "filters": ["quality"],
            },
            package_hashes={weight_relative: sha256_file(weight_path)},
            router_feature_specification={
                "feature_type": "identity",
                "input": "pre_mlp_hidden_state",
                "output_size": router_output_size,
                "version": 1,
            },
            routing_policies_supported=["anchored_top1"],
        )
        manifest.save_json(package_root / "manifest.json")
        if add_untracked_file:
            (package_root / "notes.txt").write_text(
                "untracked",
                encoding="utf-8",
            )
        return ExpertPackage.from_directory(package_root, strict=strict)

    return create
