"""Construct and validate a synthetic, independently portable expert package."""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file

from anchored_moe import ExpertPackageManifest
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def main() -> None:
    """Build a temporary safetensors package and verify every recorded digest."""

    with tempfile.TemporaryDirectory(prefix="anchored-moe-package-") as directory:
        package_root = Path(directory)
        weight_path = package_root / "experts" / "model.safetensors"
        weight_path.parent.mkdir()
        save_file({"expert.weight": torch.zeros(2, 2)}, weight_path)

        relative_weight_path = "experts/model.safetensors"
        manifest = ExpertPackageManifest(
            package_format_version=PACKAGE_FORMAT_VERSION,
            specialization_name="synthetic-code",
            specialization_version="0.1.0",
            package_structure="single_expert",
            base_model_repository="Qwen/Qwen3-4B",
            base_model_revision="synthetic-pinned-revision",
            base_config_sha256=sha256_json({"model_type": "qwen3"}),
            base_weight_sha256={"model.safetensors": "0" * 64},
            tokenizer_repository="Qwen/Qwen3-4B",
            tokenizer_revision="synthetic-pinned-revision",
            base_model_dtype="float32",
            converted_layers=[2, 3],
            expert_parameterization="full_mlp",
            expert_parameterization_config={"initialization": "anchor_copy"},
            expert_count=1,
            weight_dtype="float32",
            expert_weight_files=[relative_weight_path],
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
                relative_weight_path: sha256_file(weight_path),
            },
        )
        manifest_path = package_root / "manifest.json"
        manifest.save_json(manifest_path)

        reloaded = ExpertPackageManifest.from_json(manifest_path)
        report = reloaded.validate_package_files(package_root)
        print(
            "Expert package validation passed: "
            f"files={report.verified_files}, "
            f"bytes={report.verified_bytes}, "
            f"manifest_sha256={report.manifest_sha256}"
        )


if __name__ == "__main__":
    main()
