"""Verify strict package installation, unloading, and reinsertion offline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from anchored_moe import (
    ExpertPackage,
    ExpertPackageManifest,
    ExpertRegistry,
    evaluate_registry_compositions,
)
from anchored_moe.evaluation import evaluate_registry_lifecycle
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _create_package(root: Path, specialization: str, offset: float) -> ExpertPackage:
    """Create one deterministic package with complete local provenance."""

    package_root = root / specialization
    weight_relative = "experts/model.safetensors"
    weight_path = package_root / weight_relative
    weight_path.parent.mkdir(parents=True)
    save_file(
        {
            "expert.weight": (
                torch.arange(4, dtype=torch.float32).reshape(2, 2) + offset
            )
        },
        weight_path,
    )
    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name=specialization,
        specialization_version="1.0.0",
        package_structure="single_expert",
        base_model_repository="Qwen/Qwen3-4B",
        base_model_revision="example-base-revision",
        base_config_sha256=sha256_json({"model_type": "qwen3"}),
        base_weight_sha256={"model.safetensors": "1" * 64},
        tokenizer_repository="Qwen/Qwen3-4B",
        tokenizer_revision="example-tokenizer-revision",
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
            "python": "3.12",
            "safetensors": "installed",
            "torch": torch.__version__,
            "transformers": "installed",
        },
        training_dataset={
            "dataset_id": f"example/{specialization}",
            "revision": "example-data-revision",
            "split": "train",
            "filters": [],
        },
        package_hashes={weight_relative: sha256_file(weight_path)},
        router_feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 4,
            "version": 1,
        },
        routing_policies_supported=["anchored_top1"],
    )
    manifest.save_json(package_root / "manifest.json")
    return ExpertPackage.from_directory(package_root)


def _package_set_fingerprint(registry: ExpertRegistry) -> str:
    """Hash active package content independently of installation order."""

    identities = sorted(registry.snapshot().active_packages)
    return sha256_json(
        {"packages": [identity.to_dict() for identity in identities]}
    )


def main() -> None:
    """Exercise lifecycle restoration plus every two-package composition."""

    with TemporaryDirectory(prefix="anchored-moe-registry-") as temporary:
        root = Path(temporary)
        code = _create_package(root, "code", 0.0)
        math = _create_package(root, "math", 10.0)
        registry = ExpertRegistry()
        registry.install(code)
        registry.install(math)
        installed = registry.snapshot()

        registry.unload(code.identity)
        removed = registry.snapshot()
        registry.install(ExpertPackage.from_directory(code.package_directory))
        reinserted = registry.snapshot()

        evaluation = evaluate_registry_lifecycle(
            installed,
            removed,
            reinserted,
        )
        assert evaluation.exact
        assert reinserted.active_packages == (code.identity, math.identity)

        compositions = evaluate_registry_compositions(
            (math, code),
            behavior_fingerprint=_package_set_fingerprint,
        )
        assert compositions.subset_count == 4
        assert compositions.trial_count == 5
        assert compositions.registry_order_sensitive_subset_count == 1
        assert compositions.behavior_order_invariant

        print(
            "Registry lifecycle validation passed: "
            f"active={reinserted.active_package_count}, "
            f"registry_sha256={reinserted.registry_sha256}, "
            f"order={[item.specialization_name for item in reinserted.active_packages]}, "
            f"subsets={compositions.subset_count}, "
            f"orders={compositions.trial_count}"
        )


if __name__ == "__main__":
    main()
