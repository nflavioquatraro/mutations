"""Verify explicit package materialization and model lifecycle offline."""

from __future__ import annotations

import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import safetensors
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import (
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    ExpertPackage,
    ExpertPackageManifest,
    ExpertRegistry,
    FullMLPExpert,
    RoutedMLP,
    synchronize_anchored_top1_registry,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import assert_tensors_equivalent
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _tiny_model() -> Qwen3ForCausalLM:
    """Create the deterministic local Qwen3 fixture used by the example."""

    torch.manual_seed(13)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
    ).eval()


def _export_package(root: Path, model: Qwen3ForCausalLM) -> ExpertPackage:
    """Export one safely bound two-layer package with positive router scores."""

    weight_files: list[str] = []
    bindings: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for layer_index in (2, 3):
        wrapper = model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        expert = FullMLPExpert.from_anchor(
            wrapper.anchor_mlp,
            trainable=False,
        )
        relative = f"experts/layer-{layer_index}-code-0.safetensors"
        destination = root / relative
        destination.parent.mkdir(exist_ok=True)
        expert.save_safetensors(destination)
        weight_files.append(relative)
        bindings.append(
            {
                "layer_index": layer_index,
                "expert_id": "code:0",
                "weight_file": relative,
            }
        )
        hashes[relative] = sha256_file(destination)

    router = AppendOnlyRouterHeads(16)
    for layer_index in (2, 3):
        head = AffineRouterHead(
            16,
            layer_index=layer_index,
            expert_id="code:0",
            trainable=False,
        )
        with torch.no_grad():
            head.linear.bias.fill_(1.0)
        router.append(head)
    router_relative = "routers/affine-heads.safetensors"
    router_path = root / router_relative
    router_path.parent.mkdir()
    router.save_safetensors(router_path)
    hashes[router_relative] = sha256_file(router_path)

    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="code",
        specialization_version="1.0.0",
        package_structure="single_expert",
        base_model_repository="local-random-qwen3",
        base_model_revision="synthetic-seed-13",
        base_config_sha256=sha256_json(model.config.to_dict()),
        base_weight_sha256={"model.safetensors": "0" * 64},
        tokenizer_repository="local-random-qwen3",
        tokenizer_revision="synthetic-seed-13",
        base_model_dtype="float32",
        converted_layers=[2, 3],
        expert_parameterization="full_mlp",
        expert_parameterization_config={
            "initialization": "anchor_copy",
            "payload_binding_format_version": (
                EXPERT_PAYLOAD_BINDING_FORMAT_VERSION
            ),
            "payload_bindings": bindings,
        },
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=weight_files,
        router_architecture="independent_affine_heads",
        routing_policies_supported=["anchored_top1"],
        router_feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 16,
            "version": 1,
        },
        router_files=[router_relative],
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
        package_hashes=hashes,
    )
    manifest.save_json(root / "manifest.json")
    return ExpertPackage.from_directory(root)


def main() -> None:
    """Exercise install, explicit sync, unload, and exact reinsertion."""

    model = _tiny_model()
    input_ids = torch.tensor([[1, 7, 11, 3]])
    with torch.no_grad():
        dense_logits = model(input_ids=input_ids).logits

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[2, 3]))
    registry = ExpertRegistry()

    with TemporaryDirectory(prefix="anchored-moe-runtime-") as temporary:
        package = _export_package(Path(temporary), model)
        registry.install(package)
        installed = synchronize_anchored_top1_registry(
            model,
            backend=backend,
            registry=registry,
        )
        with torch.no_grad():
            routed_logits = model(input_ids=input_ids).logits

        registry.unload(package.identity)
        synchronize_anchored_top1_registry(
            model,
            backend=backend,
            registry=registry,
        )
        with torch.no_grad():
            removed_logits = model(input_ids=input_ids).logits
        assert_tensors_equivalent(
            removed_logits,
            dense_logits,
            name="unloaded anchor logits",
            atol=0.0,
            rtol=0.0,
        )

        registry.install(ExpertPackage.from_directory(package.package_directory))
        reinserted = synchronize_anchored_top1_registry(
            model,
            backend=backend,
            registry=registry,
        )
        with torch.no_grad():
            reinserted_logits = model(input_ids=input_ids).logits
        assert_tensors_equivalent(
            reinserted_logits,
            routed_logits,
            name="reinserted routed logits",
            atol=0.0,
            rtol=0.0,
        )
        assert reinserted.registry_sha256 == installed.registry_sha256

        print(
            "PKG-007 runtime validation passed: "
            f"layers={installed.configured_layers}, "
            f"experts={installed.expert_ids_by_layer}, "
            f"registry_sha256={installed.registry_sha256}"
        )


if __name__ == "__main__":
    main()
