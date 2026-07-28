"""Export, reload, remove, upgrade, and roll back an anchored composition.

This NB-007 companion example uses a deterministic local Qwen3 fixture and
only public ``anchored_moe`` APIs. It performs no network access and keeps the
anchor external to the saved composition, matching the CORE-006 checkpoint
contract.
"""

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
    AnchoredMoEForCausalLM,
    AppendOnlyRouterHeads,
    ExpertPackage,
    ExpertPackageManifest,
    FullMLPExpert,
    RoutedMLP,
)
from anchored_moe.evaluation import assert_tensors_equivalent
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _tiny_anchor() -> Qwen3ForCausalLM:
    """Return the deterministic external anchor used by the offline workflow."""

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


def _export_package_version(
    destination: Path,
    model: AnchoredMoEForCausalLM,
    *,
    version: str,
    expert_parameter_value: float,
) -> ExpertPackage:
    """Create one immutable, attachable package version for the tiny anchor."""

    destination.mkdir()
    weight_files: list[str] = []
    payload_bindings: list[dict[str, object]] = []
    package_hashes: dict[str, str] = {}
    for layer_index in (2, 3):
        wrapper = model.anchor_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp, trainable=False)
        with torch.no_grad():
            for parameter in expert.parameters():
                parameter.fill_(expert_parameter_value)
        relative_path = f"experts/layer-{layer_index}-code-0.safetensors"
        payload_path = destination / relative_path
        payload_path.parent.mkdir(exist_ok=True)
        expert.save_safetensors(payload_path)
        weight_files.append(relative_path)
        payload_bindings.append(
            {
                "layer_index": layer_index,
                "expert_id": "code:0",
                "weight_file": relative_path,
            }
        )
        package_hashes[relative_path] = sha256_file(payload_path)

    router = AppendOnlyRouterHeads(model.anchor_model.config.hidden_size)
    for layer_index in (2, 3):
        head = AffineRouterHead(
            model.anchor_model.config.hidden_size,
            layer_index=layer_index,
            expert_id="code:0",
            trainable=False,
        )
        with torch.no_grad():
            head.linear.bias.fill_(1.0)
        router.append(head)
    router_relative_path = "routers/affine-heads.safetensors"
    router_path = destination / router_relative_path
    router_path.parent.mkdir()
    router.save_safetensors(router_path)
    package_hashes[router_relative_path] = sha256_file(router_path)

    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="code",
        specialization_version=version,
        package_structure="single_expert",
        base_model_repository="local-random-qwen3",
        base_model_revision="synthetic-seed-13",
        base_config_sha256=sha256_json(model.anchor_model.config.to_dict()),
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
            "payload_bindings": payload_bindings,
        },
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=weight_files,
        router_architecture="independent_affine_heads",
        routing_policies_supported=["anchored_top1"],
        router_feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": model.anchor_model.config.hidden_size,
            "version": 1,
        },
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
            "revision": version,
            "split": "train",
            "filters": [],
        },
        package_hashes=package_hashes,
    )
    manifest.save_json(destination / "manifest.json")
    return ExpertPackage.from_directory(destination)


def main() -> None:
    """Run the complete NB-007 lifecycle and assert exact state restoration."""

    input_ids = torch.tensor([[1, 7, 11, 3]])
    anchor = _tiny_anchor()
    with torch.no_grad():
        dense_logits = anchor(input_ids=input_ids).logits

    with TemporaryDirectory(prefix="anchored-moe-nb007-") as temporary:
        workspace = Path(temporary)
        anchor_checkpoint = workspace / "anchor"
        anchor.save_pretrained(anchor_checkpoint)
        model = AnchoredMoEForCausalLM.from_pretrained(
            anchor_checkpoint,
            converted_layers=[2, 3],
            local_files_only=True,
        ).eval()
        code_v1 = _export_package_version(
            workspace / "code-v1",
            model,
            version="1.0.0",
            expert_parameter_value=0.0,
        )
        code_v2 = _export_package_version(
            workspace / "code-v2",
            model,
            version="2.0.0",
            expert_parameter_value=0.05,
        )

        model.install(code_v1)
        installed = model.synchronize_registry()
        with torch.no_grad():
            v1_logits = model(input_ids=input_ids).logits
        assert not torch.equal(v1_logits, dense_logits)

        composition_checkpoint = workspace / "composition-v1"
        model.save_pretrained(composition_checkpoint)
        reloaded = AnchoredMoEForCausalLM.from_pretrained(
            composition_checkpoint,
            local_files_only=True,
        ).eval()
        with torch.no_grad():
            reloaded_v1_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            reloaded_v1_logits,
            v1_logits,
            name="reloaded v1 composition logits",
            atol=0.0,
            rtol=0.0,
        )
        assert reloaded.synchronized_registry_sha256 == installed.registry_sha256

        reloaded.unload(code_v1.identity)
        with torch.no_grad():
            stale_unload_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            stale_unload_logits,
            v1_logits,
            name="unsynchronized unload logits",
            atol=0.0,
            rtol=0.0,
        )
        reloaded.synchronize_registry()
        with torch.no_grad():
            removed_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            removed_logits,
            dense_logits,
            name="removed anchor logits",
            atol=0.0,
            rtol=0.0,
        )

        reloaded.install(
            ExpertPackage.from_directory(code_v1.package_directory)
        )
        with torch.no_grad():
            stale_reinsert_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            stale_reinsert_logits,
            dense_logits,
            name="unsynchronized reinsertion logits",
            atol=0.0,
            rtol=0.0,
        )
        reinserted = reloaded.synchronize_registry()
        with torch.no_grad():
            reinserted_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            reinserted_logits,
            v1_logits,
            name="reinserted v1 logits",
            atol=0.0,
            rtol=0.0,
        )
        assert reinserted.registry_sha256 == installed.registry_sha256

        upgrade = reloaded.upgrade(code_v1.identity, code_v2)
        with torch.no_grad():
            stale_upgrade_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            stale_upgrade_logits,
            v1_logits,
            name="unsynchronized upgrade logits",
            atol=0.0,
            rtol=0.0,
        )
        upgraded = reloaded.synchronize_registry()
        with torch.no_grad():
            v2_logits = reloaded(input_ids=input_ids).logits
        assert not torch.equal(v2_logits, v1_logits)
        assert upgraded.registry_sha256 != installed.registry_sha256

        rollback = reloaded.rollback(upgrade, code_v1)
        with torch.no_grad():
            stale_rollback_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            stale_rollback_logits,
            v2_logits,
            name="unsynchronized rollback logits",
            atol=0.0,
            rtol=0.0,
        )
        restored = reloaded.synchronize_registry()
        with torch.no_grad():
            restored_logits = reloaded(input_ids=input_ids).logits
        assert_tensors_equivalent(
            restored_logits,
            v1_logits,
            name="rolled-back v1 logits",
            atol=0.0,
            rtol=0.0,
        )
        assert rollback.after == upgrade.before
        assert restored.registry_sha256 == installed.registry_sha256
        assert all(not parameter.requires_grad for parameter in reloaded.parameters())

        print(
            "NB-007 validation passed: "
            f"registry_sha256={restored.registry_sha256}, "
            f"reloaded_exact=True, removed_to_anchor=True, "
            f"reinserted_exact=True, upgrade_changed_output=True, "
            f"rollback_exact=True"
        )


if __name__ == "__main__":
    main()
