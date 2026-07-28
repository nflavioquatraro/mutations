"""End-to-end PKG-007 package materialization and model lifecycle tests."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest
import safetensors
import torch
import transformers
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    COMPOSITION_MANIFEST_FILENAME,
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    AffineRouterHead,
    AnchoredModelError,
    AnchoredMoEConfig,
    AnchoredMoEForCausalLM,
    AppendOnlyRouterHeads,
    ExpertPackage,
    ExpertPackageManifest,
    ExpertRegistry,
    FullMLPExpert,
    LogicalAppendOnlyRouter,
    PackageMaterializationError,
    RoutedMLP,
    synchronize_anchored_top1_registry,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import assert_tensors_equivalent
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


def _runtime_package(
    package_root: Path,
    model: Qwen3ForCausalLM,
    *,
    specialization: str = "code",
    specialization_version: str = "1.0.0",
    expert_parameter_value: float = 0.0,
    router_expert_id: str | None = None,
) -> ExpertPackage:
    """Create one deterministic two-layer full-MLP package fixture."""

    package_root.mkdir()
    expert_id = f"{specialization}:0"
    router_expert_id = router_expert_id or expert_id
    payload_bindings: list[dict[str, object]] = []
    weight_files: list[str] = []
    package_hashes: dict[str, str] = {}
    for layer_index in (2, 3):
        wrapper = model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        expert = FullMLPExpert.from_anchor(
            wrapper.anchor_mlp,
            trainable=False,
        )
        with torch.no_grad():
            for parameter in expert.parameters():
                parameter.fill_(expert_parameter_value)
        relative_path = (
            f"experts/layer-{layer_index}-{specialization}-0.safetensors"
        )
        payload_path = package_root / relative_path
        payload_path.parent.mkdir(exist_ok=True)
        expert.save_safetensors(payload_path)
        weight_files.append(relative_path)
        package_hashes[relative_path] = sha256_file(payload_path)
        payload_bindings.append(
            {
                "layer_index": layer_index,
                "expert_id": expert_id,
                "weight_file": relative_path,
            }
        )

    router = AppendOnlyRouterHeads(feature_size=16)
    for layer_index in (2, 3):
        head = AffineRouterHead(
            16,
            layer_index=layer_index,
            expert_id=router_expert_id,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.bias.fill_(1.0)
        router.append(head)
    router_relative = "routers/affine-heads.safetensors"
    router_path = package_root / router_relative
    router_path.parent.mkdir()
    router.save_safetensors(router_path)
    package_hashes[router_relative] = sha256_file(router_path)

    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name=specialization,
        specialization_version=specialization_version,
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
            "serialization_format": "anchored_moe.full_mlp.v1",
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
        package_hashes=package_hashes,
    )
    manifest.save_json(package_root / "manifest.json")
    return ExpertPackage.from_directory(package_root)


def test_package_install_sync_unload_and_reinsert_restore_model_routes(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    input_ids = torch.tensor([[1, 7, 11, 3], [2, 5, 13, 9]])
    with torch.no_grad():
        dense_logits = tiny_qwen3_model(input_ids=input_ids).logits

    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    package = _runtime_package(tmp_path / "code-package", tiny_qwen3_model)
    registry = ExpertRegistry()

    registry.install(package)
    installed_report = synchronize_anchored_top1_registry(
        tiny_qwen3_model,
        backend=backend,
        registry=registry,
    )
    with torch.no_grad():
        installed_logits = tiny_qwen3_model(input_ids=input_ids).logits

    assert installed_report.configured_layers == (2, 3)
    assert installed_report.expert_ids_by_layer == (
        (2, ("code:0",)),
        (3, ("code:0",)),
    )
    assert all(
        isinstance(
            tiny_qwen3_model.model.layers[layer_index].mlp.router_heads,
            LogicalAppendOnlyRouter,
        )
        for layer_index in (2, 3)
    )
    assert not torch.equal(installed_logits, dense_logits)

    registry.unload(package.identity)
    removed_report = synchronize_anchored_top1_registry(
        tiny_qwen3_model,
        backend=backend,
        registry=registry,
    )
    with torch.no_grad():
        removed_logits = tiny_qwen3_model(input_ids=input_ids).logits

    assert removed_report.configured_layers == ()
    assert removed_report.reset_layers == (2, 3)
    assert_tensors_equivalent(
        removed_logits,
        dense_logits,
        name="post-unload forced-anchor logits",
        atol=0.0,
        rtol=0.0,
    )

    registry.install(ExpertPackage.from_directory(package.package_directory))
    reinserted_report = synchronize_anchored_top1_registry(
        tiny_qwen3_model,
        backend=backend,
        registry=registry,
    )
    with torch.no_grad():
        reinserted_logits = tiny_qwen3_model(input_ids=input_ids).logits

    assert (
        reinserted_report.registry_sha256
        == installed_report.registry_sha256
    )
    assert_tensors_equivalent(
        reinserted_logits,
        installed_logits,
        name="reinserted routed logits",
        atol=0.0,
        rtol=0.0,
    )


def test_local_composition_round_trip_restores_registry_routes_and_logits(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """CORE-006 must preserve package bytes and exact routed behavior."""

    anchor_checkpoint = tmp_path / "anchor"
    tiny_qwen3_model.save_pretrained(anchor_checkpoint)
    model = AnchoredMoEForCausalLM.from_pretrained(
        anchor_checkpoint,
        converted_layers=[2, 3],
        local_files_only=True,
    ).eval()
    package = _runtime_package(
        tmp_path / "code-package",
        model.anchor_model,
    )
    model.install(package)
    synchronized = model.synchronize_registry()
    model.generation_config.max_new_tokens = 7

    input_ids = torch.tensor([[1, 7, 11, 3], [2, 5, 13, 9]])
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    composition = tmp_path / "composition"
    model.save_pretrained(composition)

    assert (composition / COMPOSITION_MANIFEST_FILENAME).is_file()
    assert not (composition / "model.safetensors").exists()
    bundled_package = next((composition / "packages").iterdir())
    for relative_path in (
        *package.manifest.expert_weight_files,
        *package.manifest.router_files,
        "manifest.json",
    ):
        assert (bundled_package / relative_path).read_bytes() == (
            package.package_directory / relative_path
        ).read_bytes()

    reloaded = AnchoredMoEForCausalLM.from_pretrained(
        composition,
        local_files_only=True,
    ).eval()
    with torch.no_grad():
        actual_logits = reloaded(input_ids=input_ids).logits

    assert reloaded.registry.active_identities == (package.identity,)
    assert (
        reloaded.registry.active_registry_sha256
        == synchronized.registry_sha256
        == reloaded.synchronized_registry_sha256
    )
    assert reloaded.generation_config.max_new_tokens == 7
    assert_tensors_equivalent(
        actual_logits,
        expected_logits,
        name="CORE-006 routed round-trip logits",
        atol=0.0,
        rtol=0.0,
    )
    for layer_index in (2, 3):
        wrapper = reloaded.anchor_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "anchored_top1"
        assert wrapper.expert_ids == ("code:0",)


def test_composition_save_rejects_stale_registry_and_existing_destination(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """Serialization must never hide synchronization or replace an artifact."""

    anchor_checkpoint = tmp_path / "anchor"
    tiny_qwen3_model.save_pretrained(anchor_checkpoint)
    model = AnchoredMoEForCausalLM.from_pretrained(
        anchor_checkpoint,
        converted_layers=[2, 3],
        local_files_only=True,
    )
    package = _runtime_package(
        tmp_path / "code-package",
        model.anchor_model,
    )
    model.install(package)
    destination = tmp_path / "composition"

    with pytest.raises(AnchoredModelError, match="stale composition"):
        model.save_pretrained(destination)
    assert not destination.exists()

    model.synchronize_registry()
    model.save_pretrained(destination)
    with pytest.raises(FileExistsError, match="already exists"):
        model.save_pretrained(destination)


def test_composition_load_rejects_tampered_bundled_package(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """Strict package hashes must be checked before an anchor is composed."""

    anchor_checkpoint = tmp_path / "anchor"
    tiny_qwen3_model.save_pretrained(anchor_checkpoint)
    model = AnchoredMoEForCausalLM.from_pretrained(
        anchor_checkpoint,
        converted_layers=[2, 3],
        local_files_only=True,
    )
    package = _runtime_package(
        tmp_path / "code-package",
        model.anchor_model,
    )
    model.install(package)
    model.synchronize_registry()
    composition = tmp_path / "composition"
    model.save_pretrained(composition)

    bundled_package = next((composition / "packages").iterdir())
    payload = bundled_package / package.manifest.expert_weight_files[0]
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(AnchoredModelError, match="Cannot load anchored composition"):
        AnchoredMoEForCausalLM.from_pretrained(
            composition,
            local_files_only=True,
        )


def test_composition_load_rejects_incompatible_relocated_anchor(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """A relocated anchor must retain the saved configuration identity."""

    anchor_checkpoint = tmp_path / "anchor"
    tiny_qwen3_model.save_pretrained(anchor_checkpoint)
    model = AnchoredMoEForCausalLM.from_pretrained(
        anchor_checkpoint,
        converted_layers=[2, 3],
        local_files_only=True,
    )
    composition = tmp_path / "composition"
    model.save_pretrained(composition)

    incompatible_config = tiny_qwen3_model.config.to_dict()
    incompatible_config["vocab_size"] += 1
    incompatible_anchor = Qwen3ForCausalLM(
        type(tiny_qwen3_model.config).from_dict(incompatible_config)
    )
    incompatible_checkpoint = tmp_path / "incompatible-anchor"
    incompatible_anchor.save_pretrained(incompatible_checkpoint)

    with pytest.raises(AnchoredModelError, match="base_config_sha256"):
        AnchoredMoEForCausalLM.from_pretrained(
            composition,
            anchor_model_name_or_path=incompatible_checkpoint,
            local_files_only=True,
        )


def test_model_upgrade_and_rollback_require_explicit_synchronization(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    """PKG-004 registry transactions must never mutate live routes implicitly."""

    input_ids = torch.tensor([[1, 7, 11, 3]])
    model = AnchoredMoEForCausalLM.from_anchor_model(
        tiny_qwen3_model,
        converted_layers=[2, 3],
    ).eval()
    code_v1 = _runtime_package(
        tmp_path / "code-v1",
        model.anchor_model,
        specialization_version="1.0.0",
        expert_parameter_value=0.0,
    )
    code_v2 = _runtime_package(
        tmp_path / "code-v2",
        model.anchor_model,
        specialization_version="2.0.0",
        expert_parameter_value=0.05,
    )
    model.install(code_v1)
    model.synchronize_registry()
    with torch.no_grad():
        v1_logits = model(input_ids=input_ids).logits

    upgrade = model.upgrade(code_v1.identity, code_v2)
    with torch.no_grad():
        unsynchronized_upgrade_logits = model(input_ids=input_ids).logits

    assert model.registry.active_identities == (code_v2.identity,)
    assert_tensors_equivalent(
        unsynchronized_upgrade_logits,
        v1_logits,
        name="unsynchronized upgrade logits",
        atol=0.0,
        rtol=0.0,
    )

    model.synchronize_registry()
    with torch.no_grad():
        v2_logits = model(input_ids=input_ids).logits
    assert not torch.equal(v2_logits, v1_logits)

    rollback = model.rollback(upgrade, code_v1)
    with torch.no_grad():
        unsynchronized_rollback_logits = model(input_ids=input_ids).logits

    assert rollback.after == upgrade.before
    assert model.registry.active_identities == (code_v1.identity,)
    assert_tensors_equivalent(
        unsynchronized_rollback_logits,
        v2_logits,
        name="unsynchronized rollback logits",
        atol=0.0,
        rtol=0.0,
    )

    model.synchronize_registry()
    with torch.no_grad():
        restored_logits = model(input_ids=input_ids).logits
    assert_tensors_equivalent(
        restored_logits,
        v1_logits,
        name="rolled-back routed logits",
        atol=0.0,
        rtol=0.0,
    )


def test_invalid_router_binding_fails_before_model_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    package = _runtime_package(
        tmp_path / "mismatched-package",
        tiny_qwen3_model,
        router_expert_id="math:0",
    )
    registry = ExpertRegistry()
    registry.install(package)

    with pytest.raises(PackageMaterializationError, match="exactly match"):
        synchronize_anchored_top1_registry(
            tiny_qwen3_model,
            backend=backend,
            registry=registry,
        )

    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"
        assert wrapper.expert_ids == ()


def test_multiple_packages_follow_registry_then_router_append_order(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    code = _runtime_package(
        tmp_path / "code-package",
        tiny_qwen3_model,
        specialization="code",
    )
    math = _runtime_package(
        tmp_path / "math-package",
        tiny_qwen3_model,
        specialization="math",
    )
    registry = ExpertRegistry()
    registry.install(code)
    registry.install(math)

    report = synchronize_anchored_top1_registry(
        tiny_qwen3_model,
        backend=backend,
        registry=registry,
    )

    assert report.active_packages == (code.identity, math.identity)
    assert report.expert_ids_by_layer == (
        (2, ("code:0", "math:0")),
        (3, ("code:0", "math:0")),
    )


def test_live_payload_tampering_is_rejected_before_model_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    package = _runtime_package(
        tmp_path / "tampered-package",
        tiny_qwen3_model,
    )
    registry = ExpertRegistry()
    registry.install(package)
    payload = (
        package.package_directory
        / package.manifest.expert_weight_files[0]
    )
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(PackageMaterializationError, match="not attachable"):
        synchronize_anchored_top1_registry(
            tiny_qwen3_model,
            backend=backend,
            registry=registry,
        )

    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        assert wrapper.routing_mode == "forced_anchor"
