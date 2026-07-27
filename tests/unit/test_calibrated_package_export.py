"""Acceptance tests for immutable NB-004 router-package derivation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anchored_moe import (
    AnchoredMoEConfig,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    CalibratedRouterPackageConfig,
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ExpertRegistry,
    FullMLPExpert,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    derive_calibrated_router_package,
    export_full_mlp_expert_package,
    fit_independent_ridge,
    synchronize_anchored_top1_registry,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import fingerprint_module
from anchored_moe.utils import sha256_file, sha256_json


def _source_package(
    model: torch.nn.Module,
    package_root: Path,
):
    """Export one deterministic two-layer NB-003 source package."""

    bindings = [
        ExpertPackageExportBinding(
            layer_index,
            "code:0",
            FullMLPExpert.from_anchor(model.model.layers[layer_index].mlp),
        )
        for layer_index in (2, 3)
    ]
    return export_full_mlp_expert_package(
        package_root,
        bindings=bindings,
        config=ExpertPackageExportConfig(
            specialization_name="code",
            specialization_version="1.0.0",
            base_model_repository="local-random-qwen3",
            base_model_revision="synthetic-seed-13",
            base_config_sha256=sha256_json(model.config.to_dict()),
            base_weight_sha256={
                "in-memory-anchor-state": fingerprint_module(model)
            },
            tokenizer_repository="local-random-qwen3",
            tokenizer_revision="synthetic-seed-13",
            base_model_dtype="float32",
            runtime_versions=_runtime_versions(),
            training_dataset={
                "dataset_id": "synthetic-local-code",
                "revision": "1",
                "split": "train",
                "filters": [],
            },
            router_feature_specification=_feature_specification(),
            expert_parameterization_config={
                "initialization": "anchor_copy",
                "training_strategy": "forced_path",
            },
        ),
    )


def _runtime_versions() -> dict[str, str]:
    """Return complete deterministic runtime metadata."""

    return {
        "accelerate": "test",
        "kernels": "torch-default-cpu",
        "peft": "not-installed",
        "python": "test",
        "safetensors": "test",
        "torch": torch.__version__,
        "transformers": "test",
    }


def _feature_specification() -> dict[str, object]:
    """Return the identity feature contract shared with the source package."""

    return {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": 16,
        "version": 1,
    }


def _bank(
    scalar_features: torch.Tensor,
    *,
    row_ids: torch.Tensor,
    split: str,
) -> RouterFeatureBank:
    """Expand one informative scalar into the model's 16-D feature space."""

    features = torch.zeros(
        scalar_features.shape[0],
        16,
        dtype=torch.float32,
    )
    features[:, 0] = scalar_features
    return RouterFeatureBank(
        {2: features, 3: features.clone()},
        feature_specification=_feature_specification(),
        dataset_provenance={
            "dataset_id": "synthetic-local-router",
            "revision": "1",
            "split": split,
            "filters": [],
        },
        seed=73,
        row_ids=row_ids,
    )


def _labels(
    bank: RouterFeatureBank,
    utilities: torch.Tensor,
) -> UtilityLabelBank:
    """Bind identical single-expert utility columns at both package layers."""

    values = utilities.reshape(-1, 1)
    return UtilityLabelBank(
        {2: values, 3: values.clone()},
        expert_ids_by_layer={2: ("code:0",), 3: ("code:0",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def _calibration_inputs():
    """Fit ridge heads and calibrate them on a disjoint held-out bank."""

    fitting_scalars = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    fitting_bank = _bank(
        fitting_scalars,
        row_ids=torch.arange(6),
        split="train",
    )
    fitting_labels = _labels(fitting_bank, fitting_scalars)
    fitted = fit_independent_ridge(
        fitting_bank,
        fitting_labels,
        regularization=1e-6,
    )

    calibration_scalars = torch.tensor(
        [-2.0, -1.0, 1.0, 2.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]
    )
    calibration_utilities = torch.tensor(
        [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    )
    calibration_bank = _bank(
        calibration_scalars,
        row_ids=torch.arange(10, 20),
        split="validation",
    )
    calibration_labels = _labels(
        calibration_bank,
        calibration_utilities,
    )
    calibrated, report = calibrate_temperature_and_thresholds(
        fitted,
        calibration_bank,
        calibration_labels,
        temperature_row_ids=[10, 11, 12, 13],
        threshold_row_ids=[14, 15, 16, 17, 18, 19],
        false_activation_rate=1.0 / 3.0,
    )
    return (
        fitting_bank,
        fitting_labels,
        calibration_bank,
        calibration_labels,
        calibrated,
        report,
    )


def _derivation_config() -> CalibratedRouterPackageConfig:
    """Return the immutable derived-version configuration."""

    return CalibratedRouterPackageConfig(
        specialization_version="1.0.0+router.1",
        calibration_dataset_version="synthetic-router-v1",
        ridge_regularization=1e-6,
        runtime_versions=_runtime_versions(),
    )


def test_derivation_preserves_source_and_is_deterministic_and_attachable(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Derive identical routed artifacts while leaving NB-003 bytes unchanged."""

    source = _source_package(tiny_qwen3_model, tmp_path / "source")
    source_bytes = {
        path.relative_to(source.package_directory).as_posix(): path.read_bytes()
        for path in source.package_directory.rglob("*")
        if path.is_file()
    }
    (
        fitting_bank,
        fitting_labels,
        calibration_bank,
        calibration_labels,
        calibrated,
        report,
    ) = _calibration_inputs()

    first = derive_calibrated_router_package(
        source,
        tmp_path / "derived-first",
        calibrated_router=calibrated,
        calibration_report=report,
        fitting_feature_bank=fitting_bank,
        fitting_utility_labels=fitting_labels,
        calibration_feature_bank=calibration_bank,
        calibration_utility_labels=calibration_labels,
        config=_derivation_config(),
    )
    second = derive_calibrated_router_package(
        source,
        tmp_path / "derived-second",
        calibrated_router=calibrated,
        calibration_report=report,
        fitting_feature_bank=fitting_bank,
        fitting_utility_labels=fitting_labels,
        calibration_feature_bank=calibration_bank,
        calibration_utility_labels=calibration_labels,
        config=_derivation_config(),
    )

    assert {
        path.relative_to(source.package_directory).as_posix(): path.read_bytes()
        for path in source.package_directory.rglob("*")
        if path.is_file()
    } == source_bytes
    assert first.identity.manifest_sha256 == second.identity.manifest_sha256
    assert first.manifest.package_hashes == second.manifest.package_hashes
    assert first.manifest.router_architecture == "independent_affine_heads"
    assert first.manifest.routing_policies_supported == ("anchored_top1",)
    assert first.manifest.router_files == (
        "routers/independent-affine-heads.safetensors",
    )
    assert first.manifest.calibration_dataset_version == "synthetic-router-v1"
    assert (
        first.manifest.calibration_metrics[
            "source_expert_package_manifest_sha256"
        ]
        == source.identity.manifest_sha256
    )
    for relative_path in source.manifest.expert_weight_files:
        assert (
            first.manifest.package_hashes[relative_path]
            == source.manifest.package_hashes[relative_path]
        )
        assert (
            first.package_directory / relative_path
        ).read_bytes() == (
            source.package_directory / relative_path
        ).read_bytes()

    restored_router = AppendOnlyRouterHeads.from_safetensors(
        first.package_directory / first.manifest.router_files[0]
    )
    assert restored_router.head_keys == calibrated.head_keys
    for layer_index in (2, 3):
        expected = calibrated(
            calibration_bank.features_for_layer(layer_index),
            layer_index=layer_index,
        )
        actual = restored_router(
            calibration_bank.features_for_layer(layer_index),
            layer_index=layer_index,
        )
        assert torch.equal(actual, expected)
    assert all(
        not parameter.requires_grad
        for parameter in restored_router.parameters()
    )

    scores = restored_router(
        calibration_bank.features_for_layer(2),
        layer_index=2,
    )
    decision = AnchoredTop1Selection()(
        scores,
        expert_ids=restored_router.expert_ids_for_layer(2),
    )
    assert 0 in decision.selected_indices.tolist()
    assert 1 in decision.selected_indices.tolist()

    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    registry = ExpertRegistry()
    registry.install(first)
    composition = synchronize_anchored_top1_registry(
        tiny_qwen3_model,
        backend=backend,
        registry=registry,
    )
    assert composition.configured_layers == (2, 3)
    assert composition.expert_ids_by_layer == (
        (2, ("code:0",)),
        (3, ("code:0",)),
    )


def test_derivation_rejects_version_data_and_head_mismatches(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Reject mutation-like versioning and mismatched calibration ownership."""

    source = _source_package(tiny_qwen3_model, tmp_path / "source")
    (
        fitting_bank,
        fitting_labels,
        calibration_bank,
        calibration_labels,
        calibrated,
        report,
    ) = _calibration_inputs()
    arguments = {
        "source_package": source,
        "calibrated_router": calibrated,
        "calibration_report": report,
        "fitting_feature_bank": fitting_bank,
        "fitting_utility_labels": fitting_labels,
        "calibration_feature_bank": calibration_bank,
        "calibration_utility_labels": calibration_labels,
    }

    with pytest.raises(ValueError, match="new specialization_version"):
        derive_calibrated_router_package(
            package_directory=tmp_path / "same-version",
            config=CalibratedRouterPackageConfig(
                specialization_version="1.0.0",
                calibration_dataset_version="synthetic-router-v1",
                ridge_regularization=1e-6,
                runtime_versions=_runtime_versions(),
            ),
            **arguments,
        )

    overlapping_bank = _bank(
        torch.arange(10, dtype=torch.float32),
        row_ids=torch.arange(10),
        split="validation-overlap",
    )
    overlapping_labels = _labels(
        overlapping_bank,
        torch.arange(10, dtype=torch.float32) - 4,
    )
    with pytest.raises(ValueError, match="row IDs must be disjoint"):
        derive_calibrated_router_package(
            package_directory=tmp_path / "overlap",
            config=_derivation_config(),
            **{
                **arguments,
                "calibration_feature_bank": overlapping_bank,
                "calibration_utility_labels": overlapping_labels,
            },
        )

    mismatched = AppendOnlyRouterHeads(16)
    mismatched.append(calibrated.heads[0])
    with pytest.raises(ValueError, match="exactly match source"):
        derive_calibrated_router_package(
            package_directory=tmp_path / "mismatched",
            config=_derivation_config(),
            **{
                **arguments,
                "calibrated_router": mismatched,
            },
        )


def test_derived_router_payload_hash_is_manifest_tracked(
    tiny_qwen3_model: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """Bind the exact serialized calibrated router bytes into the manifest."""

    source = _source_package(tiny_qwen3_model, tmp_path / "source")
    (
        fitting_bank,
        fitting_labels,
        calibration_bank,
        calibration_labels,
        calibrated,
        report,
    ) = _calibration_inputs()
    derived = derive_calibrated_router_package(
        source,
        tmp_path / "derived",
        calibrated_router=calibrated,
        calibration_report=report,
        fitting_feature_bank=fitting_bank,
        fitting_utility_labels=fitting_labels,
        calibration_feature_bank=calibration_bank,
        calibration_utility_labels=calibration_labels,
        config=_derivation_config(),
    )

    router_relative = derived.manifest.router_files[0]
    assert derived.manifest.package_hashes[router_relative] == sha256_file(
        derived.package_directory / router_relative
    )
    derived.validate_live_files()
