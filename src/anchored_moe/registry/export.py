"""Staged export and calibrated derivation of immutable expert packages."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from anchored_moe.calibration import (
    RouterFeatureBank,
    TemperatureThresholdCalibrationReport,
    UtilityLabelBank,
)
from anchored_moe.experts import ExpertSerializationError, FullMLPExpert
from anchored_moe.registry.bindings import (
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
)
from anchored_moe.registry.manifest import (
    PACKAGE_FORMAT_VERSION,
    ExpertPackageManifest,
    PackageValidationError,
)
from anchored_moe.registry.package import ExpertPackage
from anchored_moe.routers import (
    AppendOnlyRouterHeads,
    RouterSerializationError,
)
from anchored_moe.utils.hashing import sha256_file, sha256_json


_RESERVED_PARAMETERIZATION_KEYS = frozenset(
    {"payload_binding_format_version", "payload_bindings"}
)


class ExpertPackageExportError(RuntimeError):
    """Raised when a staged expert-only package cannot be published safely."""


@dataclass(frozen=True)
class ExpertPackageExportBinding:
    """Bind one trained full-MLP expert to its package-owned runtime slot.

    Attributes:
        layer_index: Zero-based converted decoder layer.
        expert_id: Stable expert identifier shared across package layers.
        expert: Independently owned full-MLP expert to serialize.
    """

    layer_index: int
    expert_id: str
    expert: FullMLPExpert

    def __post_init__(self) -> None:
        """Validate explicit layer ownership and the supported representation."""

        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        if (
            not isinstance(self.expert_id, str)
            or not self.expert_id
            or self.expert_id != self.expert_id.strip()
        ):
            raise ValueError(
                "expert_id must be a non-empty string without surrounding "
                "whitespace."
            )
        if not isinstance(self.expert, FullMLPExpert):
            raise TypeError("expert must be a FullMLPExpert.")


@dataclass(frozen=True)
class ExpertPackageExportConfig:
    """Immutable provenance needed to export one expert-only package.

    Generated values such as converted layers, expert count, payload paths,
    payload bindings, payload hashes, package structure, and weight dtype are
    deliberately absent. The exporter derives them from the supplied experts
    so manifest metadata cannot disagree with serialized tensors.

    ``base_weight_sha256`` should contain the published checkpoint shard
    hashes in production. Deterministic in-memory smoke models may instead
    record a named state-dict fingerprint.
    """

    specialization_name: str
    specialization_version: str
    base_model_repository: str
    base_model_revision: str
    base_config_sha256: str
    base_weight_sha256: Mapping[str, str]
    tokenizer_repository: str
    tokenizer_revision: str
    base_model_dtype: str
    runtime_versions: Mapping[str, str]
    training_dataset: Mapping[str, Any]
    router_feature_specification: Mapping[str, Any]
    expert_parameterization_config: Mapping[str, Any] = field(
        default_factory=lambda: {"initialization": "anchor_copy"}
    )
    training_parent_registry: str | None = None

    def __post_init__(self) -> None:
        """Defensively freeze JSON metadata and reject reserved binding keys."""

        for field_name in (
            "specialization_name",
            "specialization_version",
            "base_model_repository",
            "base_model_revision",
            "base_config_sha256",
            "tokenizer_repository",
            "tokenizer_revision",
            "base_model_dtype",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if (
            self.training_parent_registry is not None
            and (
                not isinstance(self.training_parent_registry, str)
                or not self.training_parent_registry
                or self.training_parent_registry
                != self.training_parent_registry.strip()
            )
        ):
            raise ValueError(
                "training_parent_registry must be None or a non-empty string "
                "without surrounding whitespace."
            )

        base_hashes = _freeze_string_mapping(
            self.base_weight_sha256,
            field_name="base_weight_sha256",
        )
        runtime_versions = _freeze_string_mapping(
            self.runtime_versions,
            field_name="runtime_versions",
        )
        training_dataset = _freeze_json_mapping(
            self.training_dataset,
            field_name="training_dataset",
        )
        router_features = _freeze_json_mapping(
            self.router_feature_specification,
            field_name="router_feature_specification",
        )
        parameterization = _freeze_json_mapping(
            self.expert_parameterization_config,
            field_name="expert_parameterization_config",
        )
        reserved = sorted(_RESERVED_PARAMETERIZATION_KEYS & set(parameterization))
        if reserved:
            raise ValueError(
                "expert_parameterization_config cannot set exporter-owned "
                f"binding keys: {reserved}."
            )

        object.__setattr__(self, "base_weight_sha256", base_hashes)
        object.__setattr__(self, "runtime_versions", runtime_versions)
        object.__setattr__(self, "training_dataset", training_dataset)
        object.__setattr__(
            self,
            "router_feature_specification",
            router_features,
        )
        object.__setattr__(
            self,
            "expert_parameterization_config",
            parameterization,
        )


@dataclass(frozen=True)
class CalibratedRouterPackageConfig:
    """Provenance for deriving one immutable router-bearing package version.

    Attributes:
        specialization_version:
            New immutable package version. It must differ from the source
            expert-only package version.
        calibration_dataset_version:
            Stable identity for the complete router fitting/calibration data
            protocol, including its fit and held-out partitions.
        ridge_regularization:
            Positive CAL-004 ridge coefficient used to produce the supplied
            uncalibrated heads.
        runtime_versions:
            Runtime used for calibration and package derivation. Required keys
            are validated again by :class:`ExpertPackageManifest`.
    """

    specialization_version: str
    calibration_dataset_version: str
    ridge_regularization: float
    runtime_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate finite fitting metadata and freeze runtime provenance."""

        _require_nonempty_string(
            self.specialization_version,
            "specialization_version",
        )
        _require_nonempty_string(
            self.calibration_dataset_version,
            "calibration_dataset_version",
        )
        if (
            isinstance(self.ridge_regularization, bool)
            or not isinstance(self.ridge_regularization, (int, float))
        ):
            raise TypeError("ridge_regularization must be a real number.")
        regularization = float(self.ridge_regularization)
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError(
                "ridge_regularization must be finite and strictly positive."
            )
        object.__setattr__(self, "ridge_regularization", regularization)
        object.__setattr__(
            self,
            "runtime_versions",
            _freeze_string_mapping(
                self.runtime_versions,
                field_name="runtime_versions",
            ),
        )


def derive_calibrated_router_package(
    source_package: ExpertPackage,
    package_directory: str | Path,
    *,
    calibrated_router: AppendOnlyRouterHeads,
    calibration_report: TemperatureThresholdCalibrationReport,
    fitting_feature_bank: RouterFeatureBank,
    fitting_utility_labels: UtilityLabelBank,
    calibration_feature_bank: RouterFeatureBank,
    calibration_utility_labels: UtilityLabelBank,
    config: CalibratedRouterPackageConfig,
) -> ExpertPackage:
    """Derive a router-bearing package without mutating expert-only content.

    The source must be a strictly validated NB-003 package with explicit
    expert payload bindings and no router. The four data banks are revalidated
    against the package feature space and expert topology. Calibrated heads
    must be frozen and have an exact one-to-one ownership match with the
    package's ``(layer_index, expert_id)`` bindings and CAL-005 report.

    Expert payload bytes are copied unchanged into a private sibling staging
    directory. The calibrated RTR-002 heads are serialized separately, and a
    new manifest records source identity, CAL-004 regularization, exact bank
    identities, disjoint CAL-005 partition metadata, thresholds, metrics, and
    calibration runtime versions. The source manifest and payloads are read
    only. Existing destinations are never replaced.

    Args:
        source_package: Strictly validated expert-only package from NB-003.
        package_directory: New derived package root.
        calibrated_router: Frozen heads returned by CAL-005.
        calibration_report: Exact report returned beside ``calibrated_router``.
        fitting_feature_bank: CAL-001 bank used for CAL-004 ridge fitting.
        fitting_utility_labels: CAL-002 targets aligned to the fitting bank.
        calibration_feature_bank: Distinct held-out CAL-001 calibration bank.
        calibration_utility_labels:
            CAL-002 targets aligned to the held-out calibration bank.
        config: New package version and reproducibility metadata.

    Returns:
        A strictly validated router-bearing :class:`ExpertPackage`.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError:
            If source state, banks, router heads, report, or versioning are
            incompatible.
        ExpertPackageExportError:
            If copying, router serialization, validation, or publication fails.

    This NB-004 artifact boundary composes the paper-derived CAL-004/CAL-005
    results but does not itself implement a paper-derived algorithm.
    """

    _validate_router_derivation_inputs(
        source_package=source_package,
        calibrated_router=calibrated_router,
        calibration_report=calibration_report,
        fitting_feature_bank=fitting_feature_bank,
        fitting_utility_labels=fitting_utility_labels,
        calibration_feature_bank=calibration_feature_bank,
        calibration_utility_labels=calibration_utility_labels,
        config=config,
    )
    destination = _validate_new_package_directory(package_directory)
    source_package.validate_live_files()
    source_manifest = source_package.manifest

    temporary_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    published = False
    try:
        package_hashes: dict[str, str] = {}
        for relative_path in source_manifest.expert_weight_files:
            source_path = source_package.package_directory.joinpath(
                *PurePosixPath(relative_path).parts
            )
            target_path = temporary_root.joinpath(
                *PurePosixPath(relative_path).parts
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            copied_digest = sha256_file(target_path)
            expected_digest = source_manifest.package_hashes[relative_path]
            if copied_digest != expected_digest:
                raise ExpertPackageExportError(
                    "Copied expert payload digest changed for "
                    f"{relative_path!r}."
                )
            package_hashes[relative_path] = copied_digest

        router_relative_path = "routers/independent-affine-heads.safetensors"
        router_path = temporary_root / router_relative_path
        router_path.parent.mkdir()
        calibrated_router.save_safetensors(router_path)
        package_hashes[router_relative_path] = sha256_file(router_path)

        calibration_metrics = {
            "source_expert_package_manifest_sha256": (
                source_package.identity.manifest_sha256
            ),
            "fitting": {
                "strategy": "independent_ridge",
                "regularization": config.ridge_regularization,
                "feature_bank_metadata_sha256": (
                    fitting_feature_bank.metadata_sha256
                ),
                "utility_label_bank_metadata_sha256": (
                    fitting_utility_labels.metadata_sha256
                ),
                "feature_bank_dataset_provenance": (
                    fitting_feature_bank.dataset_provenance
                ),
            },
            "calibration": {
                "strategy": calibration_report.strategy_name,
                "feature_bank_metadata_sha256": (
                    calibration_feature_bank.metadata_sha256
                ),
                "utility_label_bank_metadata_sha256": (
                    calibration_utility_labels.metadata_sha256
                ),
                "feature_bank_dataset_provenance": (
                    calibration_feature_bank.dataset_provenance
                ),
                "report": calibration_report.to_dict(),
            },
        }
        activation_thresholds = {
            _head_metadata_key(head.layer_index, head.expert_id): head.threshold
            for head in calibrated_router.heads
        }
        manifest = ExpertPackageManifest(
            package_format_version=source_manifest.package_format_version,
            specialization_name=source_manifest.specialization_name,
            specialization_version=config.specialization_version,
            package_structure=source_manifest.package_structure,
            base_model_repository=source_manifest.base_model_repository,
            base_model_revision=source_manifest.base_model_revision,
            base_config_sha256=source_manifest.base_config_sha256,
            base_weight_sha256=source_manifest.base_weight_sha256,
            tokenizer_repository=source_manifest.tokenizer_repository,
            tokenizer_revision=source_manifest.tokenizer_revision,
            base_model_dtype=source_manifest.base_model_dtype,
            converted_layers=source_manifest.converted_layers,
            expert_parameterization=source_manifest.expert_parameterization,
            expert_parameterization_config=(
                source_manifest.expert_parameterization_config
            ),
            expert_count=source_manifest.expert_count,
            weight_dtype=source_manifest.weight_dtype,
            expert_weight_files=source_manifest.expert_weight_files,
            runtime_versions=config.runtime_versions,
            training_dataset=source_manifest.training_dataset,
            package_hashes=package_hashes,
            router_architecture=calibrated_router.architecture,
            routing_policies_supported=("anchored_top1",),
            router_feature_specification=(
                source_manifest.router_feature_specification
            ),
            router_projection_hash=source_manifest.router_projection_hash,
            mixture_compatibility_group=(
                source_manifest.mixture_compatibility_group
            ),
            mixture_temperature=source_manifest.mixture_temperature,
            activation_thresholds=activation_thresholds,
            router_files=(router_relative_path,),
            calibration_dataset_version=(
                config.calibration_dataset_version
            ),
            calibration_metrics=calibration_metrics,
            training_parent_registry=source_manifest.training_parent_registry,
        )
        manifest.save_json(temporary_root / "manifest.json")

        ExpertPackage.from_directory(temporary_root, strict=True)
        if destination.exists():
            raise FileExistsError(
                f"Package destination already exists: {destination}"
            )
        os.rename(temporary_root, destination)
        published = True
        return ExpertPackage.from_directory(destination, strict=True)
    except FileExistsError:
        raise
    except ExpertPackageExportError:
        raise
    except (
        OSError,
        PackageValidationError,
        RouterSerializationError,
        RuntimeError,
    ) as error:
        raise ExpertPackageExportError(
            f"Cannot derive calibrated router package at {destination}: {error}"
        ) from error
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


def export_full_mlp_expert_package(
    package_directory: str | Path,
    *,
    bindings: Sequence[ExpertPackageExportBinding],
    config: ExpertPackageExportConfig,
) -> ExpertPackage:
    """Export and strictly reload one immutable full-MLP expert-only package.

    The exporter stages all expert tensors and the manifest in a private
    sibling directory. It derives a complete bijection

    ``(layer_index, expert_id) <-> safetensors payload``

    in stable layer/expert order, hashes every payload, validates strict
    directory closure, and only then publishes the package directory. Router
    tensors are intentionally absent; NB-004 owns router fitting and calibrated
    router artifact construction.

    Args:
        package_directory: New package root. Existing paths are never replaced.
        bindings: Complete rectangular layer-by-expert topology.
        config: Frozen base-checkpoint, runtime, and dataset provenance.

    Returns:
        A strictly validated local :class:`ExpertPackage`.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If bindings are incomplete, aliased, or dtype-incompatible.
        ExpertPackageExportError:
            If safe tensor serialization, validation, or publication fails.

    This workflow implements NB-003 infrastructure and does not derive an
    optimization algorithm from a paper.
    """

    if not isinstance(config, ExpertPackageExportConfig):
        raise TypeError("config must be an ExpertPackageExportConfig.")
    normalized = _normalize_bindings(bindings)
    destination = _validate_new_package_directory(package_directory)
    weight_dtype = _common_weight_dtype(normalized)

    expert_ids = tuple(sorted({binding.expert_id for binding in normalized}))
    expert_slot = {
        expert_id: index for index, expert_id in enumerate(expert_ids)
    }
    package_structure = (
        "single_expert" if len(expert_ids) == 1 else "latent_expert_set"
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    published = False
    try:
        expert_root = temporary_root / "experts"
        expert_root.mkdir()
        weight_files: list[str] = []
        payload_bindings: list[dict[str, int | str]] = []
        package_hashes: dict[str, str] = {}
        for binding in normalized:
            slot = expert_slot[binding.expert_id]
            relative_path = (
                f"experts/layer-{binding.layer_index:04d}-"
                f"expert-{slot:04d}.safetensors"
            )
            payload_path = temporary_root / relative_path
            binding.expert.save_safetensors(payload_path)
            weight_files.append(relative_path)
            payload_bindings.append(
                {
                    "layer_index": binding.layer_index,
                    "expert_id": binding.expert_id,
                    "weight_file": relative_path,
                }
            )
            package_hashes[relative_path] = sha256_file(payload_path)

        parameterization_config = dict(
            config.expert_parameterization_config
        )
        parameterization_config.update(
            {
                "payload_binding_format_version": (
                    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION
                ),
                "payload_bindings": payload_bindings,
            }
        )
        manifest = ExpertPackageManifest(
            package_format_version=PACKAGE_FORMAT_VERSION,
            specialization_name=config.specialization_name,
            specialization_version=config.specialization_version,
            package_structure=package_structure,
            base_model_repository=config.base_model_repository,
            base_model_revision=config.base_model_revision,
            base_config_sha256=config.base_config_sha256,
            base_weight_sha256=config.base_weight_sha256,
            tokenizer_repository=config.tokenizer_repository,
            tokenizer_revision=config.tokenizer_revision,
            base_model_dtype=config.base_model_dtype,
            converted_layers=sorted(
                {binding.layer_index for binding in normalized}
            ),
            expert_parameterization="full_mlp",
            expert_parameterization_config=parameterization_config,
            expert_count=len(expert_ids),
            weight_dtype=weight_dtype,
            expert_weight_files=weight_files,
            runtime_versions=config.runtime_versions,
            training_dataset=config.training_dataset,
            package_hashes=package_hashes,
            router_architecture="none",
            routing_policies_supported=(),
            router_feature_specification=(
                config.router_feature_specification
            ),
            training_parent_registry=config.training_parent_registry,
        )
        manifest.save_json(temporary_root / "manifest.json")

        # Strict validation before publication prevents a partially described
        # directory from becoming visible as a successful package.
        ExpertPackage.from_directory(temporary_root, strict=True)
        if destination.exists():
            raise FileExistsError(
                f"Package destination already exists: {destination}"
            )
        os.rename(temporary_root, destination)
        published = True
        return ExpertPackage.from_directory(destination, strict=True)
    except FileExistsError:
        raise
    except (
        ExpertSerializationError,
        PackageValidationError,
        OSError,
        RuntimeError,
    ) as error:
        raise ExpertPackageExportError(
            f"Cannot export full-MLP expert package to {destination}: {error}"
        ) from error
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


def _validate_router_derivation_inputs(
    *,
    source_package: ExpertPackage,
    calibrated_router: AppendOnlyRouterHeads,
    calibration_report: TemperatureThresholdCalibrationReport,
    fitting_feature_bank: RouterFeatureBank,
    fitting_utility_labels: UtilityLabelBank,
    calibration_feature_bank: RouterFeatureBank,
    calibration_utility_labels: UtilityLabelBank,
    config: CalibratedRouterPackageConfig,
) -> None:
    """Reject incompatible source, topology, calibration, or deployment state."""

    if not isinstance(source_package, ExpertPackage):
        raise TypeError("source_package must be an ExpertPackage.")
    if not isinstance(calibrated_router, AppendOnlyRouterHeads):
        raise TypeError(
            "calibrated_router must be an AppendOnlyRouterHeads."
        )
    if not isinstance(
        calibration_report,
        TemperatureThresholdCalibrationReport,
    ):
        raise TypeError(
            "calibration_report must be a "
            "TemperatureThresholdCalibrationReport."
        )
    if not isinstance(fitting_feature_bank, RouterFeatureBank):
        raise TypeError("fitting_feature_bank must be a RouterFeatureBank.")
    if not isinstance(fitting_utility_labels, UtilityLabelBank):
        raise TypeError(
            "fitting_utility_labels must be a UtilityLabelBank."
        )
    if not isinstance(calibration_feature_bank, RouterFeatureBank):
        raise TypeError(
            "calibration_feature_bank must be a RouterFeatureBank."
        )
    if not isinstance(calibration_utility_labels, UtilityLabelBank):
        raise TypeError(
            "calibration_utility_labels must be a UtilityLabelBank."
        )
    if not isinstance(config, CalibratedRouterPackageConfig):
        raise TypeError(
            "config must be a CalibratedRouterPackageConfig."
        )

    manifest = source_package.manifest
    if not source_package.strict_validation:
        raise ValueError(
            "source_package must have been loaded with strict validation."
        )
    if config.specialization_version == manifest.specialization_version:
        raise ValueError(
            "The derived package must use a new specialization_version."
        )
    if (
        manifest.router_architecture != "none"
        or manifest.router_files
        or manifest.routing_policies_supported
    ):
        raise ValueError(
            "source_package must be an expert-only NB-003 package without "
            "router payloads or routing policies."
        )
    if set(manifest.package_hashes) != set(manifest.expert_weight_files):
        raise ValueError(
            "source_package must contain only declared expert payload files."
        )

    bindings = source_package.payload_bindings
    expected_head_keys = tuple(binding.key for binding in bindings)
    expected_layers = tuple(manifest.converted_layers)
    expected_ids_by_layer = {
        layer_index: tuple(
            binding.expert_id
            for binding in bindings
            if binding.layer_index == layer_index
        )
        for layer_index in expected_layers
    }

    fitting_utility_labels.assert_aligned(fitting_feature_bank)
    calibration_utility_labels.assert_aligned(calibration_feature_bank)
    if (
        fitting_feature_bank.metadata_sha256
        == calibration_feature_bank.metadata_sha256
    ):
        raise ValueError(
            "Fitting and calibration feature banks must be distinct."
        )
    fitting_row_ids = set(fitting_feature_bank.row_ids.tolist())
    calibration_row_ids = set(calibration_feature_bank.row_ids.tolist())
    if fitting_row_ids & calibration_row_ids:
        raise ValueError(
            "Fitting and calibration feature-bank row IDs must be disjoint."
        )

    source_feature_sha256 = sha256_json(
        manifest.to_dict()["router_feature_specification"]
    )
    for field_name, bank in (
        ("fitting_feature_bank", fitting_feature_bank),
        ("calibration_feature_bank", calibration_feature_bank),
    ):
        if sha256_json(bank.feature_specification) != source_feature_sha256:
            raise ValueError(
                f"{field_name} feature specification does not match the "
                "source package."
            )
        if bank.layer_indices != expected_layers:
            raise ValueError(
                f"{field_name} layers do not match the source package: "
                f"{bank.layer_indices} != {expected_layers}."
            )

    for field_name, labels in (
        ("fitting_utility_labels", fitting_utility_labels),
        ("calibration_utility_labels", calibration_utility_labels),
    ):
        for layer_index in expected_layers:
            actual_ids = labels.expert_ids_for_layer(layer_index)
            if actual_ids != expected_ids_by_layer[layer_index]:
                raise ValueError(
                    f"{field_name} expert order does not match source "
                    f"bindings at layer {layer_index}: {actual_ids} != "
                    f"{expected_ids_by_layer[layer_index]}."
                )

    if not calibrated_router.heads:
        raise ValueError("calibrated_router must not be empty.")
    if calibrated_router.head_keys != expected_head_keys:
        raise ValueError(
            "Calibrated router heads must exactly match source payload "
            f"bindings: {calibrated_router.head_keys} != {expected_head_keys}."
        )
    output_size = manifest.router_feature_specification.get("output_size")
    if calibrated_router.feature_size != output_size:
        raise ValueError(
            "Calibrated router feature size does not match the source "
            f"package: {calibrated_router.feature_size} != {output_size}."
        )
    if any(
        parameter.requires_grad
        for parameter in calibrated_router.parameters()
    ):
        raise ValueError(
            "Calibrated router heads must be frozen before package derivation."
        )
    if any(
        not torch.isfinite(parameter.detach()).all().item()
        for parameter in calibrated_router.parameters()
    ):
        raise ValueError("Calibrated router parameters must be finite.")
    router_dtypes = {
        str(head.linear.weight.dtype).removeprefix("torch.")
        for head in calibrated_router.heads
    }
    if router_dtypes != {manifest.weight_dtype}:
        raise ValueError(
            "Calibrated router dtype must match package expert/hidden dtype: "
            f"{sorted(router_dtypes)} != {[manifest.weight_dtype]}."
        )

    report_keys = tuple(
        (report.layer_index, report.expert_id)
        for report in calibration_report.head_reports
    )
    if report_keys != calibrated_router.head_keys:
        raise ValueError(
            "Calibration report head order does not match the calibrated "
            f"router: {report_keys} != {calibrated_router.head_keys}."
        )
    for head, report in zip(
        calibrated_router.heads,
        calibration_report.head_reports,
        strict=True,
    ):
        if (
            head.temperature != report.temperature
            or head.threshold != report.threshold
        ):
            raise ValueError(
                "Calibration report metadata does not match router head "
                f"{head.key!r}."
            )
    report_temperature_ids = set(calibration_report.temperature_row_ids)
    report_threshold_ids = set(calibration_report.threshold_row_ids)
    if report_temperature_ids & report_threshold_ids:
        raise ValueError(
            "Calibration report temperature and threshold rows overlap."
        )
    if report_temperature_ids | report_threshold_ids != calibration_row_ids:
        raise ValueError(
            "Calibration report row partitions must exactly cover the "
            "calibration feature bank."
        )


def _head_metadata_key(layer_index: int, expert_id: str) -> str:
    """Return the stable manifest key for one layer/expert scalar."""

    return f"layer={layer_index};expert={expert_id}"


def _normalize_bindings(
    value: Sequence[ExpertPackageExportBinding],
) -> tuple[ExpertPackageExportBinding, ...]:
    """Validate rectangular topology, unique objects, and independent storage."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "bindings must be a sequence of ExpertPackageExportBinding."
        )
    if not value or not all(
        isinstance(binding, ExpertPackageExportBinding) for binding in value
    ):
        raise TypeError(
            "bindings must be a non-empty sequence of "
            "ExpertPackageExportBinding."
        )
    normalized = tuple(
        sorted(value, key=lambda binding: (binding.layer_index, binding.expert_id))
    )
    keys = [(binding.layer_index, binding.expert_id) for binding in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError("bindings must not repeat a layer/expert slot.")

    layers = sorted({binding.layer_index for binding in normalized})
    expert_ids = sorted({binding.expert_id for binding in normalized})
    expected = {(layer, expert_id) for layer in layers for expert_id in expert_ids}
    if set(keys) != expected:
        raise ValueError(
            "bindings must form a complete layer-by-expert rectangle."
        )

    expert_object_ids: set[int] = set()
    occupied_storage: set[tuple[str, int | None, int]] = set()
    for binding in normalized:
        if id(binding.expert) in expert_object_ids:
            raise ValueError(
                "Every package layer/expert slot must own an independent "
                "expert object."
            )
        expert_object_ids.add(id(binding.expert))
        binding_storage = {
            _tensor_storage_key(tensor)
            for tensor in binding.expert.state_dict().values()
            if tensor.numel() > 0
        }
        if occupied_storage & binding_storage:
            raise ValueError(
                "Expert tensors must not share storage across package slots."
            )
        occupied_storage.update(binding_storage)
    return normalized


def _common_weight_dtype(
    bindings: Sequence[ExpertPackageExportBinding],
) -> str:
    """Return one manifest dtype after checking every floating expert tensor."""

    dtypes: set[str] = set()
    for binding in bindings:
        for tensor in binding.expert.state_dict().values():
            if tensor.is_floating_point():
                dtypes.add(str(tensor.dtype).removeprefix("torch."))
    if not dtypes:
        raise ValueError("Full-MLP experts must contain floating tensors.")
    if len(dtypes) != 1:
        raise ValueError(
            "All exported full-MLP floating tensors must share one dtype: "
            f"{sorted(dtypes)}."
        )
    return next(iter(dtypes))


def _tensor_storage_key(tensor: Tensor) -> tuple[str, int | None, int]:
    """Identify tensor storage while preserving device identity."""

    return (
        tensor.device.type,
        tensor.device.index,
        tensor.untyped_storage().data_ptr(),
    )


def _validate_new_package_directory(value: str | Path) -> Path:
    """Resolve a new package root without creating or replacing it."""

    if not isinstance(value, (str, Path)):
        raise TypeError("package_directory must be a path string or Path.")
    destination = Path(value)
    if destination.name in ("", ".", ".."):
        raise ValueError("package_directory must name one new directory.")
    if destination.is_symlink():
        raise ValueError("package_directory must not be a symbolic link.")
    if destination.exists():
        raise FileExistsError(
            f"Package destination already exists: {destination}"
        )
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Package destination parent does not exist: {parent}"
        )
    if parent.is_symlink():
        raise ValueError(
            "Package destination parent must not be a symbolic link."
        )
    return parent.resolve() / destination.name


def _freeze_string_mapping(
    value: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    """Copy a non-empty string-to-string mapping into immutable storage."""

    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{field_name} must be a non-empty mapping.")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _require_nonempty_string(key, f"{field_name} key")
        _require_nonempty_string(item, f"{field_name}[{key!r}]")
        normalized[key] = item
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_json_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate JSON compatibility and recursively freeze one mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    try:
        serialized = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
        copied = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain finite JSON data.") from error
    return _deep_freeze_json_mapping(copied)


def _deep_freeze_json_mapping(value: dict[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze a JSON object after round-trip normalization."""

    return MappingProxyType(
        {
            key: _deep_freeze_json(item)
            for key, item in sorted(value.items())
        }
    )


def _deep_freeze_json(value: Any) -> Any:
    """Recursively freeze JSON arrays and objects."""

    if isinstance(value, dict):
        return _deep_freeze_json_mapping(value)
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _require_nonempty_string(value: Any, field_name: str) -> str:
    """Return a normalized metadata string or raise a field-specific error."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace."
        )
    return value
