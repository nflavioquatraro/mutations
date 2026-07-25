"""Versioned, JSON-only manifest schema for immutable expert packages."""

from __future__ import annotations

import hmac
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from safetensors import safe_open

from anchored_moe.utils.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)


PACKAGE_FORMAT_VERSION = 1
DEFAULT_MANIFEST_FILENAME = "manifest.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_STRUCTURES = frozenset({"single_expert", "latent_expert_set"})
_ROUTER_FILE_SUFFIXES = frozenset({".json", ".safetensors"})
_REQUIRED_RUNTIME_VERSIONS = frozenset(
    {
        "accelerate",
        "kernels",
        "peft",
        "python",
        "safetensors",
        "torch",
        "transformers",
    }
)
_REQUIRED_DATASET_FIELDS = frozenset(
    {"dataset_id", "filters", "revision", "split"}
)


class PackageValidationError(ValueError):
    """Raised when a package manifest or payload violates its contract."""


@dataclass(frozen=True)
class PackageValidationReport:
    """Auditable result of validating one package directory.

    Attributes:
        manifest_sha256: Digest of the canonical manifest representation.
        verified_files: Stable, sorted payload paths whose hashes matched.
        verified_bytes: Sum of the verified payload file sizes.
    """

    manifest_sha256: str
    verified_files: tuple[str, ...]
    verified_bytes: int


@dataclass(frozen=True)
class ExpertPackageManifest:
    """Immutable schema describing one independently distributable expert package.

    All payload paths use normalized POSIX syntax relative to the package root.
    ``package_hashes`` covers every payload file but deliberately excludes
    ``manifest.json`` because a manifest cannot recursively contain its own
    digest. ``manifest_sha256`` provides the separate manifest identity.

    Tensor payloads declared in ``expert_weight_files`` must use safetensors.
    Router payloads may use safetensors for tensors or JSON for non-tensor
    configuration. Pickle-backed formats are not accepted.
    """

    package_format_version: int
    specialization_name: str
    specialization_version: str
    package_structure: str
    base_model_repository: str
    base_model_revision: str
    base_config_sha256: str
    base_weight_sha256: Mapping[str, str]
    tokenizer_repository: str
    tokenizer_revision: str
    base_model_dtype: str
    converted_layers: Sequence[int]
    expert_parameterization: str
    expert_parameterization_config: Mapping[str, Any]
    expert_count: int
    weight_dtype: str
    expert_weight_files: Sequence[str]
    runtime_versions: Mapping[str, str]
    training_dataset: Mapping[str, Any]
    package_hashes: Mapping[str, str]
    router_architecture: str = "none"
    routing_policies_supported: Sequence[str] = ()
    router_feature_specification: Mapping[str, Any] = field(default_factory=dict)
    router_projection_hash: str | None = None
    mixture_compatibility_group: str | None = None
    mixture_temperature: float | None = None
    activation_thresholds: Mapping[str, float] = field(default_factory=dict)
    router_files: Sequence[str] = ()
    calibration_dataset_version: str | None = None
    calibration_metrics: Mapping[str, Any] = field(default_factory=dict)
    training_parent_registry: str | None = None

    def __post_init__(self) -> None:
        """Normalize collections and validate every cross-field invariant."""

        self._validate_scalar_fields()

        converted_layers = _normalize_layer_indices(self.converted_layers)
        expert_weight_files = _normalize_paths(
            self.expert_weight_files,
            field_name="expert_weight_files",
        )
        router_files = _normalize_paths(
            self.router_files,
            field_name="router_files",
        )
        routing_policies = _normalize_unique_strings(
            self.routing_policies_supported,
            field_name="routing_policies_supported",
        )

        if any(not path.endswith(".safetensors") for path in expert_weight_files):
            raise PackageValidationError(
                "Every expert weight file must use the .safetensors extension."
            )
        if not expert_weight_files:
            raise PackageValidationError(
                "expert_weight_files must contain at least one safetensors payload."
            )
        invalid_router_files = [
            path
            for path in router_files
            if PurePosixPath(path).suffix not in _ROUTER_FILE_SUFFIXES
        ]
        if invalid_router_files:
            raise PackageValidationError(
                "Router files must use .json or .safetensors: "
                f"{invalid_router_files}"
            )

        overlap = sorted(set(expert_weight_files) & set(router_files))
        if overlap:
            raise PackageValidationError(
                f"Files cannot be both expert and router payloads: {overlap}"
            )

        base_weight_sha256 = _normalize_hash_mapping(
            self.base_weight_sha256,
            field_name="base_weight_sha256",
            require_nonempty=True,
        )
        package_hashes = _normalize_hash_mapping(
            self.package_hashes,
            field_name="package_hashes",
            require_nonempty=True,
        )
        if DEFAULT_MANIFEST_FILENAME in package_hashes:
            raise PackageValidationError(
                f"{DEFAULT_MANIFEST_FILENAME!r} cannot hash itself."
            )

        declared_payloads = set(expert_weight_files) | set(router_files)
        unhashed = sorted(declared_payloads - set(package_hashes))
        if unhashed:
            raise PackageValidationError(
                f"Declared payload files are missing SHA-256 digests: {unhashed}"
            )

        runtime_versions = _normalize_string_mapping(
            self.runtime_versions,
            field_name="runtime_versions",
        )
        missing_runtime_versions = sorted(
            _REQUIRED_RUNTIME_VERSIONS - set(runtime_versions)
        )
        if missing_runtime_versions:
            raise PackageValidationError(
                "runtime_versions is missing required keys: "
                f"{missing_runtime_versions}"
            )

        training_dataset = _freeze_json_mapping(
            self.training_dataset,
            field_name="training_dataset",
        )
        missing_dataset_fields = sorted(
            _REQUIRED_DATASET_FIELDS - set(training_dataset)
        )
        if missing_dataset_fields:
            raise PackageValidationError(
                "training_dataset is missing required keys: "
                f"{missing_dataset_fields}"
            )
        for field_name in ("dataset_id", "revision", "split"):
            _require_nonempty_string(
                training_dataset[field_name],
                f"training_dataset.{field_name}",
            )

        parameterization_config = _freeze_json_mapping(
            self.expert_parameterization_config,
            field_name="expert_parameterization_config",
        )
        router_features = _freeze_json_mapping(
            self.router_feature_specification,
            field_name="router_feature_specification",
        )
        calibration_metrics = _freeze_json_mapping(
            self.calibration_metrics,
            field_name="calibration_metrics",
        )
        activation_thresholds = _normalize_float_mapping(
            self.activation_thresholds,
            field_name="activation_thresholds",
        )

        if self.router_architecture == "none" and router_files:
            raise PackageValidationError(
                "router_files must be empty when router_architecture is 'none'."
            )
        if self.router_architecture != "none" and not router_files:
            raise PackageValidationError(
                "A configured router architecture requires at least one router file."
            )
        if self.router_projection_hash is not None:
            _validate_sha256(
                self.router_projection_hash,
                field_name="router_projection_hash",
            )

        object.__setattr__(self, "converted_layers", converted_layers)
        object.__setattr__(self, "expert_weight_files", expert_weight_files)
        object.__setattr__(self, "router_files", router_files)
        object.__setattr__(self, "routing_policies_supported", routing_policies)
        object.__setattr__(self, "base_weight_sha256", base_weight_sha256)
        object.__setattr__(self, "package_hashes", package_hashes)
        object.__setattr__(self, "runtime_versions", runtime_versions)
        object.__setattr__(self, "training_dataset", training_dataset)
        object.__setattr__(
            self,
            "expert_parameterization_config",
            parameterization_config,
        )
        object.__setattr__(
            self,
            "router_feature_specification",
            router_features,
        )
        object.__setattr__(self, "calibration_metrics", calibration_metrics)
        object.__setattr__(self, "activation_thresholds", activation_thresholds)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExpertPackageManifest:
        """Construct a manifest while rejecting missing and unknown fields."""

        if not isinstance(value, Mapping):
            raise PackageValidationError("Manifest root must be a JSON object.")

        field_definitions = {field.name: field for field in fields(cls)}
        unknown = sorted(set(value) - set(field_definitions))
        if unknown:
            raise PackageValidationError(f"Unknown manifest fields: {unknown}")

        required = {
            name
            for name, field_definition in field_definitions.items()
            if field_definition.default is MISSING
            and field_definition.default_factory is MISSING
        }
        missing = sorted(required - set(value))
        if missing:
            raise PackageValidationError(f"Missing required manifest fields: {missing}")

        try:
            return cls(**dict(value))
        except PackageValidationError:
            raise
        except (TypeError, ValueError) as error:
            raise PackageValidationError(f"Invalid manifest: {error}") from error

    @classmethod
    def from_json(cls, path: str | Path) -> ExpertPackageManifest:
        """Load a UTF-8 JSON manifest without executing package content."""

        manifest_path = Path(path)
        try:
            raw_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PackageValidationError(
                f"Cannot read JSON manifest {manifest_path}: {error}"
            ) from error
        return cls.from_dict(raw_value)

    @property
    def manifest_sha256(self) -> str:
        """Return the content identity of the canonical manifest."""

        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete schema as an independently mutable JSON object."""

        return {
            "package_format_version": self.package_format_version,
            "specialization_name": self.specialization_name,
            "specialization_version": self.specialization_version,
            "package_structure": self.package_structure,
            "base_model_repository": self.base_model_repository,
            "base_model_revision": self.base_model_revision,
            "base_config_sha256": self.base_config_sha256,
            "base_weight_sha256": dict(self.base_weight_sha256),
            "tokenizer_repository": self.tokenizer_repository,
            "tokenizer_revision": self.tokenizer_revision,
            "base_model_dtype": self.base_model_dtype,
            "converted_layers": list(self.converted_layers),
            "expert_parameterization": self.expert_parameterization,
            "expert_parameterization_config": _thaw_json(
                self.expert_parameterization_config
            ),
            "expert_count": self.expert_count,
            "router_architecture": self.router_architecture,
            "routing_policies_supported": list(self.routing_policies_supported),
            "router_feature_specification": _thaw_json(
                self.router_feature_specification
            ),
            "router_projection_hash": self.router_projection_hash,
            "mixture_compatibility_group": self.mixture_compatibility_group,
            "mixture_temperature": self.mixture_temperature,
            "activation_thresholds": dict(self.activation_thresholds),
            "weight_dtype": self.weight_dtype,
            "expert_weight_files": list(self.expert_weight_files),
            "router_files": list(self.router_files),
            "calibration_dataset_version": self.calibration_dataset_version,
            "calibration_metrics": _thaw_json(self.calibration_metrics),
            "training_parent_registry": self.training_parent_registry,
            "runtime_versions": dict(self.runtime_versions),
            "training_dataset": _thaw_json(self.training_dataset),
            "package_hashes": dict(self.package_hashes),
        }

    def to_json_bytes(self) -> bytes:
        """Return the canonical JSON representation followed by one newline."""

        return canonical_json_bytes(self.to_dict()) + b"\n"

    def save_json(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically write a canonical UTF-8 manifest.

        Existing files are protected unless ``overwrite=True`` is explicit.
        The temporary file is created beside the destination so ``os.replace``
        remains atomic on ordinary local filesystems.
        """

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {destination}")
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Manifest parent directory does not exist: {destination.parent}"
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(self.to_json_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def validate_package_files(
        self,
        package_directory: str | Path,
        *,
        strict: bool = True,
        manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    ) -> PackageValidationReport:
        """Verify payload paths, hashes, formats, and optional directory closure.

        When ``strict`` is true, every regular file other than the manifest must
        appear in ``package_hashes`` and symbolic links are rejected. This makes
        installation inputs auditable and prevents untracked payloads.
        """

        package_root = Path(package_directory)
        if not package_root.is_dir():
            raise PackageValidationError(
                f"Package directory does not exist: {package_root}"
            )
        package_root = package_root.resolve()
        verified_bytes = 0

        for relative_path, expected_digest in self.package_hashes.items():
            candidate = package_root.joinpath(*PurePosixPath(relative_path).parts)
            if candidate.is_symlink():
                raise PackageValidationError(
                    f"Symbolic links are not allowed in packages: {relative_path}"
                )
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except FileNotFoundError as error:
                raise PackageValidationError(
                    f"Package payload is missing: {relative_path}"
                ) from error
            if not resolved_candidate.is_relative_to(package_root):
                raise PackageValidationError(
                    f"Package payload escapes its root: {relative_path}"
                )
            if not resolved_candidate.is_file():
                raise PackageValidationError(
                    f"Package payload is not a regular file: {relative_path}"
                )

            actual_digest = sha256_file(resolved_candidate)
            if not hmac.compare_digest(actual_digest, expected_digest):
                raise PackageValidationError(
                    f"SHA-256 mismatch for package payload: {relative_path}"
                )
            verified_bytes += resolved_candidate.stat().st_size

        if strict:
            actual_files: set[str] = set()
            for candidate in package_root.rglob("*"):
                if candidate.is_symlink():
                    relative = candidate.relative_to(package_root).as_posix()
                    raise PackageValidationError(
                        f"Symbolic links are not allowed in packages: {relative}"
                    )
                if candidate.is_file():
                    relative = candidate.relative_to(package_root).as_posix()
                    if relative != manifest_filename:
                        actual_files.add(relative)

            untracked = sorted(actual_files - set(self.package_hashes))
            missing = sorted(set(self.package_hashes) - actual_files)
            if untracked:
                raise PackageValidationError(
                    f"Package contains untracked payload files: {untracked}"
                )
            if missing:
                raise PackageValidationError(
                    f"Package is missing tracked payload files: {missing}"
                )

        self._validate_payload_formats(package_root)
        return PackageValidationReport(
            manifest_sha256=self.manifest_sha256,
            verified_files=tuple(self.package_hashes),
            verified_bytes=verified_bytes,
        )

    def _validate_scalar_fields(self) -> None:
        if self.package_format_version != PACKAGE_FORMAT_VERSION:
            raise PackageValidationError(
                f"Unsupported package_format_version={self.package_format_version!r}; "
                f"supported version is {PACKAGE_FORMAT_VERSION}."
            )

        for field_name in (
            "specialization_name",
            "specialization_version",
            "base_model_repository",
            "base_model_revision",
            "tokenizer_repository",
            "tokenizer_revision",
            "base_model_dtype",
            "expert_parameterization",
            "weight_dtype",
            "router_architecture",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)

        if self.package_structure not in _PACKAGE_STRUCTURES:
            raise PackageValidationError(
                f"package_structure must be one of {sorted(_PACKAGE_STRUCTURES)}."
            )
        if isinstance(self.expert_count, bool) or not isinstance(self.expert_count, int):
            raise PackageValidationError("expert_count must be an integer.")
        if self.expert_count <= 0:
            raise PackageValidationError("expert_count must be greater than zero.")
        if self.package_structure == "single_expert" and self.expert_count != 1:
            raise PackageValidationError(
                "single_expert packages must declare expert_count=1."
            )
        if self.package_structure == "latent_expert_set" and self.expert_count < 2:
            raise PackageValidationError(
                "latent_expert_set packages must declare at least two experts."
            )

        _validate_sha256(self.base_config_sha256, field_name="base_config_sha256")
        if self.mixture_temperature is not None:
            if (
                isinstance(self.mixture_temperature, bool)
                or not isinstance(self.mixture_temperature, (int, float))
                or not math.isfinite(self.mixture_temperature)
                or self.mixture_temperature <= 0
            ):
                raise PackageValidationError(
                    "mixture_temperature must be a finite positive number."
                )

        for optional_name in (
            "router_projection_hash",
            "mixture_compatibility_group",
            "calibration_dataset_version",
            "training_parent_registry",
        ):
            optional_value = getattr(self, optional_name)
            if optional_value is not None:
                _require_nonempty_string(optional_value, optional_name)

    def _validate_payload_formats(self, package_root: Path) -> None:
        tensor_files = set(self.expert_weight_files)
        tensor_files.update(
            path for path in self.router_files if path.endswith(".safetensors")
        )
        for relative_path in sorted(tensor_files):
            candidate = package_root.joinpath(*PurePosixPath(relative_path).parts)
            try:
                # Opening and enumerating keys validates the safetensors header
                # without materializing tensor payloads.
                with safe_open(candidate, framework="pt", device="cpu") as handle:
                    tuple(handle.keys())
            except Exception as error:
                raise PackageValidationError(
                    f"Invalid safetensors payload: {relative_path}"
                ) from error

        for relative_path in sorted(
            path for path in self.router_files if path.endswith(".json")
        ):
            candidate = package_root.joinpath(*PurePosixPath(relative_path).parts)
            try:
                json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PackageValidationError(
                    f"Invalid router JSON payload: {relative_path}"
                ) from error


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PackageValidationError(
            f"{field_name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _validate_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PackageValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _normalize_layer_indices(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PackageValidationError("converted_layers must be a sequence of integers.")
    normalized: list[int] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PackageValidationError(
                "converted_layers must contain only non-negative integers."
            )
        normalized.append(index)
    if not normalized:
        raise PackageValidationError("converted_layers must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise PackageValidationError("converted_layers must not contain duplicates.")
    return tuple(sorted(normalized))


def _normalize_paths(value: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PackageValidationError(f"{field_name} must be a sequence of paths.")
    normalized = tuple(
        _normalize_package_path(path, field_name=field_name) for path in value
    )
    if len(set(normalized)) != len(normalized):
        raise PackageValidationError(f"{field_name} must not contain duplicates.")
    return tuple(sorted(normalized))


def _normalize_package_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PackageValidationError(f"{field_name} entries must be non-empty strings.")
    if "\\" in value:
        raise PackageValidationError(
            f"{field_name} paths must use POSIX separators: {value!r}"
        )

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or any(ord(character) < 32 for character in value)
    ):
        raise PackageValidationError(
            f"{field_name} contains an unsafe or non-canonical path: {value!r}"
        )
    return value


def _normalize_hash_mapping(
    value: Mapping[str, str],
    *,
    field_name: str,
    require_nonempty: bool,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{field_name} must be a mapping.")
    if require_nonempty and not value:
        raise PackageValidationError(f"{field_name} must not be empty.")

    normalized: dict[str, str] = {}
    for path, digest in value.items():
        normalized_path = _normalize_package_path(path, field_name=field_name)
        if normalized_path in normalized:
            raise PackageValidationError(f"{field_name} contains duplicate paths.")
        normalized[normalized_path] = _validate_sha256(
            digest,
            field_name=f"{field_name}[{normalized_path!r}]",
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_unique_strings(
    value: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PackageValidationError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(
        _require_nonempty_string(item, field_name) for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise PackageValidationError(f"{field_name} must not contain duplicates.")
    return tuple(sorted(normalized))


def _normalize_string_mapping(
    value: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{field_name} must be a mapping.")
    normalized = {
        _require_nonempty_string(key, field_name): _require_nonempty_string(
            item,
            f"{field_name}.{key}",
        )
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_float_mapping(
    value: Mapping[str, float],
    *,
    field_name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{field_name} must be a mapping.")
    normalized: dict[str, float] = {}
    for key, item in value.items():
        normalized_key = _require_nonempty_string(key, field_name)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
        ):
            raise PackageValidationError(
                f"{field_name}.{normalized_key} must be a finite number."
            )
        normalized[normalized_key] = float(item)
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_json_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{field_name} must be a mapping.")
    return _freeze_json(value, field_name=field_name)


def _freeze_json(value: Any, *, field_name: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _require_nonempty_string(key, field_name)
            normalized[normalized_key] = _freeze_json(
                item,
                field_name=f"{field_name}.{normalized_key}",
            )
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, field_name=f"{field_name}[]") for item in value
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise PackageValidationError(
        f"{field_name} contains a non-JSON or non-finite value: "
        f"{type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
