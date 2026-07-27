"""Versioned expert-payload ownership bindings for package materialization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from anchored_moe.registry.manifest import (
    ExpertPackageManifest,
    PackageValidationError,
)


EXPERT_PAYLOAD_BINDING_FORMAT_VERSION = 1
_FORMAT_KEY = "payload_binding_format_version"
_BINDINGS_KEY = "payload_bindings"
_BINDING_FIELDS = frozenset({"expert_id", "layer_index", "weight_file"})


@dataclass(frozen=True, order=True)
class ExpertPayloadBinding:
    """Own one full-MLP payload at a stable layer/expert runtime slot.

    Attributes:
        layer_index: Zero-based converted decoder layer.
        expert_id: Stable package-provided expert identifier.
        weight_file: Normalized package-relative safetensors payload path.
    """

    layer_index: int
    expert_id: str
    weight_file: str

    def __post_init__(self) -> None:
        """Validate public construction independently of a manifest."""

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
        if not _is_normalized_package_path(self.weight_file):
            raise ValueError(
                "weight_file must be a normalized package-relative POSIX path."
            )
        if not self.weight_file.endswith(".safetensors"):
            raise ValueError("weight_file must use the .safetensors extension.")

    @property
    def key(self) -> tuple[int, str]:
        """Return the stable ``(layer_index, expert_id)`` ownership key."""

        return self.layer_index, self.expert_id

    def to_dict(self) -> dict[str, int | str]:
        """Return the JSON representation stored in package configuration."""

        return {
            "layer_index": self.layer_index,
            "expert_id": self.expert_id,
            "weight_file": self.weight_file,
        }


def parse_expert_payload_bindings(
    manifest: ExpertPackageManifest,
    *,
    require: bool = False,
) -> tuple[ExpertPayloadBinding, ...]:
    """Parse and cross-validate version-1 full-MLP payload bindings.

    Package format version 1 predates runtime ownership metadata. Legacy
    manifests therefore remain valid for hashing and registry lifecycle, but
    callers that materialize tensors pass ``require=True`` and receive an
    explicit error when bindings are absent.

    The parser requires one payload per ``(converted layer, expert ID)`` slot,
    the same expert-ID set at every converted layer, and exact one-to-one
    coverage of ``expert_weight_files``. Filenames never imply ownership.

    Args:
        manifest: Validated immutable package manifest.
        require: Whether absence of the binding block is an error.

    Returns:
        Bindings sorted by layer and expert identifier.

    Raises:
        TypeError: If ``manifest`` is not an :class:`ExpertPackageManifest`.
        PackageValidationError: If metadata is absent when required or invalid.
    """

    if not isinstance(manifest, ExpertPackageManifest):
        raise TypeError("manifest must be an ExpertPackageManifest.")

    config = manifest.expert_parameterization_config
    has_format = _FORMAT_KEY in config
    has_bindings = _BINDINGS_KEY in config
    if not has_format and not has_bindings:
        if require:
            raise PackageValidationError(
                "Package materialization requires explicit payload bindings."
            )
        return ()
    if has_format != has_bindings:
        raise PackageValidationError(
            f"{_FORMAT_KEY} and {_BINDINGS_KEY} must be declared together."
        )

    format_version = config[_FORMAT_KEY]
    if (
        isinstance(format_version, bool)
        or format_version != EXPERT_PAYLOAD_BINDING_FORMAT_VERSION
    ):
        raise PackageValidationError(
            "Unsupported payload_binding_format_version="
            f"{format_version!r}; supported version is "
            f"{EXPERT_PAYLOAD_BINDING_FORMAT_VERSION}."
        )

    raw_bindings = config[_BINDINGS_KEY]
    if (
        isinstance(raw_bindings, (str, bytes))
        or not isinstance(raw_bindings, Sequence)
        or not raw_bindings
    ):
        raise PackageValidationError(
            "payload_bindings must be a non-empty sequence of objects."
        )

    bindings = tuple(
        _binding_from_mapping(value, index=index)
        for index, value in enumerate(raw_bindings)
    )
    keys = [binding.key for binding in bindings]
    duplicate_keys = sorted(
        key for key, count in Counter(keys).items() if count > 1
    )
    if duplicate_keys:
        raise PackageValidationError(
            f"payload_bindings repeats layer/expert slots: {duplicate_keys}"
        )

    paths = [binding.weight_file for binding in bindings]
    duplicate_paths = sorted(
        path for path, count in Counter(paths).items() if count > 1
    )
    if duplicate_paths:
        raise PackageValidationError(
            "Each full-MLP payload must own exactly one runtime slot; repeated "
            f"files: {duplicate_paths}"
        )

    converted_layers = tuple(manifest.converted_layers)
    converted_layer_set = set(converted_layers)
    unexpected_layers = sorted(
        {binding.layer_index for binding in bindings} - converted_layer_set
    )
    if unexpected_layers:
        raise PackageValidationError(
            "payload_bindings contains unconverted layers: "
            f"{unexpected_layers}"
        )

    declared_files = set(manifest.expert_weight_files)
    bound_files = set(paths)
    missing_files = sorted(declared_files - bound_files)
    undeclared_files = sorted(bound_files - declared_files)
    if missing_files or undeclared_files:
        raise PackageValidationError(
            "payload_bindings must exactly cover expert_weight_files: "
            f"missing={missing_files}, undeclared={undeclared_files}"
        )

    expert_ids_by_layer = {
        layer_index: frozenset(
            binding.expert_id
            for binding in bindings
            if binding.layer_index == layer_index
        )
        for layer_index in converted_layers
    }
    wrong_counts = {
        layer_index: len(expert_ids)
        for layer_index, expert_ids in expert_ids_by_layer.items()
        if len(expert_ids) != manifest.expert_count
    }
    if wrong_counts:
        raise PackageValidationError(
            "Every converted layer must bind expert_count distinct experts: "
            f"expected={manifest.expert_count}, actual={wrong_counts}"
        )

    first_ids = expert_ids_by_layer[converted_layers[0]]
    inconsistent_layers = [
        layer_index
        for layer_index in converted_layers[1:]
        if expert_ids_by_layer[layer_index] != first_ids
    ]
    if inconsistent_layers:
        raise PackageValidationError(
            "Every converted layer must use the same package expert IDs; "
            f"inconsistent layers: {inconsistent_layers}"
        )

    return tuple(sorted(bindings))


def _binding_from_mapping(
    value: Any,
    *,
    index: int,
) -> ExpertPayloadBinding:
    """Construct one binding while converting errors to package validation."""

    if not isinstance(value, Mapping):
        raise PackageValidationError(
            f"payload_bindings[{index}] must be an object."
        )
    unknown = sorted(set(value) - _BINDING_FIELDS)
    missing = sorted(_BINDING_FIELDS - set(value))
    if unknown or missing:
        raise PackageValidationError(
            f"payload_bindings[{index}] fields are invalid: "
            f"missing={missing}, unknown={unknown}"
        )
    try:
        return ExpertPayloadBinding(
            layer_index=value["layer_index"],
            expert_id=value["expert_id"],
            weight_file=value["weight_file"],
        )
    except (TypeError, ValueError) as error:
        raise PackageValidationError(
            f"Invalid payload_bindings[{index}]: {error}"
        ) from error


def _is_normalized_package_path(value: Any) -> bool:
    """Return whether a value follows the manifest's safe path contract."""

    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or any(ord(character) < 32 for character in value)
    )
