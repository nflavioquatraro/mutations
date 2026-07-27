"""Versioned router feature caches and immutable reference banks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from anchored_moe.utils.hashing import canonical_json_bytes, sha256_json


ROUTER_FEATURE_BANK_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = "anchored_moe.router_feature_bank.v1"
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"
_REQUIRED_DATASET_FIELDS: Final = frozenset(
    {"dataset_id", "filters", "revision", "split"}
)


class FeatureBankSerializationError(ValueError):
    """Raised when a router feature-bank payload is unsafe or incompatible."""


class RouterFeatureBank:
    """Immutable, row-aligned router features for calibration experiments.

    A bank stores one two-dimensional feature tensor
    ``X_l in R^(N x D)`` for every declared decoder layer ``l``. The same
    integer row identifiers align examples across layers without storing
    prompts, token text, or hidden content outside the feature tensors.

    Construction validates, detaches, copies, and moves all tensors to
    contiguous CPU storage. Accessors return defensive copies, so callers
    cannot mutate the bank after construction.

    Args:
        features_by_layer:
            Mapping from non-negative decoder-layer indices to floating-point
            tensors with shape ``[row_count, feature_size]``.
        feature_specification:
            JSON-compatible router feature-space metadata. ``output_size`` is
            required and must equal every tensor's final dimension.
        dataset_provenance:
            JSON-compatible dataset identity containing ``dataset_id``,
            ``revision``, ``split``, and ``filters``. Raw examples should not
            be placed in this metadata.
        seed:
            Non-negative seed governing deterministic collection order.
        row_ids:
            Optional unique, non-negative ``torch.long`` tensor with shape
            ``[row_count]``. The default is ``torch.arange(row_count)``.

    This class implements CAL-001 infrastructure. It does not collect model
    hooks, construct labels, fit router heads, choose thresholds, or shard
    datasets.
    """

    format_version: ClassVar[int] = ROUTER_FEATURE_BANK_FORMAT_VERSION
    serialization_format: ClassVar[str] = _SERIALIZATION_FORMAT

    def __init__(
        self,
        features_by_layer: Mapping[int, Tensor],
        *,
        feature_specification: Mapping[str, Any],
        dataset_provenance: Mapping[str, Any],
        seed: int = 0,
        row_ids: Tensor | None = None,
    ) -> None:
        normalized_specification = _normalize_feature_specification(
            feature_specification
        )
        normalized_provenance = _normalize_dataset_provenance(dataset_provenance)
        normalized_seed = _validate_non_negative_integer(seed, field_name="seed")
        normalized_features = _normalize_features_by_layer(
            features_by_layer,
            feature_size=normalized_specification["output_size"],
        )

        first_features = next(iter(normalized_features.values()))
        row_count, feature_size = first_features.shape
        normalized_row_ids = _normalize_row_ids(row_ids, row_count=row_count)

        self._features_by_layer = normalized_features
        self._row_ids = normalized_row_ids
        # Canonical JSON strings make nested metadata immutable without
        # exposing internal mutable mappings through public accessors.
        self._feature_specification_json = canonical_json_bytes(
            normalized_specification
        ).decode("utf-8")
        self._dataset_provenance_json = canonical_json_bytes(
            normalized_provenance
        ).decode("utf-8")
        self._seed = normalized_seed
        self._row_count = row_count
        self._feature_size = feature_size
        self._dtype = first_features.dtype

    @classmethod
    def from_feature_batches(
        cls,
        feature_batches_by_layer: Mapping[int, Sequence[Tensor]],
        *,
        feature_specification: Mapping[str, Any],
        dataset_provenance: Mapping[str, Any],
        seed: int = 0,
        row_ids: Tensor | None = None,
    ) -> RouterFeatureBank:
        """Flatten ordered feature batches and construct one reference bank.

        Every batch may have arbitrary leading dimensions and must end in the
        configured feature dimension. Leading dimensions are flattened in
        caller-supplied batch order. All layers must therefore produce the
        same total number of rows.

        Gradient-bearing inputs are detached only in the cached copy; the
        caller's tensors and autograd graph are not mutated.
        """

        if not isinstance(feature_batches_by_layer, Mapping):
            raise TypeError("feature_batches_by_layer must be a mapping.")
        if not feature_batches_by_layer:
            raise ValueError("feature_batches_by_layer must not be empty.")

        normalized_specification = _normalize_feature_specification(
            feature_specification
        )
        feature_size = normalized_specification["output_size"]
        flattened: dict[int, Tensor] = {}
        for raw_layer_index, batches in feature_batches_by_layer.items():
            layer_index = _validate_non_negative_integer(
                raw_layer_index,
                field_name="layer_index",
            )
            if isinstance(batches, Tensor) or not isinstance(batches, Sequence):
                raise TypeError(
                    f"Feature batches for layer {layer_index} must be a sequence."
                )
            if not batches:
                raise ValueError(
                    f"Feature batches for layer {layer_index} must not be empty."
                )

            flattened_batches: list[Tensor] = []
            expected_dtype: torch.dtype | None = None
            for batch_index, batch in enumerate(batches):
                _validate_feature_tensor(
                    batch,
                    feature_size=feature_size,
                    field_name=(
                        f"feature_batches_by_layer[{layer_index}]"
                        f"[{batch_index}]"
                    ),
                    require_matrix=False,
                )
                if expected_dtype is None:
                    expected_dtype = batch.dtype
                elif batch.dtype != expected_dtype:
                    raise ValueError(
                        f"Feature batches for layer {layer_index} must share "
                        f"one dtype: {batch.dtype} != {expected_dtype}."
                    )
                # CPU conversion before concatenation permits collection from
                # successive accelerator devices without retaining device
                # storage or autograd history in the reference bank.
                flattened_batches.append(
                    batch.detach()
                    .to(device="cpu")
                    .reshape(-1, feature_size)
                    .contiguous()
                )
            flattened[layer_index] = torch.cat(flattened_batches, dim=0)

        return cls(
            flattened,
            feature_specification=normalized_specification,
            dataset_provenance=dataset_provenance,
            seed=seed,
            row_ids=row_ids,
        )

    def __len__(self) -> int:
        """Return the number of aligned feature rows in every layer."""

        return self._row_count

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return cached decoder-layer indices in ascending order."""

        return tuple(self._features_by_layer)

    @property
    def row_count(self) -> int:
        """Return the common number of cached rows."""

        return self._row_count

    @property
    def feature_size(self) -> int:
        """Return the common final feature dimension."""

        return self._feature_size

    @property
    def dtype(self) -> torch.dtype:
        """Return the common cached feature dtype."""

        return self._dtype

    @property
    def seed(self) -> int:
        """Return the deterministic collection seed recorded by the bank."""

        return self._seed

    @property
    def feature_specification(self) -> dict[str, Any]:
        """Return a defensive JSON copy of router feature-space metadata."""

        return json.loads(self._feature_specification_json)

    @property
    def dataset_provenance(self) -> dict[str, Any]:
        """Return a defensive JSON copy of dataset identity metadata."""

        return json.loads(self._dataset_provenance_json)

    @property
    def row_ids(self) -> Tensor:
        """Return a defensive copy of the aligned integer row identifiers."""

        return self._row_ids.clone()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return canonical serialization metadata including tensor digests."""

        return {
            "dataset_provenance": self.dataset_provenance,
            "feature_specification": self.feature_specification,
            "format": self.serialization_format,
            "format_version": self.format_version,
            "layers": [
                {
                    "dtype": _dtype_name(features.dtype),
                    "feature_size": features.shape[1],
                    "layer_index": layer_index,
                    "row_count": features.shape[0],
                    "sha256": _sha256_tensor(features),
                    "tensor_name": _feature_tensor_name(layer_index),
                }
                for layer_index, features in self._features_by_layer.items()
            ],
            "row_count": self._row_count,
            "row_ids": {
                "dtype": _dtype_name(self._row_ids.dtype),
                "sha256": _sha256_tensor(self._row_ids),
                "tensor_name": "row_ids",
            },
            "seed": self._seed,
        }

    @property
    def metadata_sha256(self) -> str:
        """Return the SHA-256 identity of canonical bank metadata."""

        return sha256_json(self.metadata)

    def features_for_layer(self, layer_index: int) -> Tensor:
        """Return a defensive copy of one ``[row_count, feature_size]`` tensor.

        Raises:
            ValueError: If ``layer_index`` is not a non-negative integer.
            KeyError: If that layer is not present in the bank.
        """

        normalized_index = _validate_non_negative_integer(
            layer_index,
            field_name="layer_index",
        )
        try:
            return self._features_by_layer[normalized_index].clone()
        except KeyError as error:
            raise KeyError(
                f"Router feature bank has no layer {normalized_index}."
            ) from error

    def assert_compatible(
        self,
        feature_specification: Mapping[str, Any],
    ) -> None:
        """Raise when another router feature specification is not identical."""

        normalized = _normalize_feature_specification(feature_specification)
        if canonical_json_bytes(normalized).decode("utf-8") != (
            self._feature_specification_json
        ):
            raise ValueError(
                "Router feature specification is incompatible with the "
                "reference bank."
            )

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize tensors and canonical metadata.

        The payload contains no executable Python. Existing destinations are
        protected unless ``overwrite=True`` is explicit. Embedded tensor
        digests detect accidental finite-value changes, while distribution as
        part of an expert package should additionally use the manifest's
        external whole-file SHA-256 digest.
        """

        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state = {"row_ids": self._row_ids}
        tensor_state.update(
            {
                _feature_tensor_name(layer_index): features
                for layer_index, features in self._features_by_layer.items()
            }
        )
        metadata = {
            _METADATA_KEY: canonical_json_bytes(self.metadata).decode("utf-8")
        }

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=_SAFETENSORS_SUFFIX,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            save_file(tensor_state, temporary_path, metadata=metadata)
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError, ValueError) as error:
            raise FeatureBankSerializationError(
                f"Cannot save router feature bank to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def from_safetensors(cls, path: str | Path) -> RouterFeatureBank:
        """Strictly load and validate a router feature-bank payload.

        Format version, metadata fields, tensor keys, shapes, dtypes, finite
        values, row alignment, and embedded tensor digests are validated before
        returning a bank.
        """

        source = _validate_load_path(path)
        try:
            with safe_open(source, framework="pt", device="cpu") as payload:
                file_metadata = payload.metadata() or {}
        except Exception as error:
            raise FeatureBankSerializationError(
                f"Cannot read router feature bank {source}: {error}"
            ) from error

        serialized_metadata = file_metadata.get(_METADATA_KEY)
        try:
            metadata = json.loads(serialized_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise FeatureBankSerializationError(
                f"Invalid router feature-bank metadata in {source}."
            ) from error
        _validate_loaded_metadata_structure(metadata, source=source)

        try:
            tensor_state = load_file(source, device="cpu")
            expected_tensor_names = {"row_ids"}
            features_by_layer: dict[int, Tensor] = {}
            for descriptor in metadata["layers"]:
                layer_index = descriptor["layer_index"]
                tensor_name = _feature_tensor_name(layer_index)
                if descriptor["tensor_name"] != tensor_name:
                    raise FeatureBankSerializationError(
                        "Router feature-bank layer tensor name is not canonical: "
                        f"{descriptor['tensor_name']!r}."
                    )
                expected_tensor_names.add(tensor_name)
                features_by_layer[layer_index] = tensor_state.get(tensor_name)

            actual_tensor_names = set(tensor_state)
            if actual_tensor_names != expected_tensor_names:
                raise FeatureBankSerializationError(
                    "Router feature-bank tensor keys are incompatible: "
                    f"missing={sorted(expected_tensor_names - actual_tensor_names)}, "
                    f"unexpected={sorted(actual_tensor_names - expected_tensor_names)}"
                )

            bank = cls(
                features_by_layer,
                feature_specification=metadata["feature_specification"],
                dataset_provenance=metadata["dataset_provenance"],
                seed=metadata["seed"],
                row_ids=tensor_state["row_ids"],
            )
            if bank.metadata != metadata:
                raise FeatureBankSerializationError(
                    "Router feature-bank metadata does not match its tensor payload."
                )
        except FeatureBankSerializationError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            raise FeatureBankSerializationError(
                f"Cannot load router feature bank from {source}: {error}"
            ) from error
        return bank


def _normalize_features_by_layer(
    features_by_layer: Mapping[int, Tensor],
    *,
    feature_size: int,
) -> dict[int, Tensor]:
    if not isinstance(features_by_layer, Mapping):
        raise TypeError("features_by_layer must be a mapping.")
    if not features_by_layer:
        raise ValueError("features_by_layer must not be empty.")

    normalized: dict[int, Tensor] = {}
    row_count: int | None = None
    common_dtype: torch.dtype | None = None
    for raw_layer_index, features in features_by_layer.items():
        layer_index = _validate_non_negative_integer(
            raw_layer_index,
            field_name="layer_index",
        )
        if layer_index in normalized:
            raise ValueError(f"Duplicate router feature layer: {layer_index}.")
        _validate_feature_tensor(
            features,
            feature_size=feature_size,
            field_name=f"features_by_layer[{layer_index}]",
            require_matrix=True,
        )
        if features.shape[0] <= 0:
            raise ValueError("Router feature banks must contain at least one row.")
        if row_count is None:
            row_count = features.shape[0]
            common_dtype = features.dtype
        elif features.shape[0] != row_count:
            raise ValueError(
                "Every cached layer must have the same row count: "
                f"{features.shape[0]} != {row_count}."
            )
        elif features.dtype != common_dtype:
            raise ValueError(
                "Every cached layer must have the same dtype: "
                f"{features.dtype} != {common_dtype}."
            )

        # clone() owns independent storage, and detachment ensures calibration
        # caches cannot retain a training graph or mutate the source tensor.
        normalized[layer_index] = (
            features.detach().to(device="cpu").contiguous().clone()
        )
    return dict(sorted(normalized.items()))


def _validate_feature_tensor(
    features: Any,
    *,
    feature_size: int,
    field_name: str,
    require_matrix: bool,
) -> None:
    if not isinstance(features, Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if not features.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype.")
    required_dimensions = 2 if require_matrix else 1
    if features.ndim < required_dimensions:
        shape_description = "two-dimensional" if require_matrix else "non-scalar"
        raise ValueError(f"{field_name} must be {shape_description}.")
    if require_matrix and features.ndim != 2:
        raise ValueError(f"{field_name} must have shape [rows, feature_size].")
    if features.shape[-1] != feature_size:
        raise ValueError(
            f"{field_name} final dimension does not match output_size: "
            f"{features.shape[-1]} != {feature_size}."
        )
    if not torch.isfinite(features).all().item():
        raise ValueError(f"{field_name} must contain only finite values.")


def _normalize_row_ids(row_ids: Tensor | None, *, row_count: int) -> Tensor:
    if row_ids is None:
        return torch.arange(row_count, dtype=torch.long)
    if not isinstance(row_ids, Tensor):
        raise TypeError("row_ids must be a torch.Tensor.")
    if row_ids.dtype != torch.long:
        raise TypeError("row_ids must use torch.long.")
    if row_ids.ndim != 1 or row_ids.shape[0] != row_count:
        raise ValueError(f"row_ids must have shape [{row_count}].")
    if row_ids.numel() and int(row_ids.min().item()) < 0:
        raise ValueError("row_ids must be non-negative.")
    if torch.unique(row_ids).numel() != row_count:
        raise ValueError("row_ids must be unique.")
    return row_ids.detach().to(device="cpu").contiguous().clone()


def _normalize_feature_specification(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json_mapping(value, field_name="feature_specification")
    if "output_size" not in normalized:
        raise ValueError("feature_specification must contain output_size.")
    normalized["output_size"] = _validate_positive_integer(
        normalized["output_size"],
        field_name="feature_specification.output_size",
    )
    return normalized


def _normalize_dataset_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json_mapping(value, field_name="dataset_provenance")
    missing = sorted(_REQUIRED_DATASET_FIELDS - set(normalized))
    if missing:
        raise ValueError(
            f"dataset_provenance is missing required fields: {missing}"
        )
    unknown = sorted(set(normalized) - _REQUIRED_DATASET_FIELDS)
    if unknown:
        raise ValueError(
            "dataset_provenance contains unsupported fields that could retain "
            f"example content: {unknown}"
        )
    for field_name in ("dataset_id", "revision", "split"):
        _validate_nonempty_string(
            normalized[field_name],
            field_name=f"dataset_provenance.{field_name}",
        )
    return normalized


def _normalize_json_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized = _normalize_json(value, field_name=field_name)
    return dict(normalized)


def _normalize_json(value: Any, *, field_name: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _validate_nonempty_string(key, field_name=field_name)
            normalized[normalized_key] = _normalize_json(
                item,
                field_name=f"{field_name}.{normalized_key}",
            )
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, field_name=f"{field_name}[]") for item in value
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        f"{field_name} contains a non-JSON or non-finite value: "
        f"{type(value).__name__}."
    )


def _validate_loaded_metadata_structure(
    metadata: Any,
    *,
    source: Path,
) -> None:
    if not isinstance(metadata, dict):
        raise FeatureBankSerializationError(
            f"Invalid router feature-bank metadata in {source}."
        )
    expected_fields = {
        "dataset_provenance",
        "feature_specification",
        "format",
        "format_version",
        "layers",
        "row_count",
        "row_ids",
        "seed",
    }
    if set(metadata) != expected_fields:
        raise FeatureBankSerializationError(
            "Router feature-bank metadata fields are incompatible: "
            f"missing={sorted(expected_fields - set(metadata))}, "
            f"unexpected={sorted(set(metadata) - expected_fields)}"
        )
    if metadata["format"] != _SERIALIZATION_FORMAT:
        raise FeatureBankSerializationError(
            f"Unsupported router feature-bank format: {metadata['format']!r}."
        )
    if metadata["format_version"] != ROUTER_FEATURE_BANK_FORMAT_VERSION:
        raise FeatureBankSerializationError(
            "Unsupported router feature-bank format_version: "
            f"{metadata['format_version']!r}."
        )
    if not isinstance(metadata["layers"], list) or not metadata["layers"]:
        raise FeatureBankSerializationError(
            "Router feature-bank layers metadata must be a non-empty list."
        )
    expected_layer_fields = {
        "dtype",
        "feature_size",
        "layer_index",
        "row_count",
        "sha256",
        "tensor_name",
    }
    for descriptor in metadata["layers"]:
        if not isinstance(descriptor, dict) or set(descriptor) != (
            expected_layer_fields
        ):
            raise FeatureBankSerializationError(
                "Router feature-bank layer metadata is malformed."
            )
    if not isinstance(metadata["row_ids"], dict) or set(
        metadata["row_ids"]
    ) != {"dtype", "sha256", "tensor_name"}:
        raise FeatureBankSerializationError(
            "Router feature-bank row_ids metadata is malformed."
        )


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise FeatureBankSerializationError(
            f"Feature-bank payload must end in {_SAFETENSORS_SUFFIX}: {destination}"
        )
    if destination.is_symlink():
        raise FeatureBankSerializationError(
            f"Feature-bank payload must not be a symbolic link: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Feature-bank payload already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise FeatureBankSerializationError(
            f"Feature-bank destination is not a regular file: {destination}"
        )
    if not destination.parent.is_dir():
        raise FeatureBankSerializationError(
            f"Feature-bank parent directory does not exist: {destination.parent}"
        )
    return destination


def _validate_load_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise FeatureBankSerializationError(
            f"Feature-bank payload must end in {_SAFETENSORS_SUFFIX}: {source}"
        )
    if source.is_symlink():
        raise FeatureBankSerializationError(
            f"Feature-bank payload must not be a symbolic link: {source}"
        )
    if not source.is_file():
        raise FeatureBankSerializationError(
            f"Feature-bank payload is not a regular file: {source}"
        )
    return source


def _feature_tensor_name(layer_index: int) -> str:
    return f"layers.{layer_index}.features"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sha256_tensor(tensor: Tensor) -> str:
    portable = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = portable.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw_bytes).hexdigest()


def _validate_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace."
        )
    return value


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _validate_non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value
