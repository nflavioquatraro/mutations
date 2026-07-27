"""Row-aligned domain and individual-utility router calibration targets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from anchored_moe.calibration.features import RouterFeatureBank
from anchored_moe.utils.hashing import canonical_json_bytes, sha256_json


ROUTER_LABEL_BANK_FORMAT_VERSION: Final = 1
_DOMAIN_FORMAT: Final = "anchored_moe.domain_label_bank.v1"
_UTILITY_FORMAT: Final = "anchored_moe.utility_label_bank.v1"
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class LabelBankSerializationError(ValueError):
    """Raised when a router label-bank payload is unsafe or incompatible."""


class DomainLabelBank:
    """Immutable categorical domain targets aligned to a router feature bank.

    Integer label ``c_t`` identifies the domain class for reference row ``t``.
    Class index order is explicit and serialized. Use
    :meth:`from_domain_names` to derive deterministic lexicographic class order
    from string labels.

    Args:
        labels:
            One-dimensional ``torch.long`` tensor with shape ``[row_count]``.
        class_names:
            Unique non-empty domain identifiers in integer-label order.
        row_ids:
            Unique non-negative ``torch.long`` identifiers with shape
            ``[row_count]``.
        feature_bank_metadata_sha256:
            Content identity of the exact CAL-001 feature bank.
        label_source:
            Non-empty provenance identifier such as ``"dataset:domain"``.

    Construction owns detached CPU copies. Accessors return defensive copies.
    Raw prompts, token text, and reusable feature tensors are deliberately not
    part of this label format.
    """

    format_version: ClassVar[int] = ROUTER_LABEL_BANK_FORMAT_VERSION
    serialization_format: ClassVar[str] = _DOMAIN_FORMAT

    def __init__(
        self,
        labels: Tensor,
        *,
        class_names: Sequence[str],
        row_ids: Tensor,
        feature_bank_metadata_sha256: str,
        label_source: str,
    ) -> None:
        normalized_class_names = _normalize_identifiers(
            class_names,
            field_name="class_names",
        )
        normalized_labels = _normalize_domain_labels(
            labels,
            class_count=len(normalized_class_names),
        )
        normalized_row_ids = _normalize_row_ids(
            row_ids,
            row_count=normalized_labels.shape[0],
        )

        self._labels = normalized_labels
        self._class_names = normalized_class_names
        self._row_ids = normalized_row_ids
        self._feature_bank_metadata_sha256 = _validate_sha256(
            feature_bank_metadata_sha256,
            field_name="feature_bank_metadata_sha256",
        )
        self._label_source = _validate_identifier(
            label_source,
            field_name="label_source",
        )

    @classmethod
    def from_domain_names(
        cls,
        feature_bank: RouterFeatureBank,
        domain_names: Sequence[str],
        *,
        label_source: str,
    ) -> DomainLabelBank:
        """Encode string domains in sorted order and align them to a bank."""

        _validate_feature_bank(feature_bank)
        normalized_names = _normalize_label_values(
            domain_names,
            row_count=feature_bank.row_count,
        )
        class_names = tuple(sorted(set(normalized_names)))
        class_indices = {name: index for index, name in enumerate(class_names)}
        labels = torch.tensor(
            [class_indices[name] for name in normalized_names],
            dtype=torch.long,
        )
        return cls(
            labels,
            class_names=class_names,
            row_ids=feature_bank.row_ids,
            feature_bank_metadata_sha256=feature_bank.metadata_sha256,
            label_source=label_source,
        )

    def __len__(self) -> int:
        """Return the number of labeled reference rows."""

        return self._labels.shape[0]

    @property
    def row_count(self) -> int:
        """Return the number of labeled reference rows."""

        return len(self)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return categorical domains in stable integer-label order."""

        return self._class_names

    @property
    def class_count(self) -> int:
        """Return the number of declared domain classes."""

        return len(self._class_names)

    @property
    def labels(self) -> Tensor:
        """Return a defensive copy of integer domain targets."""

        return self._labels.clone()

    @property
    def domain_names(self) -> tuple[str, ...]:
        """Return decoded domain identifiers in row order."""

        return tuple(self._class_names[index] for index in self._labels.tolist())

    @property
    def row_ids(self) -> Tensor:
        """Return a defensive copy of aligned row identifiers."""

        return self._row_ids.clone()

    @property
    def feature_bank_metadata_sha256(self) -> str:
        """Return the exact CAL-001 feature-bank content identity."""

        return self._feature_bank_metadata_sha256

    @property
    def label_source(self) -> str:
        """Return the declared categorical-label provenance identifier."""

        return self._label_source

    @property
    def metadata(self) -> dict[str, Any]:
        """Return canonical serialization metadata and tensor digests."""

        return {
            "class_names": list(self._class_names),
            "feature_bank_metadata_sha256": self._feature_bank_metadata_sha256,
            "format": self.serialization_format,
            "format_version": self.format_version,
            "label_source": self._label_source,
            "labels": {
                "dtype": _dtype_name(self._labels.dtype),
                "sha256": _sha256_tensor(self._labels),
                "tensor_name": "labels",
            },
            "row_count": self.row_count,
            "row_ids": {
                "dtype": _dtype_name(self._row_ids.dtype),
                "sha256": _sha256_tensor(self._row_ids),
                "tensor_name": "row_ids",
            },
        }

    @property
    def metadata_sha256(self) -> str:
        """Return the content identity of canonical label metadata."""

        return sha256_json(self.metadata)

    def assert_aligned(self, feature_bank: RouterFeatureBank) -> None:
        """Raise unless content identity and row identifiers match exactly."""

        _validate_feature_bank(feature_bank)
        if feature_bank.metadata_sha256 != self._feature_bank_metadata_sha256:
            raise ValueError(
                "Domain labels reference a different router feature bank."
            )
        if not torch.equal(feature_bank.row_ids, self._row_ids):
            raise ValueError(
                "Domain-label row identifiers do not match the feature bank."
            )

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize categorical labels without executable content."""

        _save_label_bank(
            path,
            tensors={"labels": self._labels, "row_ids": self._row_ids},
            metadata=self.metadata,
            overwrite=overwrite,
        )

    @classmethod
    def from_safetensors(cls, path: str | Path) -> DomainLabelBank:
        """Strictly load categorical labels, schema, and integrity metadata."""

        source, metadata, tensors = _load_label_payload(
            path,
            expected_format=_DOMAIN_FORMAT,
            expected_metadata_fields={
                "class_names",
                "feature_bank_metadata_sha256",
                "format",
                "format_version",
                "label_source",
                "labels",
                "row_count",
                "row_ids",
            },
        )
        if set(tensors) != {"labels", "row_ids"}:
            raise LabelBankSerializationError(
                "Domain label-bank tensor keys are incompatible: "
                f"missing={sorted({'labels', 'row_ids'} - set(tensors))}, "
                f"unexpected={sorted(set(tensors) - {'labels', 'row_ids'})}"
            )
        _validate_tensor_descriptor(
            metadata["labels"],
            expected_tensor_name="labels",
            field_name="labels",
        )
        _validate_tensor_descriptor(
            metadata["row_ids"],
            expected_tensor_name="row_ids",
            field_name="row_ids",
        )
        try:
            bank = cls(
                tensors["labels"],
                class_names=metadata["class_names"],
                row_ids=tensors["row_ids"],
                feature_bank_metadata_sha256=(
                    metadata["feature_bank_metadata_sha256"]
                ),
                label_source=metadata["label_source"],
            )
        except (TypeError, ValueError, KeyError) as error:
            raise LabelBankSerializationError(
                f"Cannot load domain label bank from {source}: {error}"
            ) from error
        if bank.metadata != metadata:
            raise LabelBankSerializationError(
                "Domain label-bank metadata does not match its tensor payload."
            )
        return bank


class UtilityLabelBank:
    """Immutable individual-expert utility targets aligned by layer and row.

    For layer ``l``, reference row ``t``, and expert ``k``, the stored target
    is the loss advantage

    ``a_(l,t,k) = incumbent_loss_(l,t) - expert_loss_(l,t,k)``.

    Positive utility means the forced expert path improved on the declared
    incumbent. Each utility matrix has shape ``[row_count, expert_count_l]``;
    columns follow the explicitly serialized expert identifier order.

    Args:
        utilities_by_layer:
            Mapping from layer index to finite floating-point utility matrices.
        expert_ids_by_layer:
            Matching mapping from layer index to unique expert identifiers.
        row_ids:
            Unique non-negative ``torch.long`` identifiers.
        feature_bank_metadata_sha256:
            Content identity of the exact CAL-001 feature bank.
        loss_name:
            Loss definition used for both incumbent and forced expert paths.
        incumbent_route:
            Identifier for the comparison path, such as ``"anchor"`` or a
            registry hash.

    Pair utilities, pair sampling, counterfactual execution, and router fitting
    are deliberately excluded.
    """

    format_version: ClassVar[int] = ROUTER_LABEL_BANK_FORMAT_VERSION
    serialization_format: ClassVar[str] = _UTILITY_FORMAT

    def __init__(
        self,
        utilities_by_layer: Mapping[int, Tensor],
        *,
        expert_ids_by_layer: Mapping[int, Sequence[str]],
        row_ids: Tensor,
        feature_bank_metadata_sha256: str,
        loss_name: str,
        incumbent_route: str,
    ) -> None:
        normalized_utilities, normalized_expert_ids = _normalize_utilities(
            utilities_by_layer,
            expert_ids_by_layer=expert_ids_by_layer,
        )
        first_utility = next(iter(normalized_utilities.values()))
        normalized_row_ids = _normalize_row_ids(
            row_ids,
            row_count=first_utility.shape[0],
        )

        self._utilities_by_layer = normalized_utilities
        self._expert_ids_by_layer = normalized_expert_ids
        self._row_ids = normalized_row_ids
        self._feature_bank_metadata_sha256 = _validate_sha256(
            feature_bank_metadata_sha256,
            field_name="feature_bank_metadata_sha256",
        )
        self._loss_name = _validate_identifier(
            loss_name,
            field_name="loss_name",
        )
        self._incumbent_route = _validate_identifier(
            incumbent_route,
            field_name="incumbent_route",
        )
        self._dtype = first_utility.dtype

    @classmethod
    def from_losses(
        cls,
        feature_bank: RouterFeatureBank,
        *,
        incumbent_losses_by_layer: Mapping[int, Tensor],
        expert_losses_by_layer: Mapping[int, Tensor],
        expert_ids_by_layer: Mapping[int, Sequence[str]],
        loss_name: str,
        incumbent_route: str,
    ) -> UtilityLabelBank:
        """Compute individual utility targets from aligned forced-path losses.

        Incumbent tensors must have shape ``[row_count]`` and expert tensors
        ``[row_count, expert_count]``. All three layer mappings must exactly
        cover the feature bank's layers. Inputs are read without mutation or
        gradient detachment side effects.
        """

        _validate_feature_bank(feature_bank)
        if not isinstance(incumbent_losses_by_layer, Mapping):
            raise TypeError("incumbent_losses_by_layer must be a mapping.")
        if not isinstance(expert_losses_by_layer, Mapping):
            raise TypeError("expert_losses_by_layer must be a mapping.")
        if not isinstance(expert_ids_by_layer, Mapping):
            raise TypeError("expert_ids_by_layer must be a mapping.")

        expected_layers = set(feature_bank.layer_indices)
        _require_exact_layer_keys(
            incumbent_losses_by_layer,
            expected_layers=expected_layers,
            field_name="incumbent_losses_by_layer",
        )
        _require_exact_layer_keys(
            expert_losses_by_layer,
            expected_layers=expected_layers,
            field_name="expert_losses_by_layer",
        )
        _require_exact_layer_keys(
            expert_ids_by_layer,
            expected_layers=expected_layers,
            field_name="expert_ids_by_layer",
        )

        utilities: dict[int, Tensor] = {}
        for layer_index in feature_bank.layer_indices:
            incumbent_losses = incumbent_losses_by_layer[layer_index]
            expert_losses = expert_losses_by_layer[layer_index]
            _validate_loss_tensors(
                incumbent_losses,
                expert_losses,
                row_count=feature_bank.row_count,
                layer_index=layer_index,
            )
            # The advantage sign is a semantic contract: positive values mean
            # the expert reduced loss relative to the incumbent.
            utilities[layer_index] = (
                incumbent_losses.detach().to(device="cpu").unsqueeze(-1)
                - expert_losses.detach().to(device="cpu")
            )

        return cls(
            utilities,
            expert_ids_by_layer=expert_ids_by_layer,
            row_ids=feature_bank.row_ids,
            feature_bank_metadata_sha256=feature_bank.metadata_sha256,
            loss_name=loss_name,
            incumbent_route=incumbent_route,
        )

    def __len__(self) -> int:
        """Return the common number of labeled reference rows."""

        first_utility = next(iter(self._utilities_by_layer.values()))
        return first_utility.shape[0]

    @property
    def row_count(self) -> int:
        """Return the common number of labeled reference rows."""

        return len(self)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return utility-target layers in ascending order."""

        return tuple(self._utilities_by_layer)

    @property
    def dtype(self) -> torch.dtype:
        """Return the common utility dtype."""

        return self._dtype

    @property
    def row_ids(self) -> Tensor:
        """Return a defensive copy of aligned row identifiers."""

        return self._row_ids.clone()

    @property
    def feature_bank_metadata_sha256(self) -> str:
        """Return the exact CAL-001 feature-bank content identity."""

        return self._feature_bank_metadata_sha256

    @property
    def loss_name(self) -> str:
        """Return the loss definition used to compute utility."""

        return self._loss_name

    @property
    def incumbent_route(self) -> str:
        """Return the comparison route used to compute utility."""

        return self._incumbent_route

    def expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return expert columns for one layer in stable order."""

        normalized_index = _validate_non_negative_integer(
            layer_index,
            field_name="layer_index",
        )
        try:
            return self._expert_ids_by_layer[normalized_index]
        except KeyError as error:
            raise KeyError(
                f"Utility label bank has no layer {normalized_index}."
            ) from error

    def utilities_for_layer(self, layer_index: int) -> Tensor:
        """Return a defensive copy of one utility design target matrix."""

        normalized_index = _validate_non_negative_integer(
            layer_index,
            field_name="layer_index",
        )
        try:
            return self._utilities_by_layer[normalized_index].clone()
        except KeyError as error:
            raise KeyError(
                f"Utility label bank has no layer {normalized_index}."
            ) from error

    @property
    def metadata(self) -> dict[str, Any]:
        """Return canonical serialization metadata and tensor digests."""

        return {
            "feature_bank_metadata_sha256": self._feature_bank_metadata_sha256,
            "format": self.serialization_format,
            "format_version": self.format_version,
            "incumbent_route": self._incumbent_route,
            "layers": [
                {
                    "dtype": _dtype_name(utilities.dtype),
                    "expert_count": utilities.shape[1],
                    "expert_ids": list(self._expert_ids_by_layer[layer_index]),
                    "layer_index": layer_index,
                    "row_count": utilities.shape[0],
                    "sha256": _sha256_tensor(utilities),
                    "tensor_name": _utility_tensor_name(layer_index),
                }
                for layer_index, utilities in self._utilities_by_layer.items()
            ],
            "loss_name": self._loss_name,
            "row_count": self.row_count,
            "row_ids": {
                "dtype": _dtype_name(self._row_ids.dtype),
                "sha256": _sha256_tensor(self._row_ids),
                "tensor_name": "row_ids",
            },
        }

    @property
    def metadata_sha256(self) -> str:
        """Return the content identity of canonical utility metadata."""

        return sha256_json(self.metadata)

    def assert_aligned(self, feature_bank: RouterFeatureBank) -> None:
        """Raise unless feature content, rows, and layer sets match exactly."""

        _validate_feature_bank(feature_bank)
        if feature_bank.metadata_sha256 != self._feature_bank_metadata_sha256:
            raise ValueError(
                "Utility labels reference a different router feature bank."
            )
        if not torch.equal(feature_bank.row_ids, self._row_ids):
            raise ValueError(
                "Utility-label row identifiers do not match the feature bank."
            )
        if self.layer_indices != feature_bank.layer_indices:
            raise ValueError(
                "Utility-label layers do not match the feature bank: "
                f"{self.layer_indices} != {feature_bank.layer_indices}."
            )

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize utility targets without executable content."""

        tensors = {"row_ids": self._row_ids}
        tensors.update(
            {
                _utility_tensor_name(layer_index): utilities
                for layer_index, utilities in self._utilities_by_layer.items()
            }
        )
        _save_label_bank(
            path,
            tensors=tensors,
            metadata=self.metadata,
            overwrite=overwrite,
        )

    @classmethod
    def from_safetensors(cls, path: str | Path) -> UtilityLabelBank:
        """Strictly load utility targets, schema, and integrity metadata."""

        source, metadata, tensors = _load_label_payload(
            path,
            expected_format=_UTILITY_FORMAT,
            expected_metadata_fields={
                "feature_bank_metadata_sha256",
                "format",
                "format_version",
                "incumbent_route",
                "layers",
                "loss_name",
                "row_count",
                "row_ids",
            },
        )
        if not isinstance(metadata["layers"], list) or not metadata["layers"]:
            raise LabelBankSerializationError(
                "Utility label-bank layers metadata must be a non-empty list."
            )
        _validate_tensor_descriptor(
            metadata["row_ids"],
            expected_tensor_name="row_ids",
            field_name="row_ids",
        )

        expected_tensor_names = {"row_ids"}
        utilities_by_layer: dict[int, Tensor] = {}
        expert_ids_by_layer: dict[int, Sequence[str]] = {}
        expected_layer_fields = {
            "dtype",
            "expert_count",
            "expert_ids",
            "layer_index",
            "row_count",
            "sha256",
            "tensor_name",
        }
        for descriptor in metadata["layers"]:
            if not isinstance(descriptor, dict) or set(descriptor) != (
                expected_layer_fields
            ):
                raise LabelBankSerializationError(
                    "Utility label-bank layer metadata is malformed."
                )
            layer_index = _validate_loaded_layer_index(descriptor["layer_index"])
            tensor_name = _utility_tensor_name(layer_index)
            _validate_tensor_descriptor(
                descriptor,
                expected_tensor_name=tensor_name,
                field_name=f"layers[{layer_index}]",
                additional_fields={
                    "expert_count",
                    "expert_ids",
                    "layer_index",
                    "row_count",
                },
            )
            if layer_index in utilities_by_layer:
                raise LabelBankSerializationError(
                    f"Duplicate utility layer metadata: {layer_index}."
                )
            expected_tensor_names.add(tensor_name)
            utilities_by_layer[layer_index] = tensors.get(tensor_name)
            expert_ids_by_layer[layer_index] = descriptor["expert_ids"]

        if set(tensors) != expected_tensor_names:
            raise LabelBankSerializationError(
                "Utility label-bank tensor keys are incompatible: "
                f"missing={sorted(expected_tensor_names - set(tensors))}, "
                f"unexpected={sorted(set(tensors) - expected_tensor_names)}"
            )
        try:
            bank = cls(
                utilities_by_layer,
                expert_ids_by_layer=expert_ids_by_layer,
                row_ids=tensors["row_ids"],
                feature_bank_metadata_sha256=(
                    metadata["feature_bank_metadata_sha256"]
                ),
                loss_name=metadata["loss_name"],
                incumbent_route=metadata["incumbent_route"],
            )
        except (TypeError, ValueError, KeyError) as error:
            raise LabelBankSerializationError(
                f"Cannot load utility label bank from {source}: {error}"
            ) from error
        if bank.metadata != metadata:
            raise LabelBankSerializationError(
                "Utility label-bank metadata does not match its tensor payload."
            )
        return bank


def _normalize_domain_labels(labels: Any, *, class_count: int) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a torch.Tensor.")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long.")
    if labels.ndim != 1 or labels.shape[0] <= 0:
        raise ValueError("labels must be a non-empty one-dimensional tensor.")
    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < 0 or maximum >= class_count:
        raise ValueError(
            "labels contain a class index outside class_names: "
            f"minimum={minimum}, maximum={maximum}, class_count={class_count}."
        )
    return labels.detach().to(device="cpu").contiguous().clone()


def _normalize_label_values(
    domain_names: Sequence[str],
    *,
    row_count: int,
) -> tuple[str, ...]:
    if isinstance(domain_names, (str, bytes)) or not isinstance(
        domain_names,
        Sequence,
    ):
        raise TypeError("domain_names must be a sequence of strings.")
    if len(domain_names) != row_count:
        raise ValueError(
            f"domain_names must contain {row_count} aligned values."
        )
    return tuple(
        _validate_identifier(name, field_name="domain_names")
        for name in domain_names
    )


def _normalize_utilities(
    utilities_by_layer: Mapping[int, Tensor],
    *,
    expert_ids_by_layer: Mapping[int, Sequence[str]],
) -> tuple[dict[int, Tensor], dict[int, tuple[str, ...]]]:
    if not isinstance(utilities_by_layer, Mapping):
        raise TypeError("utilities_by_layer must be a mapping.")
    if not utilities_by_layer:
        raise ValueError("utilities_by_layer must not be empty.")
    if not isinstance(expert_ids_by_layer, Mapping):
        raise TypeError("expert_ids_by_layer must be a mapping.")

    raw_layers = set(utilities_by_layer)
    expert_layers = set(expert_ids_by_layer)
    if raw_layers != expert_layers:
        raise ValueError(
            "expert_ids_by_layer keys must match utilities_by_layer: "
            f"missing={sorted(raw_layers - expert_layers)}, "
            f"unexpected={sorted(expert_layers - raw_layers)}"
        )

    normalized_utilities: dict[int, Tensor] = {}
    normalized_expert_ids: dict[int, tuple[str, ...]] = {}
    row_count: int | None = None
    common_dtype: torch.dtype | None = None
    for raw_layer_index, utilities in utilities_by_layer.items():
        layer_index = _validate_non_negative_integer(
            raw_layer_index,
            field_name="layer_index",
        )
        if layer_index in normalized_utilities:
            raise ValueError(f"Duplicate utility layer: {layer_index}.")
        if not isinstance(utilities, Tensor):
            raise TypeError(
                f"utilities_by_layer[{layer_index}] must be a torch.Tensor."
            )
        if not utilities.is_floating_point():
            raise TypeError(
                f"utilities_by_layer[{layer_index}] must be floating-point."
            )
        if utilities.ndim != 2 or min(utilities.shape) <= 0:
            raise ValueError(
                f"utilities_by_layer[{layer_index}] must have non-empty "
                "shape [row_count, expert_count]."
            )
        if not torch.isfinite(utilities).all().item():
            raise ValueError(
                f"utilities_by_layer[{layer_index}] must contain finite values."
            )

        expert_ids = _normalize_identifiers(
            expert_ids_by_layer[raw_layer_index],
            field_name=f"expert_ids_by_layer[{layer_index}]",
        )
        if utilities.shape[1] != len(expert_ids):
            raise ValueError(
                f"Utility columns for layer {layer_index} do not match expert "
                f"identifiers: {utilities.shape[1]} != {len(expert_ids)}."
            )
        if row_count is None:
            row_count = utilities.shape[0]
            common_dtype = utilities.dtype
        elif utilities.shape[0] != row_count:
            raise ValueError(
                "Every utility layer must have the same row count: "
                f"{utilities.shape[0]} != {row_count}."
            )
        elif utilities.dtype != common_dtype:
            raise ValueError(
                "Every utility layer must have the same dtype: "
                f"{utilities.dtype} != {common_dtype}."
            )

        normalized_utilities[layer_index] = (
            utilities.detach().to(device="cpu").contiguous().clone()
        )
        normalized_expert_ids[layer_index] = expert_ids
    return (
        dict(sorted(normalized_utilities.items())),
        dict(sorted(normalized_expert_ids.items())),
    )


def _validate_loss_tensors(
    incumbent_losses: Any,
    expert_losses: Any,
    *,
    row_count: int,
    layer_index: int,
) -> None:
    if not isinstance(incumbent_losses, Tensor):
        raise TypeError(
            f"incumbent_losses_by_layer[{layer_index}] must be a tensor."
        )
    if not isinstance(expert_losses, Tensor):
        raise TypeError(
            f"expert_losses_by_layer[{layer_index}] must be a tensor."
        )
    if not incumbent_losses.is_floating_point():
        raise TypeError("Incumbent losses must be floating-point.")
    if not expert_losses.is_floating_point():
        raise TypeError("Expert losses must be floating-point.")
    if incumbent_losses.shape != (row_count,):
        raise ValueError(
            f"Incumbent losses for layer {layer_index} must have shape "
            f"[{row_count}]."
        )
    if (
        expert_losses.ndim != 2
        or expert_losses.shape[0] != row_count
        or expert_losses.shape[1] <= 0
    ):
        raise ValueError(
            f"Expert losses for layer {layer_index} must have non-empty shape "
            f"[{row_count}, expert_count]."
        )
    if incumbent_losses.dtype != expert_losses.dtype:
        raise ValueError(
            f"Loss dtypes for layer {layer_index} must match: "
            f"{incumbent_losses.dtype} != {expert_losses.dtype}."
        )
    if incumbent_losses.device != expert_losses.device:
        raise ValueError(
            f"Loss devices for layer {layer_index} must match: "
            f"{incumbent_losses.device} != {expert_losses.device}."
        )
    if not torch.isfinite(incumbent_losses).all().item():
        raise ValueError("Incumbent losses must contain finite values.")
    if not torch.isfinite(expert_losses).all().item():
        raise ValueError("Expert losses must contain finite values.")


def _require_exact_layer_keys(
    value: Mapping[int, Any],
    *,
    expected_layers: set[int],
    field_name: str,
) -> None:
    actual_layers = {
        _validate_non_negative_integer(
            layer_index,
            field_name=f"{field_name} layer index",
        )
        for layer_index in value
    }
    if actual_layers != expected_layers:
        raise ValueError(
            f"{field_name} must exactly match feature-bank layers: "
            f"missing={sorted(expected_layers - actual_layers)}, "
            f"unexpected={sorted(actual_layers - expected_layers)}"
        )


def _normalize_row_ids(row_ids: Any, *, row_count: int) -> Tensor:
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


def _normalize_identifiers(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(
        _validate_identifier(value, field_name=field_name) for value in values
    )
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return normalized


def _validate_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must contain non-empty strings without surrounding "
            "whitespace."
        )
    return value


def _validate_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _validate_non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_feature_bank(feature_bank: Any) -> None:
    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")


def _utility_tensor_name(layer_index: int) -> str:
    return f"layers.{layer_index}.utilities"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sha256_tensor(tensor: Tensor) -> str:
    portable = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = portable.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw_bytes).hexdigest()


def _save_label_bank(
    path: str | Path,
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> None:
    destination = _validate_save_path(path, overwrite=overwrite)
    serialized_metadata = {
        _METADATA_KEY: canonical_json_bytes(metadata).decode("utf-8")
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
        save_file(dict(tensors), temporary_path, metadata=serialized_metadata)
        os.replace(temporary_path, destination)
    except (OSError, RuntimeError, ValueError) as error:
        raise LabelBankSerializationError(
            f"Cannot save router label bank to {destination}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_label_payload(
    path: str | Path,
    *,
    expected_format: str,
    expected_metadata_fields: set[str],
) -> tuple[Path, dict[str, Any], dict[str, Tensor]]:
    source = _validate_load_path(path)
    try:
        with safe_open(source, framework="pt", device="cpu") as payload:
            file_metadata = payload.metadata() or {}
    except Exception as error:
        raise LabelBankSerializationError(
            f"Cannot read router label bank {source}: {error}"
        ) from error

    try:
        metadata = json.loads(file_metadata.get(_METADATA_KEY))
    except (TypeError, json.JSONDecodeError) as error:
        raise LabelBankSerializationError(
            f"Invalid router label-bank metadata in {source}."
        ) from error
    if not isinstance(metadata, dict):
        raise LabelBankSerializationError(
            f"Invalid router label-bank metadata in {source}."
        )
    if metadata.get("format") != expected_format:
        raise LabelBankSerializationError(
            f"Unsupported router label-bank format: {metadata.get('format')!r}."
        )
    if metadata.get("format_version") != ROUTER_LABEL_BANK_FORMAT_VERSION:
        raise LabelBankSerializationError(
            "Unsupported router label-bank format_version: "
            f"{metadata.get('format_version')!r}."
        )
    if set(metadata) != expected_metadata_fields:
        raise LabelBankSerializationError(
            "Router label-bank metadata fields are incompatible: "
            f"missing={sorted(expected_metadata_fields - set(metadata))}, "
            f"unexpected={sorted(set(metadata) - expected_metadata_fields)}"
        )
    try:
        tensors = load_file(source, device="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise LabelBankSerializationError(
            f"Cannot load router label tensors from {source}: {error}"
        ) from error
    return source, metadata, tensors


def _validate_tensor_descriptor(
    value: Any,
    *,
    expected_tensor_name: str,
    field_name: str,
    additional_fields: set[str] | None = None,
) -> None:
    expected_fields = {"dtype", "sha256", "tensor_name"}
    if additional_fields is not None:
        expected_fields.update(additional_fields)
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise LabelBankSerializationError(
            f"Router label-bank {field_name} metadata is malformed."
        )
    if value["tensor_name"] != expected_tensor_name:
        raise LabelBankSerializationError(
            f"Router label-bank {field_name} tensor name is not canonical."
        )


def _validate_loaded_layer_index(value: Any) -> int:
    try:
        return _validate_non_negative_integer(value, field_name="layer_index")
    except ValueError as error:
        raise LabelBankSerializationError(str(error)) from error


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise LabelBankSerializationError(
            f"Label-bank payload must end in {_SAFETENSORS_SUFFIX}: {destination}"
        )
    if destination.is_symlink():
        raise LabelBankSerializationError(
            f"Label-bank payload must not be a symbolic link: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Label-bank payload already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise LabelBankSerializationError(
            f"Label-bank destination is not a regular file: {destination}"
        )
    if not destination.parent.is_dir():
        raise LabelBankSerializationError(
            f"Label-bank parent directory does not exist: {destination.parent}"
        )
    return destination


def _validate_load_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise LabelBankSerializationError(
            f"Label-bank payload must end in {_SAFETENSORS_SUFFIX}: {source}"
        )
    if source.is_symlink():
        raise LabelBankSerializationError(
            f"Label-bank payload must not be a symbolic link: {source}"
        )
    if not source.is_file():
        raise LabelBankSerializationError(
            f"Label-bank payload is not a regular file: {source}"
        )
    return source
