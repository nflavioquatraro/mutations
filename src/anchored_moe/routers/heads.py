"""Independent affine router heads with append-only safe serialization."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from anchored_moe.routers.base import RouterHead


_SERIALIZATION_FORMAT: Final = "anchored_moe.independent_affine_heads.v1"
_ROUTER_ARCHITECTURE: Final = "independent_affine_heads"
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"


class RouterSerializationError(ValueError):
    """Raised when an affine-router tensor payload is unsafe or incompatible."""


class AffineRouterHead(RouterHead):
    """Produce one calibrated expert score from frozen router features.

    For feature vector ``phi_l(h)`` with final dimension ``feature_size``, the
    head computes

    ``g_(l,k)(h) = (w_(l,k)^T phi_l(h) + b_(l,k)) / temperature - threshold``.

    The weight and bias are zero-initialized, so a head with the default
    threshold begins at the permanent anchor score of zero. Temperature and
    threshold are immutable calibration metadata rather than trainable
    parameters. This class does not compare scores or execute an expert.

    Args:
        feature_size: Size of the final router-feature dimension.
        layer_index: Zero-based decoder layer owned by the head.
        expert_id: Stable non-empty expert identifier.
        temperature: Finite positive score temperature.
        threshold: Finite activation threshold subtracted from the score.
        trainable: Whether the affine weight and bias require gradients.
    """

    architecture = _ROUTER_ARCHITECTURE

    def __init__(
        self,
        feature_size: int,
        *,
        layer_index: int,
        expert_id: str,
        temperature: float = 1.0,
        threshold: float = 0.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.feature_size = _validate_positive_integer(
            feature_size,
            field_name="feature_size",
        )
        self.layer_index = _validate_layer_index(layer_index)
        self.expert_id = _validate_expert_id(expert_id)
        self.temperature = _validate_temperature(temperature)
        self.threshold = _validate_finite_number(
            threshold,
            field_name="threshold",
        )
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")

        self.linear = nn.Linear(self.feature_size, 1, bias=True)
        # The default calibration gives a zero score matching the fixed-zero
        # anchor. Fitting strategies explicitly replace the affine values.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.requires_grad_(trainable)

    @property
    def key(self) -> tuple[int, str]:
        """Return the stable ``(layer_index, expert_id)`` ownership key."""

        return self.layer_index, self.expert_id

    def forward(self, features: Tensor) -> Tensor:
        """Return calibrated scores with shape ``features.shape[:-1]``.

        Args:
            features:
                Floating-point router features with shape
                ``[..., feature_size]``.

        Raises:
            TypeError: If ``features`` is not a floating-point tensor.
            ValueError: If its feature dimension is absent or incompatible.
        """

        _validate_features(features, feature_size=self.feature_size)
        if features.dtype != self.linear.weight.dtype:
            raise ValueError(
                "features dtype does not match the router head: "
                f"{features.dtype} != {self.linear.weight.dtype}."
            )
        if features.device != self.linear.weight.device:
            raise ValueError(
                "features device does not match the router head: "
                f"{features.device} != {self.linear.weight.device}."
            )
        raw_score = self.linear(features).squeeze(-1)
        return raw_score / self.temperature - self.threshold

    def extra_repr(self) -> str:
        """Describe ownership, dimensions, and immutable calibration values."""

        return (
            f"feature_size={self.feature_size}, layer_index={self.layer_index}, "
            f"expert_id={self.expert_id!r}, temperature={self.temperature}, "
            f"threshold={self.threshold}"
        )


class AppendOnlyRouterHeads(nn.Module):
    """Own affine heads in stable append order without replacement or removal.

    Heads are keyed by ``(layer_index, expert_id)`` and stored in global append
    order. Scoring one layer filters that order and stacks one scalar per head
    in the final tensor dimension. Appending validates uniqueness and feature
    compatibility before registering the new module, so existing head objects,
    parameters, and scores are untouched.

    Args:
        feature_size: Shared frozen-feature dimension for every appended head.
    """

    architecture = _ROUTER_ARCHITECTURE

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.feature_size = _validate_positive_integer(
            feature_size,
            field_name="feature_size",
        )
        self.heads = nn.ModuleList()
        self._head_keys: set[tuple[int, str]] = set()

    def __len__(self) -> int:
        """Return the number of appended layer-expert heads."""

        return len(self.heads)

    @property
    def head_keys(self) -> tuple[tuple[int, str], ...]:
        """Return all ownership keys in immutable append order."""

        return tuple(head.key for head in self.heads)

    def expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return expert identifiers for one layer in append order."""

        normalized_layer = _validate_layer_index(layer_index)
        return tuple(
            head.expert_id
            for head in self.heads
            if head.layer_index == normalized_layer
        )

    def append(self, head: AffineRouterHead) -> None:
        """Append one compatible, uniquely owned affine head.

        Raises:
            TypeError: If ``head`` is not an :class:`AffineRouterHead`.
            ValueError: If its feature size differs or ownership key exists.
        """

        if not isinstance(head, AffineRouterHead):
            raise TypeError("head must be an AffineRouterHead.")
        if head.feature_size != self.feature_size:
            raise ValueError(
                "Router head feature size does not match the bank: "
                f"{head.feature_size} != {self.feature_size}."
            )
        if head.key in self._head_keys:
            raise ValueError(
                "Router head ownership key is already installed: "
                f"{head.key!r}."
            )
        if self.heads:
            incumbent_weight = self.heads[0].linear.weight
            candidate_weight = head.linear.weight
            if candidate_weight.dtype != incumbent_weight.dtype:
                raise ValueError(
                    "Router heads must share one dtype: "
                    f"{candidate_weight.dtype} != {incumbent_weight.dtype}."
                )
            if candidate_weight.device != incumbent_weight.device:
                raise ValueError(
                    "Router heads must share one device: "
                    f"{candidate_weight.device} != {incumbent_weight.device}."
                )

        self.heads.append(head)
        self._head_keys.add(head.key)

    def forward(self, features: Tensor, *, layer_index: int) -> Tensor:
        """Score every head at one layer in deterministic append order.

        The result has shape ``features.shape[:-1] + (K_l,)``, where ``K_l``
        is the number of heads registered for ``layer_index``.

        Raises:
            LookupError: If the layer has no registered heads.
        """

        _validate_features(features, feature_size=self.feature_size)
        normalized_layer = _validate_layer_index(layer_index)
        selected_heads = [
            head for head in self.heads if head.layer_index == normalized_layer
        ]
        if not selected_heads:
            raise LookupError(
                f"No affine router heads are installed for layer {normalized_layer}."
            )
        return torch.stack([head(features) for head in selected_heads], dim=-1)

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize ordered router heads without executable content.

        The payload stores affine tensors plus a canonical JSON metadata
        envelope containing the format, feature size, append order, ownership,
        temperature, and threshold of every head.

        Args:
            path: Destination ending in ``.safetensors`` with an existing parent.
            overwrite: Whether an existing regular file may be replaced.

        Raises:
            RouterSerializationError:
                If the path or router state is unsafe or cannot be serialized.
            FileExistsError:
                If the destination exists and ``overwrite`` is false.
        """

        if not self.heads:
            raise RouterSerializationError(
                "Cannot serialize an empty affine router-head bank."
            )
        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state = _portable_finite_state(self.state_dict())
        metadata_values = {
            "architecture": self.architecture,
            "feature_size": self.feature_size,
            "format": _SERIALIZATION_FORMAT,
            "heads": [
                {
                    "expert_id": head.expert_id,
                    "layer_index": head.layer_index,
                    "temperature": head.temperature,
                    "threshold": head.threshold,
                }
                for head in self.heads
            ],
        }
        metadata = {
            _METADATA_KEY: json.dumps(
                metadata_values,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
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
            raise RouterSerializationError(
                f"Cannot save affine router heads to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        trainable: bool = False,
    ) -> AppendOnlyRouterHeads:
        """Strictly load an ordered affine-head bank from safetensors.

        Metadata, ownership uniqueness, tensor keys, shapes, dtypes, and finite
        tensor values are validated before the bank is returned. Loading never
        imports or executes package-supplied Python.

        Args:
            path: Existing regular ``.safetensors`` payload.
            trainable: Whether loaded affine parameters require gradients.

        Raises:
            RouterSerializationError:
                If the payload metadata or tensors violate the format contract.
        """

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        source = _validate_load_path(path)
        metadata = _load_and_validate_metadata(source)

        try:
            saved_state = load_file(source, device="cpu")
            _validate_finite_state(saved_state)
            bank = cls(metadata["feature_size"])
            for index, descriptor in enumerate(metadata["heads"]):
                weight_name = f"heads.{index}.linear.weight"
                weight = saved_state.get(weight_name)
                if weight is None or not weight.is_floating_point():
                    raise RouterSerializationError(
                        f"Affine router weight {weight_name!r} is missing or non-floating."
                    )
                head = AffineRouterHead(
                    bank.feature_size,
                    layer_index=descriptor["layer_index"],
                    expert_id=descriptor["expert_id"],
                    temperature=descriptor["temperature"],
                    threshold=descriptor["threshold"],
                    trainable=trainable,
                ).to(dtype=weight.dtype)
                bank.append(head)

            _validate_state_schema(bank.state_dict(), saved_state)
            bank.load_state_dict(saved_state, strict=True)
        except RouterSerializationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RouterSerializationError(
                f"Cannot load affine router heads from {source}: {error}"
            ) from error

        bank.requires_grad_(trainable)
        return bank


def _validate_features(features: Tensor, *, feature_size: int) -> None:
    """Validate the common feature tensor contract without changing storage."""

    if not isinstance(features, Tensor):
        raise TypeError("features must be a torch.Tensor.")
    if not features.is_floating_point():
        raise TypeError("features must use a floating-point dtype.")
    if features.ndim < 1:
        raise ValueError("features must have a final feature dimension.")
    if features.shape[-1] != feature_size:
        raise ValueError(
            "features final dimension does not match the router head: "
            f"{features.shape[-1]} != {feature_size}."
        )


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _validate_layer_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("layer_index must be a non-negative integer.")
    return value


def _validate_expert_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "expert_id must be a non-empty string without surrounding whitespace."
        )
    return value


def _validate_finite_number(value: Any, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite number.")
    return float(value)


def _validate_temperature(value: Any) -> float:
    temperature = _validate_finite_number(value, field_name="temperature")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    return temperature


def _validate_load_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise RouterSerializationError(
            f"Router payload must end in {_SAFETENSORS_SUFFIX}: {source}"
        )
    if source.is_symlink():
        raise RouterSerializationError(
            f"Router payload must not be a symbolic link: {source}"
        )
    if not source.is_file():
        raise RouterSerializationError(
            f"Router payload is not a regular file: {source}"
        )
    return source


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise RouterSerializationError(
            f"Router payload must end in {_SAFETENSORS_SUFFIX}: {destination}"
        )
    if destination.is_symlink():
        raise RouterSerializationError(
            f"Router payload must not be a symbolic link: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Router payload already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise RouterSerializationError(
            f"Router payload destination is not a regular file: {destination}"
        )
    if not destination.parent.is_dir():
        raise RouterSerializationError(
            f"Router payload parent directory does not exist: {destination.parent}"
        )
    return destination


def _portable_finite_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    portable: dict[str, Tensor] = {}
    for name, value in sorted(state.items()):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise RouterSerializationError(
                f"Affine router state {name!r} must be a floating-point tensor."
            )
        detached = value.detach().cpu().contiguous()
        if not torch.isfinite(detached).all().item():
            raise RouterSerializationError(
                f"Affine router state {name!r} contains non-finite values."
            )
        portable[name] = detached
    return portable


def _load_and_validate_metadata(source: Path) -> dict[str, Any]:
    try:
        with safe_open(source, framework="pt", device="cpu") as payload:
            serialized_metadata = (payload.metadata() or {}).get(_METADATA_KEY)
    except Exception as error:
        raise RouterSerializationError(
            f"Cannot read affine router payload {source}: {error}"
        ) from error

    try:
        metadata = json.loads(serialized_metadata)
    except (TypeError, json.JSONDecodeError) as error:
        raise RouterSerializationError(
            f"Invalid affine router metadata in {source}."
        ) from error
    if not isinstance(metadata, dict):
        raise RouterSerializationError(
            f"Invalid affine router metadata in {source}."
        )
    expected_keys = {"architecture", "feature_size", "format", "heads"}
    if set(metadata) != expected_keys:
        raise RouterSerializationError(
            "Affine router metadata fields are incompatible: "
            f"expected={sorted(expected_keys)}, actual={sorted(metadata)}."
        )
    if metadata["format"] != _SERIALIZATION_FORMAT:
        raise RouterSerializationError(
            f"Unsupported affine router format: {metadata['format']!r}."
        )
    if metadata["architecture"] != _ROUTER_ARCHITECTURE:
        raise RouterSerializationError(
            f"Unsupported router architecture: {metadata['architecture']!r}."
        )
    feature_size = _validate_positive_integer(
        metadata["feature_size"],
        field_name="feature_size",
    )
    raw_heads = metadata["heads"]
    if not isinstance(raw_heads, list) or not raw_heads:
        raise RouterSerializationError(
            "Affine router metadata must contain at least one head."
        )

    descriptors: list[dict[str, Any]] = []
    ownership_keys: set[tuple[int, str]] = set()
    descriptor_keys = {"expert_id", "layer_index", "temperature", "threshold"}
    try:
        for raw_descriptor in raw_heads:
            if not isinstance(raw_descriptor, dict) or set(raw_descriptor) != descriptor_keys:
                raise RouterSerializationError(
                    "Affine router head metadata fields are incompatible."
                )
            descriptor = {
                "expert_id": _validate_expert_id(raw_descriptor["expert_id"]),
                "layer_index": _validate_layer_index(raw_descriptor["layer_index"]),
                "temperature": _validate_temperature(
                    raw_descriptor["temperature"]
                ),
                "threshold": _validate_finite_number(
                    raw_descriptor["threshold"],
                    field_name="threshold",
                ),
            }
            key = descriptor["layer_index"], descriptor["expert_id"]
            if key in ownership_keys:
                raise RouterSerializationError(
                    f"Duplicate affine router ownership key: {key!r}."
                )
            ownership_keys.add(key)
            descriptors.append(descriptor)
    except RouterSerializationError:
        raise
    except (TypeError, ValueError) as error:
        raise RouterSerializationError(
            f"Invalid affine router head metadata in {source}: {error}"
        ) from error

    return {
        "architecture": _ROUTER_ARCHITECTURE,
        "feature_size": feature_size,
        "format": _SERIALIZATION_FORMAT,
        "heads": descriptors,
    }


def _validate_finite_state(state: dict[str, Tensor]) -> None:
    for name, value in state.items():
        if not value.is_floating_point():
            raise RouterSerializationError(
                f"Affine router tensor {name!r} must be floating-point."
            )
        if not torch.isfinite(value).all().item():
            raise RouterSerializationError(
                f"Affine router tensor {name!r} contains non-finite values."
            )


def _validate_state_schema(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
) -> None:
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RouterSerializationError(
            "Affine router state keys are incompatible: "
            f"missing={missing}, unexpected={unexpected}."
        )

    mismatches: list[str] = []
    for name in sorted(expected):
        expected_tensor = expected[name]
        actual_tensor = actual[name]
        if expected_tensor.shape != actual_tensor.shape:
            mismatches.append(
                f"{name}: shape {tuple(actual_tensor.shape)} != "
                f"{tuple(expected_tensor.shape)}"
            )
        if expected_tensor.dtype != actual_tensor.dtype:
            mismatches.append(
                f"{name}: dtype {actual_tensor.dtype} != {expected_tensor.dtype}"
            )
    if mismatches:
        raise RouterSerializationError(
            "Affine router tensor schema is incompatible: " + "; ".join(mismatches)
        )
