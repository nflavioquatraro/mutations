"""Full copied-MLP expert representation and safe tensor serialization."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from anchored_moe.experts.base import Expert


_SERIALIZATION_FORMAT: Final = "anchored_moe.full_mlp.v1"
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"


class ExpertSerializationError(ValueError):
    """Raised when a full-MLP tensor payload is unsafe or incompatible."""


class FullMLPExpert(Expert):
    """Independent full-capacity copy of one immutable anchor MLP.

    For anchor transformation ``A_l``, construction initializes the expert
    exactly as ``E_(l,k) <- A_l``. The copied parameters and
    buffers occupy independent storage, so later optimization of ``E_(l,k)``
    cannot mutate ``A_l``. Construction freezes the supplied anchor
    mechanically and makes every copied parameter trainable by default.

    Args:
        anchor_mlp:
            Dense MLP to copy. The supplied module is frozen after a successful
            copy and is never retained as expert storage.
        trainable:
            Whether every expert parameter should require gradients.

    Hidden states normally have shape ``[..., hidden_size]``. Output shape and
    additional arguments follow the copied module's forward contract.

    This representation is infrastructure, not a paper-derived algorithm.
    It does not attach itself to a routed layer or select when it executes.
    """

    parameterization = "full_mlp"

    def __init__(
        self,
        anchor_mlp: nn.Module,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")

        # Copy before freezing so a copy failure has no side effect and the
        # caller's prior gradient flags cannot leak into the expert.
        try:
            expert_mlp = deepcopy(anchor_mlp)
        except Exception as error:
            raise RuntimeError(
                f"Cannot copy anchor module {type(anchor_mlp).__qualname__}."
            ) from error
        anchor_mlp.requires_grad_(False)
        self.expert_mlp = expert_mlp
        self.expert_mlp.requires_grad_(trainable)
        # The wrapper and copied child begin in the same training/evaluation
        # mode as the anchor so exact initialization is observable immediately.
        self.train(anchor_mlp.training)

    @classmethod
    def from_anchor(
        cls,
        anchor_mlp: nn.Module,
        *,
        trainable: bool = True,
    ) -> FullMLPExpert:
        """Copy one anchor MLP, freeze the anchor, and return its expert.

        Args:
            anchor_mlp:
                Dense MLP to copy without changing tensor values.
            trainable:
                Whether copied parameters should require gradients.

        Raises:
            TypeError:
                If ``anchor_mlp`` is not a module or ``trainable`` is not a
                boolean.
            RuntimeError:
                If the module cannot be deep-copied.
        """

        return cls(anchor_mlp, trainable=trainable)

    @classmethod
    def from_safetensors(
        cls,
        anchor_mlp: nn.Module,
        path: str | Path,
        *,
        trainable: bool = True,
    ) -> FullMLPExpert:
        """Initialize from an anchor and strictly load a saved expert payload.

        The payload format marker, parameterization, source module class,
        state-dict keys, shapes, and dtypes must match. Only a compatible
        payload causes ``anchor_mlp`` to be copied and frozen; payload tensors
        are never written into the anchor.

        Args:
            anchor_mlp: Compatible dense MLP supplying the runtime structure.
            path: Existing regular ``.safetensors`` file.
            trainable: Whether loaded expert parameters require gradients.

        Raises:
            ExpertSerializationError:
                If the path, metadata, tensor schema, or tensor values cannot be
                loaded safely into the copied module.
        """

        source = _validate_load_path(path)
        expected_module_class = _qualified_class_name(anchor_mlp)

        try:
            with safe_open(source, framework="pt", device="cpu") as payload:
                file_metadata = payload.metadata() or {}
        except Exception as error:
            raise ExpertSerializationError(
                f"Cannot read safetensors payload {source}: {error}"
            ) from error

        serialized_metadata = file_metadata.get(_METADATA_KEY)
        try:
            metadata = json.loads(serialized_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ExpertSerializationError(
                f"Invalid full-MLP metadata in {source}."
            ) from error
        if not isinstance(metadata, dict):
            raise ExpertSerializationError(
                f"Invalid full-MLP metadata in {source}."
            )

        expected_metadata = {
            "format": _SERIALIZATION_FORMAT,
            "parameterization": cls.parameterization,
            "module_class": expected_module_class,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ExpertSerializationError(
                f"Incompatible full-MLP metadata in {source}: {mismatches}"
            )

        try:
            saved_state = load_file(source, device="cpu")
            expected_state = {
                f"expert_mlp.{name}": tensor
                for name, tensor in anchor_mlp.state_dict().items()
            }
            _validate_state_schema(expected_state, saved_state)
            expert = cls.from_anchor(anchor_mlp, trainable=trainable)
            expert.load_state_dict(saved_state, strict=True)
        except ExpertSerializationError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot load full-MLP tensors from {source}: {error}"
            ) from error

        # load_state_dict preserves requires_grad, but the explicit assignment
        # keeps the public contract stable if PyTorch behavior changes.
        expert.requires_grad_(trainable)
        return expert

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute the copied MLP without anchor or routing arithmetic."""

        return self.expert_mlp(hidden_states, *args, **kwargs)

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize expert state without executable Python content.

        The file records a format marker, the ``full_mlp`` parameterization,
        and the copied module's fully qualified class name. Tensors are detached
        and moved to contiguous CPU storage for portable safetensors output.

        Args:
            path: Destination ending in ``.safetensors``. Its parent must exist.
            overwrite: Whether an existing regular file may be atomically
                replaced. The default protects existing payloads.

        Raises:
            ExpertSerializationError:
                If the destination is unsafe or serialization fails.
            FileExistsError:
                If the destination exists and ``overwrite`` is false.
        """

        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state: dict[str, Tensor] = {}
        for name, value in sorted(self.state_dict().items()):
            if not isinstance(value, Tensor):
                raise ExpertSerializationError(
                    f"Full-MLP state {name!r} is not a tensor."
                )
            tensor_state[name] = value.detach().cpu().contiguous()
        metadata_values = {
            "format": _SERIALIZATION_FORMAT,
            "parameterization": self.parameterization,
            "module_class": _qualified_class_name(self.expert_mlp),
        }
        # A single canonical JSON envelope avoids relying on the safetensors
        # writer's metadata-map ordering, which is not byte deterministic.
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
            raise ExpertSerializationError(
                f"Cannot save full-MLP tensors to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _qualified_class_name(module: nn.Module) -> str:
    """Return a stable runtime architecture identifier for metadata checks."""

    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _validate_load_path(path: str | Path) -> Path:
    """Resolve the non-symlink safetensors path accepted by the loader."""

    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise ExpertSerializationError(
            f"Expert payload must end in {_SAFETENSORS_SUFFIX}: {source}"
        )
    if source.is_symlink():
        raise ExpertSerializationError(
            f"Expert payload must not be a symbolic link: {source}"
        )
    if not source.is_file():
        raise ExpertSerializationError(
            f"Expert payload is not a regular file: {source}"
        )
    return source


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    """Validate a destination without creating directories implicitly."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise ExpertSerializationError(
            f"Expert payload must end in {_SAFETENSORS_SUFFIX}: {destination}"
        )
    if destination.is_symlink():
        raise ExpertSerializationError(
            f"Expert payload must not be a symbolic link: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Expert payload already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise ExpertSerializationError(
            f"Expert payload destination is not a regular file: {destination}"
        )
    if not destination.parent.is_dir():
        raise ExpertSerializationError(
            f"Expert payload parent directory does not exist: {destination.parent}"
        )
    return destination


def _validate_state_schema(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
) -> None:
    """Reject key, shape, or dtype mismatches before mutating the expert."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ExpertSerializationError(
            "Full-MLP state keys are incompatible: "
            f"missing={missing}, unexpected={unexpected}"
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
        raise ExpertSerializationError(
            "Full-MLP tensor schema is incompatible: " + "; ".join(mismatches)
        )
