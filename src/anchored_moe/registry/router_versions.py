"""Safe registry-specific persistence for recalibrated logical routers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from anchored_moe.calibration.global_recalibration import (
    GlobalRouterRecalibrationReport,
)
from anchored_moe.registry.package import PackageIdentity
from anchored_moe.registry.registry import RegistrySnapshot
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads
from anchored_moe.routers.logical import (
    LogicalAppendOnlyRouter,
    compile_logical_router,
)
from anchored_moe.utils.hashing import canonical_json_bytes, sha256_json


REGISTRY_ROUTER_VERSION_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = "anchored_moe.registry_router_version.v1"
_METADATA_KEY: Final = "anchored_moe"
_SAFETENSORS_SUFFIX: Final = ".safetensors"


class RegistryRouterVersionError(ValueError):
    """Raised when a PKG-005 router version is unsafe or incompatible."""


class RegistryRouterVersion:
    """Immutable RTR-004 router bound to one exact active-registry snapshot.

    PKG-005 stores affine tensors in safetensors and all ownership,
    compatibility, feature-space, registry, and recalibration metadata in one
    strict canonical JSON envelope. The artifact owns a defensive frozen copy
    of the router. :attr:`router` and :meth:`activate` return fresh logical
    routers, so callers cannot mutate the persisted in-memory version.

    Activation is explicit and read-only: it validates the complete current
    :class:`RegistrySnapshot` and returns a router but never changes an
    :class:`ExpertRegistry`, model, package, or filesystem location.
    """

    format_version: ClassVar[int] = REGISTRY_ROUTER_VERSION_FORMAT_VERSION
    serialization_format: ClassVar[str] = _SERIALIZATION_FORMAT

    def __init__(
        self,
        router: LogicalAppendOnlyRouter,
        *,
        registry_snapshot: RegistrySnapshot,
        feature_specification: Mapping[str, Any],
        recalibration_report: GlobalRouterRecalibrationReport,
    ) -> None:
        """Copy and validate one frozen registry-specific router version."""

        if not isinstance(router, LogicalAppendOnlyRouter):
            raise TypeError("router must be a LogicalAppendOnlyRouter.")
        if not isinstance(registry_snapshot, RegistrySnapshot):
            raise TypeError("registry_snapshot must be a RegistrySnapshot.")
        if not registry_snapshot.active_packages:
            raise ValueError(
                "registry_snapshot must contain at least one active package."
            )
        if registry_snapshot.compatibility_sha256 is None:
            raise ValueError(
                "registry_snapshot must have a compatibility identity."
            )
        if not isinstance(
            recalibration_report,
            GlobalRouterRecalibrationReport,
        ):
            raise TypeError(
                "recalibration_report must be a "
                "GlobalRouterRecalibrationReport."
            )
        if (
            recalibration_report.registry_sha256
            != registry_snapshot.registry_sha256
        ):
            raise ValueError(
                "recalibration report is bound to a different registry hash."
            )
        if (
            recalibration_report.compatibility_sha256
            != registry_snapshot.compatibility_sha256
        ):
            raise ValueError(
                "recalibration report is bound to a different compatibility "
                "identity."
            )

        normalized_specification = _normalize_feature_specification(
            feature_specification,
            feature_size=router.feature_size,
        )
        heads = router.to_independent_heads(trainable=False)
        if not heads.heads:
            raise ValueError("router must contain at least one expert head.")
        for head in heads.heads:
            if head.linear.weight.device.type != "cpu":
                raise ValueError(
                    "Registry router versions require CPU tensor state."
                )
            for tensor in head.linear.state_dict().values():
                if (
                    not tensor.is_floating_point()
                    or not torch.isfinite(tensor).all().item()
                ):
                    raise ValueError(
                        "Registry router tensors must be finite floating-point."
                    )
        expected_layer_reports = {
            report.layer_index: report.expert_ids
            for report in recalibration_report.layer_reports
        }
        actual_layer_reports = {
            layer_index: router.expert_ids_for_layer(layer_index)
            for layer_index in sorted(
                {layer_index for layer_index, _ in router.head_keys}
            )
        }
        if expected_layer_reports != actual_layer_reports:
            raise ValueError(
                "recalibration report layer/expert order does not match router."
            )

        self._heads = heads
        self._registry_snapshot = registry_snapshot
        self._feature_specification_json = canonical_json_bytes(
            normalized_specification
        ).decode("utf-8")
        self._recalibration_report = recalibration_report

    @property
    def registry_snapshot(self) -> RegistrySnapshot:
        """Return the immutable exact active-registry binding."""

        return self._registry_snapshot

    @property
    def feature_specification(self) -> dict[str, Any]:
        """Return a defensive JSON copy of the router feature specification."""

        return json.loads(self._feature_specification_json)

    @property
    def recalibration_report(self) -> GlobalRouterRecalibrationReport:
        """Return the immutable RTR-004 report stored by this version."""

        return self._recalibration_report

    @property
    def router(self) -> LogicalAppendOnlyRouter:
        """Return a fresh frozen logical router with independent storage."""

        return compile_logical_router(_clone_heads(self._heads))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return strict canonical metadata including tensor integrity hashes."""

        return {
            "feature_specification": self.feature_specification,
            "format": self.serialization_format,
            "format_version": self.format_version,
            "heads": [
                {
                    "bias_sha256": _sha256_tensor(head.linear.bias),
                    "bias_tensor": _bias_tensor_name(index),
                    "dtype": _dtype_name(head.linear.weight.dtype),
                    "expert_id": head.expert_id,
                    "layer_index": head.layer_index,
                    "temperature": head.temperature,
                    "threshold": head.threshold,
                    "weight_sha256": _sha256_tensor(head.linear.weight),
                    "weight_tensor": _weight_tensor_name(index),
                }
                for index, head in enumerate(self._heads.heads)
            ],
            "recalibration_report": self._recalibration_report.to_dict(),
            "registry_snapshot": _snapshot_to_dict(self._registry_snapshot),
            "router_architecture": LogicalAppendOnlyRouter.architecture,
            "router_feature_size": self._heads.feature_size,
        }

    @property
    def metadata_sha256(self) -> str:
        """Return the content identity of canonical artifact metadata."""

        return sha256_json(self.metadata)

    def activate(
        self,
        registry_snapshot: RegistrySnapshot,
    ) -> LogicalAppendOnlyRouter:
        """Return a fresh router only for the exact bound registry snapshot.

        This method deliberately performs no model or registry mutation.
        Callers must explicitly install the returned score bank into their
        runtime after all other compatibility checks have passed.
        """

        if not isinstance(registry_snapshot, RegistrySnapshot):
            raise TypeError("registry_snapshot must be a RegistrySnapshot.")
        if registry_snapshot != self._registry_snapshot:
            raise RegistryRouterVersionError(
                "Registry router version does not match the active registry "
                "snapshot."
            )
        return self.router

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save tensor state and strict registry-bound metadata."""

        destination = _validate_save_path(path, overwrite=overwrite)
        tensors: dict[str, Tensor] = {}
        for index, head in enumerate(self._heads.heads):
            tensors[_weight_tensor_name(index)] = (
                head.linear.weight.detach().clone().contiguous()
            )
            tensors[_bias_tensor_name(index)] = (
                head.linear.bias.detach().clone().contiguous()
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
            save_file(tensors, temporary_path, metadata=metadata)
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError, ValueError) as error:
            raise RegistryRouterVersionError(
                f"Cannot save registry router version to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def from_safetensors(cls, path: str | Path) -> RegistryRouterVersion:
        """Strictly load, verify, and reconstruct a PKG-005 router version."""

        source = _validate_load_path(path)
        try:
            with safe_open(source, framework="pt", device="cpu") as handle:
                envelope = handle.metadata()
            if not isinstance(envelope, dict) or set(envelope) != {_METADATA_KEY}:
                raise RegistryRouterVersionError(
                    "Registry router metadata envelope is missing or invalid."
                )
            metadata = json.loads(envelope[_METADATA_KEY])
            _validate_top_level_metadata(metadata)
            tensors = load_file(source, device="cpu")
            snapshot = _snapshot_from_dict(metadata["registry_snapshot"])
            report = GlobalRouterRecalibrationReport.from_dict(
                metadata["recalibration_report"]
            )
            heads = _heads_from_payload(
                metadata["heads"],
                tensors,
                feature_size=metadata["router_feature_size"],
            )
            artifact = cls(
                compile_logical_router(heads),
                registry_snapshot=snapshot,
                feature_specification=metadata["feature_specification"],
                recalibration_report=report,
            )
            if artifact.metadata != metadata:
                raise RegistryRouterVersionError(
                    "Registry router metadata is not canonical or does not "
                    "match reconstructed state."
                )
            return artifact
        except RegistryRouterVersionError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RegistryRouterVersionError(
                f"Cannot load registry router version from {source}: {error}"
            ) from error


def _clone_heads(source: AppendOnlyRouterHeads) -> AppendOnlyRouterHeads:
    """Return frozen CPU heads without source-storage aliases."""

    clone = AppendOnlyRouterHeads(source.feature_size)
    for source_head in source.heads:
        head = AffineRouterHead(
            source.feature_size,
            layer_index=source_head.layer_index,
            expert_id=source_head.expert_id,
            temperature=source_head.temperature,
            threshold=source_head.threshold,
            trainable=False,
        ).to(dtype=source_head.linear.weight.dtype, device="cpu")
        with torch.no_grad():
            head.linear.weight.copy_(source_head.linear.weight)
            head.linear.bias.copy_(source_head.linear.bias)
        clone.append(head)
    return clone


def _heads_from_payload(
    descriptors: Any,
    tensors: dict[str, Tensor],
    *,
    feature_size: Any,
) -> AppendOnlyRouterHeads:
    """Validate exact tensor schema and reconstruct frozen affine heads."""

    normalized_feature_size = _validate_positive_integer(
        feature_size,
        field_name="router_feature_size",
    )
    if not isinstance(descriptors, list) or not descriptors:
        raise RegistryRouterVersionError(
            "Registry router heads metadata must be a non-empty list."
        )
    expected_tensor_names: set[str] = set()
    heads = AppendOnlyRouterHeads(normalized_feature_size)
    expected_fields = {
        "bias_sha256",
        "bias_tensor",
        "dtype",
        "expert_id",
        "layer_index",
        "temperature",
        "threshold",
        "weight_sha256",
        "weight_tensor",
    }
    for index, descriptor in enumerate(descriptors):
        _require_exact_fields(
            descriptor,
            expected_fields,
            field_name=f"heads[{index}]",
        )
        weight_name = _weight_tensor_name(index)
        bias_name = _bias_tensor_name(index)
        if (
            descriptor["weight_tensor"] != weight_name
            or descriptor["bias_tensor"] != bias_name
        ):
            raise RegistryRouterVersionError(
                "Registry router tensor names do not match head order."
            )
        expected_tensor_names.update((weight_name, bias_name))
        weight = tensors.get(weight_name)
        bias = tensors.get(bias_name)
        if weight is None or bias is None:
            raise RegistryRouterVersionError(
                f"Registry router tensors are missing for head {index}."
            )
        if (
            not weight.is_floating_point()
            or not bias.is_floating_point()
            or weight.shape != (1, normalized_feature_size)
            or bias.shape != (1,)
            or weight.dtype != bias.dtype
            or not torch.isfinite(weight).all().item()
            or not torch.isfinite(bias).all().item()
        ):
            raise RegistryRouterVersionError(
                f"Registry router tensor state is invalid for head {index}."
            )
        if descriptor["dtype"] != _dtype_name(weight.dtype):
            raise RegistryRouterVersionError(
                f"Registry router dtype metadata is invalid for head {index}."
            )
        if (
            descriptor["weight_sha256"] != _sha256_tensor(weight)
            or descriptor["bias_sha256"] != _sha256_tensor(bias)
        ):
            raise RegistryRouterVersionError(
                f"Registry router tensor digest mismatch for head {index}."
            )
        head = AffineRouterHead(
            normalized_feature_size,
            layer_index=descriptor["layer_index"],
            expert_id=descriptor["expert_id"],
            temperature=descriptor["temperature"],
            threshold=descriptor["threshold"],
            trainable=False,
        ).to(dtype=weight.dtype, device="cpu")
        with torch.no_grad():
            head.linear.weight.copy_(weight)
            head.linear.bias.copy_(bias)
        heads.append(head)
    if set(tensors) != expected_tensor_names:
        raise RegistryRouterVersionError(
            "Registry router tensor keys do not exactly match metadata."
        )
    return heads


def _validate_top_level_metadata(value: Any) -> None:
    expected = {
        "feature_specification",
        "format",
        "format_version",
        "heads",
        "recalibration_report",
        "registry_snapshot",
        "router_architecture",
        "router_feature_size",
    }
    _require_exact_fields(value, expected, field_name="registry router metadata")
    if value["format"] != _SERIALIZATION_FORMAT:
        raise RegistryRouterVersionError(
            "Unsupported registry router serialization format."
        )
    if value["format_version"] != REGISTRY_ROUTER_VERSION_FORMAT_VERSION:
        raise RegistryRouterVersionError(
            "Unsupported registry router format version."
        )
    if value["router_architecture"] != LogicalAppendOnlyRouter.architecture:
        raise RegistryRouterVersionError(
            "Unsupported registry router architecture."
        )


def _snapshot_to_dict(snapshot: RegistrySnapshot) -> dict[str, Any]:
    return {
        "active_packages": [
            package.to_dict() for package in snapshot.active_packages
        ],
        "compatibility_sha256": snapshot.compatibility_sha256,
        "registry_format_version": snapshot.registry_format_version,
        "registry_sha256": snapshot.registry_sha256,
    }


def _snapshot_from_dict(value: Any) -> RegistrySnapshot:
    expected = {
        "active_packages",
        "compatibility_sha256",
        "registry_format_version",
        "registry_sha256",
    }
    _require_exact_fields(value, expected, field_name="registry snapshot")
    raw_packages = value["active_packages"]
    if not isinstance(raw_packages, list):
        raise RegistryRouterVersionError(
            "registry snapshot active_packages must be a list."
        )
    packages: list[PackageIdentity] = []
    package_fields = {
        "manifest_sha256",
        "specialization_name",
        "specialization_version",
    }
    for index, package in enumerate(raw_packages):
        _require_exact_fields(
            package,
            package_fields,
            field_name=f"active_packages[{index}]",
        )
        packages.append(PackageIdentity(**package))
    return RegistrySnapshot(
        registry_format_version=value["registry_format_version"],
        registry_sha256=value["registry_sha256"],
        compatibility_sha256=value["compatibility_sha256"],
        active_packages=tuple(packages),
    )


def _normalize_feature_specification(
    value: Mapping[str, Any],
    *,
    feature_size: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("feature_specification must be a mapping.")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "feature_specification must be finite JSON-compatible metadata."
        ) from error
    if not isinstance(normalized, dict):
        raise ValueError("feature_specification must encode a JSON object.")
    if normalized.get("output_size") != feature_size:
        raise ValueError(
            "feature_specification output_size must match router feature size."
        )
    return normalized


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise RegistryRouterVersionError(
            "Registry router destination must end in .safetensors."
        )
    if not destination.parent.is_dir():
        raise RegistryRouterVersionError(
            "Registry router destination parent must exist."
        )
    if destination.is_symlink():
        raise RegistryRouterVersionError(
            "Registry router destination must not be a symbolic link."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise RegistryRouterVersionError(
            "Registry router destination must be a regular file."
        )
    return destination


def _validate_load_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise RegistryRouterVersionError(
            "Registry router source must end in .safetensors."
        )
    if source.is_symlink() or not source.is_file():
        raise RegistryRouterVersionError(
            "Registry router source must be a regular non-symlink file."
        )
    return source


def _require_exact_fields(
    value: Any,
    expected: set[str],
    *,
    field_name: str,
) -> None:
    if not isinstance(value, dict):
        raise RegistryRouterVersionError(f"{field_name} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise RegistryRouterVersionError(
            f"{field_name} fields do not match the schema: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryRouterVersionError(
            f"{field_name} must be a positive integer."
        )
    return value


def _weight_tensor_name(index: int) -> str:
    return f"heads.{index}.weight"


def _bias_tensor_name(index: int) -> str:
    return f"heads.{index}.bias"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sha256_tensor(tensor: Tensor) -> str:
    """Hash dtype, shape, and exact contiguous CPU bytes."""

    portable = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_dtype_name(portable.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(portable.shape)))
    digest.update(portable.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
