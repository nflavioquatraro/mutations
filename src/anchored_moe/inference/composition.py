"""Explicit package materialization and anchored top-1 model synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from torch import Tensor, nn

from anchored_moe.backends.base import ModelBackend
from anchored_moe.experts.full_mlp import (
    ExpertSerializationError,
    FullMLPExpert,
)
from anchored_moe.modeling import RoutedMLP
from anchored_moe.registry.manifest import PackageValidationError
from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.registry.registry import ExpertRegistry
from anchored_moe.routers.features import IdentityRouterFeatureExtractor
from anchored_moe.routers.heads import (
    AppendOnlyRouterHeads,
    RouterSerializationError,
)
from anchored_moe.routers.logical import LogicalAppendOnlyRouter


class PackageMaterializationError(ValueError):
    """Raised when active package payloads cannot form one safe runtime."""


@dataclass(frozen=True)
class RegistryCompositionReport:
    """Auditable result of explicitly synchronizing a registry to a model.

    Attributes:
        registry_sha256: Active-registry identity applied to the model.
        active_packages: Package identities in runtime append order.
        configured_layers: Layers configured for anchored top-1 routing.
        reset_layers: Routed wrappers restored to exact forced-anchor mode.
        expert_ids_by_layer: Runtime expert order for each configured layer.
    """

    registry_sha256: str
    active_packages: tuple[PackageIdentity, ...]
    configured_layers: tuple[int, ...]
    reset_layers: tuple[int, ...]
    expert_ids_by_layer: tuple[tuple[int, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        """Validate deterministic layer ordering and report consistency."""

        if (
            not isinstance(self.registry_sha256, str)
            or len(self.registry_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.registry_sha256
            )
        ):
            raise ValueError(
                "registry_sha256 must be a lowercase 64-character digest."
            )
        if not isinstance(self.active_packages, tuple) or not all(
            isinstance(package, PackageIdentity)
            for package in self.active_packages
        ):
            raise TypeError("active_packages must be a tuple of PackageIdentity.")
        if tuple(sorted(self.configured_layers)) != self.configured_layers:
            raise ValueError("configured_layers must be sorted.")
        if tuple(sorted(self.reset_layers)) != self.reset_layers:
            raise ValueError("reset_layers must be sorted.")
        if set(self.configured_layers) & set(self.reset_layers):
            raise ValueError(
                "configured_layers and reset_layers must be disjoint."
            )
        reported_layers = tuple(layer for layer, _ in self.expert_ids_by_layer)
        if reported_layers != self.configured_layers:
            raise ValueError(
                "expert_ids_by_layer must cover configured_layers in order."
            )


@dataclass(frozen=True)
class _LayerRuntime:
    """Fully materialized candidate state for one routed wrapper."""

    layer_index: int
    wrapper: RoutedMLP
    feature_extractor: IdentityRouterFeatureExtractor
    router_heads: LogicalAppendOnlyRouter
    experts: dict[str, FullMLPExpert]


def synchronize_anchored_top1_registry(
    model: nn.Module,
    *,
    backend: ModelBackend,
    registry: ExpertRegistry,
) -> RegistryCompositionReport:
    """Explicitly materialize and apply the active anchored top-1 registry.

    Synchronization is deliberately separate from
    :meth:`~anchored_moe.registry.ExpertRegistry.install` and ``unload``.
    The operation revalidates live package files, parses explicit payload
    bindings, loads only safe supported tensor formats, verifies the complete
    layer/expert/router correspondence, and builds every candidate layer
    runtime before changing any wrapper.

    An empty active registry resets every discovered :class:`RoutedMLP` to its
    exact forced-anchor path. A non-empty registry currently supports only
    full copied-MLP experts, identity pre-MLP features, independent affine
    router heads, and anchored hard top-1 selection.

    Args:
        model: Already converted model containing routed MLP wrappers.
        backend: Model-family adapter used for deterministic layer discovery.
        registry: Active package registry to apply explicitly.

    Returns:
        Immutable composition report recording registry and runtime order.

    Raises:
        TypeError: If arguments do not implement their declared interfaces.
        PackageMaterializationError:
            If model structure, package metadata, tensors, or ownership keys
            are incomplete or incompatible. Validation failures do not alter
            any wrapper runtime.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must be a ModelBackend.")
    if not isinstance(registry, ExpertRegistry):
        raise TypeError("registry must be an ExpertRegistry.")

    try:
        references = tuple(backend.iter_mlp_references(model))
    except (TypeError, ValueError, RuntimeError) as error:
        raise PackageMaterializationError(
            f"Cannot discover model MLPs for composition: {error}"
        ) from error
    wrappers = {
        reference.layer_index: reference.module
        for reference in references
        if isinstance(reference.module, RoutedMLP)
    }
    if not wrappers:
        raise PackageMaterializationError(
            "Model has no converted RoutedMLP layers."
        )
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None and (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise PackageMaterializationError(
            "model.config.hidden_size must be a positive integer when present."
        )

    active_packages = registry.active_packages
    snapshot = registry.snapshot()
    if tuple(package.identity for package in active_packages) != (
        snapshot.active_packages
    ):
        raise PackageMaterializationError(
            "Registry changed while composition state was being captured."
        )
    if not active_packages:
        if registry.snapshot() != snapshot:
            raise PackageMaterializationError(
                "Registry changed before the empty state could be applied."
            )
        reset_layers = tuple(sorted(wrappers))
        for layer_index in reset_layers:
            wrappers[layer_index].reset_to_forced_anchor()
        return RegistryCompositionReport(
            registry_sha256=snapshot.registry_sha256,
            active_packages=(),
            configured_layers=(),
            reset_layers=reset_layers,
            expert_ids_by_layer=(),
        )

    layer_runtimes = _materialize_registry(
        active_packages,
        wrappers=wrappers,
        model_hidden_size=hidden_size,
    )
    if registry.snapshot() != snapshot:
        raise PackageMaterializationError(
            "Registry changed while package tensors were being materialized."
        )

    # Every package and layer has been fully validated and materialized. These
    # swaps only register already-prepared modules and cannot execute payload
    # code or perform tensor I/O.
    for runtime in layer_runtimes:
        runtime.wrapper.replace_anchored_top1(
            layer_index=runtime.layer_index,
            router_feature_extractor=runtime.feature_extractor,
            router_heads=runtime.router_heads,
            experts=runtime.experts,
        )

    configured_layers = tuple(runtime.layer_index for runtime in layer_runtimes)
    return RegistryCompositionReport(
        registry_sha256=snapshot.registry_sha256,
        active_packages=snapshot.active_packages,
        configured_layers=configured_layers,
        reset_layers=(),
        expert_ids_by_layer=tuple(
            (runtime.layer_index, runtime.wrapper.expert_ids)
            for runtime in layer_runtimes
        ),
    )


def _materialize_registry(
    packages: tuple[ExpertPackage, ...],
    *,
    wrappers: dict[int, RoutedMLP],
    model_hidden_size: int | None,
) -> tuple[_LayerRuntime, ...]:
    """Build every active layer runtime without mutating routed wrappers."""

    first_manifest = packages[0].manifest
    converted_layers = tuple(first_manifest.converted_layers)
    wrapper_layers = tuple(sorted(wrappers))
    if wrapper_layers != converted_layers:
        raise PackageMaterializationError(
            "Converted model layers do not match package compatibility: "
            f"model={wrapper_layers}, package={converted_layers}."
        )

    feature_size = _validate_identity_feature_specification(
        first_manifest.router_feature_specification
    )
    if model_hidden_size is not None and feature_size != model_hidden_size:
        raise PackageMaterializationError(
            "Router feature size does not match model.config.hidden_size: "
            f"{feature_size} != {model_hidden_size}."
        )
    layer_heads = {
        layer_index: AppendOnlyRouterHeads(feature_size)
        for layer_index in converted_layers
    }
    layer_experts: dict[int, dict[str, FullMLPExpert]] = {
        layer_index: {} for layer_index in converted_layers
    }

    for package in packages:
        _validate_supported_package(package)
        try:
            package.validate_live_files()
            bindings = package.payload_bindings
        except PackageValidationError as error:
            raise PackageMaterializationError(
                f"Package {package.identity.version_key!r} is not attachable: "
                f"{error}"
            ) from error

        package_feature_size = _validate_identity_feature_specification(
            package.manifest.router_feature_specification
        )
        if package_feature_size != feature_size:
            raise PackageMaterializationError(
                "Active packages disagree on router feature size."
            )

        binding_by_key = {binding.key: binding for binding in bindings}
        package_heads = _load_package_router_heads(package)
        head_keys = package_heads.head_keys
        if set(head_keys) != set(binding_by_key) or len(head_keys) != len(
            binding_by_key
        ):
            missing = sorted(set(binding_by_key) - set(head_keys))
            unexpected = sorted(set(head_keys) - set(binding_by_key))
            raise PackageMaterializationError(
                "Router heads must exactly match expert payload bindings for "
                f"{package.identity.version_key!r}: "
                f"missing={missing}, unexpected={unexpected}."
            )

        for head in package_heads.heads:
            layer_index, expert_id = head.key
            if expert_id in layer_experts[layer_index]:
                raise PackageMaterializationError(
                    "Active packages contain a duplicate expert ID at layer "
                    f"{layer_index}: {expert_id!r}."
                )

            wrapper = wrappers[layer_index]
            anchor_dtype, anchor_device = _module_dtype_and_device(
                wrapper.anchor_mlp,
                layer_index=layer_index,
            )
            declared_dtype = str(anchor_dtype).removeprefix("torch.")
            if package.manifest.base_model_dtype != declared_dtype:
                raise PackageMaterializationError(
                    "Package base_model_dtype does not match the live model at "
                    f"layer {layer_index}: "
                    f"{package.manifest.base_model_dtype!r} != "
                    f"{declared_dtype!r}."
                )
            if package.manifest.weight_dtype != declared_dtype:
                raise PackageMaterializationError(
                    "Package weight_dtype does not match the live anchor at "
                    f"layer {layer_index}: "
                    f"{package.manifest.weight_dtype!r} != "
                    f"{declared_dtype!r}."
                )
            if head.linear.weight.dtype != anchor_dtype:
                raise PackageMaterializationError(
                    "Router-head dtype must match the model hidden dtype at "
                    f"layer {layer_index}: {head.linear.weight.dtype} != "
                    f"{anchor_dtype}."
                )
            head.to(device=anchor_device)
            layer_heads[layer_index].append(head)

            binding = binding_by_key[head.key]
            payload_path = package.package_directory.joinpath(
                *PurePosixPath(binding.weight_file).parts
            )
            try:
                expert = FullMLPExpert.from_safetensors(
                    wrapper.anchor_mlp,
                    payload_path,
                    trainable=False,
                )
            except ExpertSerializationError as error:
                raise PackageMaterializationError(
                    "Cannot materialize expert payload for "
                    f"{head.key!r} from {binding.weight_file!r}: {error}"
                ) from error
            layer_experts[layer_index][expert_id] = expert

    runtimes: list[_LayerRuntime] = []
    for layer_index in converted_layers:
        heads = layer_heads[layer_index]
        experts = layer_experts[layer_index]
        if not heads.heads or set(heads.expert_ids_for_layer(layer_index)) != set(
            experts
        ):
            raise PackageMaterializationError(
                f"Layer {layer_index} has an incomplete composed runtime."
            )
        runtimes.append(
            _LayerRuntime(
                layer_index=layer_index,
                wrapper=wrappers[layer_index],
                feature_extractor=IdentityRouterFeatureExtractor(feature_size),
                # RTR-002 remains the validated package representation. The
                # active registry compiles it explicitly into the frozen
                # registry-level RTR-003 scoring runtime.
                router_heads=LogicalAppendOnlyRouter.from_independent_heads(
                    heads
                ),
                experts=experts,
            )
        )
    return tuple(runtimes)


def _validate_supported_package(package: ExpertPackage) -> None:
    """Reject unsupported strategies before loading any package tensor."""

    manifest = package.manifest
    unsupported: list[str] = []
    if manifest.expert_parameterization != "full_mlp":
        unsupported.append(
            f"expert_parameterization={manifest.expert_parameterization!r}"
        )
    if manifest.router_architecture != "independent_affine_heads":
        unsupported.append(
            f"router_architecture={manifest.router_architecture!r}"
        )
    if "anchored_top1" not in manifest.routing_policies_supported:
        unsupported.append("routing_policies_supported lacks 'anchored_top1'")
    if unsupported:
        raise PackageMaterializationError(
            f"Package {package.identity.version_key!r} uses unsupported runtime "
            "strategies: "
            + ", ".join(unsupported)
        )


def _validate_identity_feature_specification(specification: Any) -> int:
    """Validate the one frozen feature transformation supported by PKG-007."""

    if not isinstance(specification, dict) and not hasattr(
        specification, "items"
    ):
        raise PackageMaterializationError(
            "router_feature_specification must be a mapping."
        )
    required = {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "version": 1,
    }
    mismatches = {
        key: (specification.get(key), expected)
        for key, expected in required.items()
        if specification.get(key) != expected
    }
    output_size = specification.get("output_size")
    if mismatches or (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size <= 0
    ):
        raise PackageMaterializationError(
            "Unsupported router feature specification: "
            f"mismatches={mismatches}, output_size={output_size!r}."
        )
    return output_size


def _load_package_router_heads(
    package: ExpertPackage,
) -> AppendOnlyRouterHeads:
    """Load and merge all declared affine router files in manifest order."""

    merged: AppendOnlyRouterHeads | None = None
    for relative_path in package.manifest.router_files:
        if not relative_path.endswith(".safetensors"):
            raise PackageMaterializationError(
                "Anchored top-1 materialization requires safetensors router "
                f"payloads, received {relative_path!r}."
            )
        source = package.package_directory.joinpath(
            *PurePosixPath(relative_path).parts
        )
        try:
            loaded = AppendOnlyRouterHeads.from_safetensors(
                source,
                trainable=False,
            )
        except RouterSerializationError as error:
            raise PackageMaterializationError(
                f"Cannot load router payload {relative_path!r}: {error}"
            ) from error
        if merged is None:
            merged = AppendOnlyRouterHeads(loaded.feature_size)
        if loaded.feature_size != merged.feature_size:
            raise PackageMaterializationError(
                "Router payloads within one package disagree on feature size."
            )
        try:
            for head in loaded.heads:
                merged.append(head)
        except ValueError as error:
            raise PackageMaterializationError(
                f"Router payload ownership is ambiguous: {error}"
            ) from error

    if merged is None:
        raise PackageMaterializationError(
            "Anchored top-1 package has no router safetensors payload."
        )
    return merged


def _module_dtype_and_device(
    module: nn.Module,
    *,
    layer_index: int,
) -> tuple[Any, Any]:
    """Return one uniform floating dtype/device contract for an anchor MLP."""

    tensors: list[Tensor] = [
        tensor
        for tensor in (*module.parameters(), *module.buffers())
        if tensor.is_floating_point()
    ]
    if not tensors:
        raise PackageMaterializationError(
            f"Anchor MLP at layer {layer_index} has no floating tensors."
        )
    dtypes = {tensor.dtype for tensor in tensors}
    devices = {tensor.device for tensor in tensors}
    if len(dtypes) != 1 or len(devices) != 1:
        raise PackageMaterializationError(
            f"Anchor MLP at layer {layer_index} has mixed dtype or device state."
        )
    return tensors[0].dtype, tensors[0].device
