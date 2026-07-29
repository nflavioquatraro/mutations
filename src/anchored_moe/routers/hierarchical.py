"""Frozen two-stage package and package-local router composition.

RTR-005 composes existing RTR-003 logical score banks without introducing a
new fitting algorithm or on-disk format. One logical bank scores packages and
one logical bank per package scores only that package's local experts. The
wrapper copies every source bank so later caller-side mutation cannot change a
deployed hierarchy.

The hierarchy deliberately does not select routes. POL-004 first chooses a
package, then passes those package indices back to
:meth:`HierarchicalRouterScoreBank.score_selected_locals`. Only token rows
assigned to a package reach its local score bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from anchored_moe.routers.logical import LogicalAppendOnlyRouter


class HierarchicalRouterError(ValueError):
    """Raised when a hierarchical router topology or score request is invalid."""


@dataclass(frozen=True)
class HierarchicalLocalScores:
    """Local scores for the flattened token rows assigned to one package.

    Attributes:
        package_id:
            Stable package identifier selected by the package-level router.
        flat_token_indices:
            Strictly increasing indices into the flattened token-leading
            dimensions of the original feature tensor, with shape ``(N,)``.
        expert_scores:
            Finite local scores with shape ``(N, K_package)``.
        expert_ids:
            Stable local expert order matching the final score dimension.
    """

    package_id: str
    flat_token_indices: Tensor
    expert_scores: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identifiers, flattened ownership, and score alignment."""

        package_id = _validate_identifier(self.package_id, field_name="package_id")
        expert_ids = _normalize_identifiers(
            self.expert_ids,
            field_name="expert_ids",
        )
        if len(expert_ids) < 2:
            raise HierarchicalRouterError(
                "A hierarchical local score group requires at least two experts."
            )
        if not isinstance(self.flat_token_indices, Tensor):
            raise TypeError("flat_token_indices must be a torch.Tensor.")
        if (
            self.flat_token_indices.dtype != torch.long
            or self.flat_token_indices.ndim != 1
        ):
            raise HierarchicalRouterError(
                "flat_token_indices must be a one-dimensional torch.long tensor."
            )
        if not isinstance(self.expert_scores, Tensor):
            raise TypeError("expert_scores must be a torch.Tensor.")
        if (
            not self.expert_scores.is_floating_point()
            or self.expert_scores.ndim != 2
        ):
            raise HierarchicalRouterError(
                "expert_scores must be a two-dimensional floating-point tensor."
            )
        if self.expert_scores.shape != (
            self.flat_token_indices.numel(),
            len(expert_ids),
        ):
            raise HierarchicalRouterError(
                "expert_scores shape must match flattened token and expert counts."
            )
        if self.flat_token_indices.device != self.expert_scores.device:
            raise HierarchicalRouterError(
                "flat_token_indices and expert_scores must share one device."
            )
        if not torch.isfinite(self.expert_scores).all().item():
            raise HierarchicalRouterError(
                "expert_scores must contain only finite values."
            )
        if self.flat_token_indices.numel():
            if int(self.flat_token_indices[0].item()) < 0:
                raise HierarchicalRouterError(
                    "flat_token_indices must be non-negative."
                )
            if self.flat_token_indices.numel() > 1 and not torch.all(
                self.flat_token_indices[1:] > self.flat_token_indices[:-1]
            ).item():
                raise HierarchicalRouterError(
                    "flat_token_indices must be strictly increasing."
                )
        object.__setattr__(self, "package_id", package_id)
        object.__setattr__(self, "expert_ids", expert_ids)


class HierarchicalRouterScoreBank(nn.Module):
    """Compose one package router and deterministic package-local routers.

    Args:
        package_router:
            Frozen RTR-003 logical bank whose layer-local identifiers are
            interpreted as package IDs.
        local_routers:
            Exact mapping from every package ID to its frozen RTR-003 local
            expert bank.

    Version 1 intentionally requires a rectangular hierarchy:

    - every package is present at every package-router layer in one order;
    - every local router covers exactly those layers;
    - each package exposes the same ordered set of at least two experts at
      every layer;
    - expert IDs are globally unique across packages.

    All source routers are converted to independent heads and compiled again.
    The deployed hierarchy therefore owns buffer-only copies and does not
    retain source tensor storage.

    This class is framework infrastructure, not a paper-derived algorithm.
    It implements the two-stage routing boundary from the architecture
    blueprint without changing RTR-003 fitting or serialization.
    """

    architecture = "hierarchical_package_local"

    def __init__(
        self,
        package_router: LogicalAppendOnlyRouter,
        local_routers: Mapping[str, LogicalAppendOnlyRouter],
    ) -> None:
        super().__init__()
        if not isinstance(package_router, LogicalAppendOnlyRouter):
            raise TypeError("package_router must be a LogicalAppendOnlyRouter.")
        if not isinstance(local_routers, Mapping):
            raise TypeError("local_routers must be a mapping.")
        if not package_router.head_keys:
            raise HierarchicalRouterError("package_router must not be empty.")

        source_package_ids, layer_indices = _package_topology(package_router)
        normalized_sources = _normalize_local_router_mapping(local_routers)
        if set(normalized_sources) != set(source_package_ids):
            raise HierarchicalRouterError(
                "local_routers keys must exactly match package-router IDs: "
                f"expected={list(source_package_ids)!r}, "
                f"actual={list(normalized_sources)!r}."
            )

        source_state = _first_buffer(package_router)
        local_expert_ids: dict[str, tuple[str, ...]] = {}
        seen_experts: dict[str, str] = {}
        for package_id in source_package_ids:
            local_router = normalized_sources[package_id]
            if local_router.feature_size != package_router.feature_size:
                raise HierarchicalRouterError(
                    f"Local router {package_id!r} has feature size "
                    f"{local_router.feature_size}, expected "
                    f"{package_router.feature_size}."
                )
            local_state = _first_buffer(local_router)
            if local_state.dtype != source_state.dtype:
                raise HierarchicalRouterError(
                    "Every hierarchical router must share one dtype."
                )
            if local_state.device != source_state.device:
                raise HierarchicalRouterError(
                    "Every hierarchical router must share one device."
                )
            expert_ids = _local_topology(
                local_router,
                package_id=package_id,
                layer_indices=layer_indices,
            )
            for expert_id in expert_ids:
                incumbent = seen_experts.get(expert_id)
                if incumbent is not None:
                    raise HierarchicalRouterError(
                        f"Expert ID {expert_id!r} is shared by packages "
                        f"{incumbent!r} and {package_id!r}."
                    )
                seen_experts[expert_id] = package_id
            local_expert_ids[package_id] = expert_ids

        self.feature_size = package_router.feature_size
        self._package_ids = source_package_ids
        self._layer_indices = layer_indices
        self._local_expert_ids = local_expert_ids
        self.package_router = _copy_logical_router(package_router)
        self.local_routers = nn.ModuleList(
            _copy_logical_router(normalized_sources[package_id])
            for package_id in self._package_ids
        )

    @property
    def package_ids(self) -> tuple[str, ...]:
        """Return the immutable package order shared by every routed layer."""

        return self._package_ids

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return the exact decoder layers covered by the hierarchy."""

        return self._layer_indices

    def package_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return package IDs for a covered layer or reject the layer."""

        self._validate_layer(layer_index)
        return self._package_ids

    def expert_ids_for_package(
        self,
        package_id: str,
        *,
        layer_index: int,
    ) -> tuple[str, ...]:
        """Return one package's stable local expert order."""

        normalized_package = _validate_identifier(
            package_id,
            field_name="package_id",
        )
        self._validate_layer(layer_index)
        try:
            return self._local_expert_ids[normalized_package]
        except KeyError as error:
            raise KeyError(f"Unknown package_id: {normalized_package!r}.") from error

    def global_expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return local experts flattened in package order for dispatch."""

        self._validate_layer(layer_index)
        return tuple(
            expert_id
            for package_id in self._package_ids
            for expert_id in self._local_expert_ids[package_id]
        )

    def expert_package_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return the package owner aligned with flattened global expert IDs."""

        self._validate_layer(layer_index)
        return tuple(
            package_id
            for package_id in self._package_ids
            for _ in self._local_expert_ids[package_id]
        )

    def score_packages(self, features: Tensor, *, layer_index: int) -> Tensor:
        """Return package scores without evaluating any local router."""

        self._validate_layer(layer_index)
        return self.package_router(features, layer_index=layer_index)

    def score_selected_locals(
        self,
        features: Tensor,
        *,
        package_indices: Tensor,
        package_ids: Sequence[str],
        layer_index: int,
    ) -> tuple[HierarchicalLocalScores, ...]:
        """Score only token rows assigned to each selected package.

        ``package_indices`` uses candidate zero for the anchor and candidate
        ``i + 1`` for ``package_ids[i]``. Its shape must equal
        ``features.shape[:-1]``. Anchor rows are omitted from all local banks.
        Returned groups follow package order and contain strictly increasing
        flattened token indices, allowing POL-004 to reconstruct the original
        token layout without ambiguity.
        """

        self._validate_layer(layer_index)
        _validate_features(features, feature_size=self.feature_size)
        normalized_ids = _normalize_identifiers(
            package_ids,
            field_name="package_ids",
        )
        if normalized_ids != self._package_ids:
            raise HierarchicalRouterError(
                "package_ids must match the hierarchy's exact package order."
            )
        if not isinstance(package_indices, Tensor):
            raise TypeError("package_indices must be a torch.Tensor.")
        if package_indices.dtype != torch.long:
            raise TypeError("package_indices must use torch.long.")
        if package_indices.shape != features.shape[:-1]:
            raise HierarchicalRouterError(
                "package_indices shape must match feature token-leading shape."
            )
        if package_indices.device != features.device:
            raise HierarchicalRouterError(
                "package_indices and features must share one device."
            )
        if package_indices.numel():
            minimum = int(package_indices.min().item())
            maximum = int(package_indices.max().item())
            if minimum < 0 or maximum > len(self._package_ids):
                raise HierarchicalRouterError(
                    "package_indices contains an unknown package candidate."
                )

        flat_features = features.reshape(-1, self.feature_size)
        flat_package_indices = package_indices.reshape(-1)
        groups: list[HierarchicalLocalScores] = []
        for package_position, package_id in enumerate(self._package_ids, start=1):
            flat_token_indices = torch.nonzero(
                flat_package_indices == package_position,
                as_tuple=False,
            ).squeeze(-1)
            if not flat_token_indices.numel():
                continue
            selected_features = torch.index_select(
                flat_features,
                dim=0,
                index=flat_token_indices,
            )
            local_router = self.local_routers[package_position - 1]
            expert_scores = local_router(
                selected_features,
                layer_index=layer_index,
            )
            groups.append(
                HierarchicalLocalScores(
                    package_id=package_id,
                    flat_token_indices=flat_token_indices,
                    expert_scores=expert_scores,
                    expert_ids=self._local_expert_ids[package_id],
                )
            )
        return tuple(groups)

    def extra_repr(self) -> str:
        """Describe the frozen rectangular hierarchy."""

        return (
            f"feature_size={self.feature_size}, "
            f"package_count={len(self._package_ids)}, "
            f"layer_count={len(self._layer_indices)}"
        )

    def _validate_layer(self, layer_index: int) -> None:
        """Require a non-boolean integer covered by the hierarchy."""

        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer.")
        if layer_index not in self._layer_indices:
            raise LookupError(
                f"Layer {layer_index} is not covered by the hierarchical router."
            )


def _copy_logical_router(
    router: LogicalAppendOnlyRouter,
) -> LogicalAppendOnlyRouter:
    """Return a buffer-only storage-independent logical-router copy."""

    return LogicalAppendOnlyRouter.from_independent_heads(
        router.to_independent_heads(trainable=False)
    ).eval()


def _package_topology(
    router: LogicalAppendOnlyRouter,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return and validate rectangular package IDs and layer coverage."""

    layer_indices = tuple(sorted({layer_index for layer_index, _ in router.head_keys}))
    first_ids = router.expert_ids_for_layer(layer_indices[0])
    if not first_ids:
        raise HierarchicalRouterError("package_router layers must not be empty.")
    for layer_index in layer_indices[1:]:
        layer_ids = router.expert_ids_for_layer(layer_index)
        if layer_ids != first_ids:
            raise HierarchicalRouterError(
                "package_router must expose one package order at every layer."
            )
    return first_ids, layer_indices


def _local_topology(
    router: LogicalAppendOnlyRouter,
    *,
    package_id: str,
    layer_indices: tuple[int, ...],
) -> tuple[str, ...]:
    """Validate one rectangular package-local expert topology."""

    local_layers = tuple(sorted({layer_index for layer_index, _ in router.head_keys}))
    if local_layers != layer_indices:
        raise HierarchicalRouterError(
            f"Local router {package_id!r} must cover exactly layers "
            f"{list(layer_indices)!r}."
        )
    first_ids = router.expert_ids_for_layer(layer_indices[0])
    if len(first_ids) < 2:
        raise HierarchicalRouterError(
            f"Local router {package_id!r} requires at least two experts."
        )
    for layer_index in layer_indices[1:]:
        layer_ids = router.expert_ids_for_layer(layer_index)
        if layer_ids != first_ids:
            raise HierarchicalRouterError(
                f"Local router {package_id!r} must expose one expert order "
                "at every layer."
            )
    return first_ids


def _normalize_local_router_mapping(
    routers: Mapping[str, LogicalAppendOnlyRouter],
) -> dict[str, LogicalAppendOnlyRouter]:
    """Validate package keys and concrete RTR-003 local banks."""

    normalized: dict[str, LogicalAppendOnlyRouter] = {}
    for package_id, router in routers.items():
        normalized_id = _validate_identifier(package_id, field_name="package_id")
        if normalized_id in normalized:
            raise HierarchicalRouterError(
                f"Duplicate package_id after normalization: {normalized_id!r}."
            )
        if not isinstance(router, LogicalAppendOnlyRouter):
            raise TypeError(
                f"Local router {normalized_id!r} must be a "
                "LogicalAppendOnlyRouter."
            )
        if not router.head_keys:
            raise HierarchicalRouterError(
                f"Local router {normalized_id!r} must not be empty."
            )
        normalized[normalized_id] = router
    return normalized


def _first_buffer(router: LogicalAppendOnlyRouter) -> Tensor:
    """Return one logical-router buffer or reject an empty bank."""

    state = next(router.buffers(), None)
    if state is None:
        raise HierarchicalRouterError("Logical router must contain buffer state.")
    return state


def _validate_features(features: Tensor, *, feature_size: int) -> None:
    """Validate the common hierarchical feature contract."""

    if not isinstance(features, Tensor) or not features.is_floating_point():
        raise TypeError("features must be a floating-point torch.Tensor.")
    if features.ndim < 1 or features.shape[-1] != feature_size:
        raise HierarchicalRouterError(
            "features final dimension must match hierarchical feature_size."
        )
    if not torch.isfinite(features).all().item():
        raise HierarchicalRouterError("features must contain only finite values.")


def _normalize_identifiers(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Return unique stable identifiers without accepting a bare string."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(
        _validate_identifier(value, field_name=field_name) for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise HierarchicalRouterError(f"{field_name} must not contain duplicates.")
    return normalized


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate one exact package or expert identifier."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise HierarchicalRouterError(
            f"{field_name} values must be non-empty strings without "
            "surrounding whitespace."
        )
    return value
