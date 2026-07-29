"""Deterministic package top-1 followed by package-local top-2 selection.

POL-004 separates the fixed-zero anchor/package decision from package-local
expert ranking. Once a package wins, its two highest local experts are
selected unconditionally: local scores need only be comparable inside that
package and are never compared with the global anchor score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from anchored_moe.policies.selection import (
    AnchoredTop1Selection,
    RouteDecision,
    Top2RouteDecision,
)
from anchored_moe.routers.hierarchical import HierarchicalLocalScores


class HierarchicalSelectionError(ValueError):
    """Raised when staged package and local route metadata is inconsistent."""


@dataclass(frozen=True)
class HierarchicalRouteDecision:
    """Auditable package routes plus flattened dispatch-compatible expert pairs.

    Attributes:
        package_routes:
            Anchored top-1 package decision. Candidate zero is the anchor.
        expert_routes:
            Flattened top-2 decision in package-major expert order. Anchor
            package rows contain ``[0, 0]``; selected package rows contain two
            distinct local experts from that package.
        expert_package_ids:
            Package owner aligned with ``expert_routes.expert_ids``.
    """

    package_routes: RouteDecision
    expert_routes: Top2RouteDecision
    expert_package_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate stage alignment and prohibit every cross-package pair."""

        if not isinstance(self.package_routes, RouteDecision):
            raise TypeError("package_routes must be a RouteDecision.")
        if not isinstance(self.expert_routes, Top2RouteDecision):
            raise TypeError("expert_routes must be a Top2RouteDecision.")
        owners = _normalize_owner_ids(
            self.expert_package_ids,
            field_name="expert_package_ids",
        )
        if len(owners) != len(self.expert_routes.expert_ids):
            raise HierarchicalSelectionError(
                "expert_package_ids must align with expert_routes.expert_ids."
            )
        if not set(owners) <= set(self.package_routes.expert_ids):
            raise HierarchicalSelectionError(
                "Every expert owner must be a declared package candidate."
            )
        if self.expert_routes.selected_indices.shape[:-1] != (
            self.package_routes.selected_indices.shape
        ):
            raise HierarchicalSelectionError(
                "Package and expert routes must share one token-leading shape."
            )
        if not (
            self.package_routes.selected_indices.device
            == self.expert_routes.selected_indices.device
        ):
            raise HierarchicalSelectionError(
                "Package and expert routes must share one device."
            )
        if self.package_routes.selected_scores.dtype != (
            self.expert_routes.selected_scores.dtype
        ):
            raise HierarchicalSelectionError(
                "Package and expert route scores must share one dtype."
            )
        if not torch.equal(
            self.package_routes.anchor_mask,
            self.expert_routes.anchor_mask,
        ):
            raise HierarchicalSelectionError(
                "Package-anchor and expert-anchor masks must be identical."
            )

        package_positions = {
            package_id: position
            for position, package_id in enumerate(
                self.package_routes.expert_ids,
                start=1,
            )
        }
        owner_candidates = torch.tensor(
            [0, *(package_positions[owner] for owner in owners)],
            dtype=torch.long,
            device=self.expert_routes.selected_indices.device,
        )
        selected_owner_positions = owner_candidates[
            self.expert_routes.selected_indices
        ]
        expected_owner_positions = (
            self.package_routes.selected_indices.unsqueeze(-1)
        )
        active_rows = ~self.package_routes.anchor_mask
        if torch.any(
            self.expert_routes.selected_indices[..., 0][active_rows]
            == self.expert_routes.selected_indices[..., 1][active_rows]
        ).item():
            raise HierarchicalSelectionError(
                "Selected local expert pairs must contain distinct experts."
            )
        if torch.any(
            selected_owner_positions[active_rows]
            != expected_owner_positions.expand_as(
                selected_owner_positions
            )[active_rows]
        ).item():
            raise HierarchicalSelectionError(
                "Every selected expert pair must stay inside its selected package."
            )
        object.__setattr__(self, "expert_package_ids", owners)

    @property
    def anchor_mask(self) -> Tensor:
        """Return tokens retained on the immutable anchor."""

        return self.package_routes.anchor_mask

    @property
    def expert_pair_mask(self) -> Tensor:
        """Return tokens assigned to one package-local expert pair."""

        return self.expert_routes.expert_pair_mask


class HierarchicalPackageTop1LocalTop2Selection:
    """Select anchor/package first and exactly two experts inside each winner.

    The staged API preserves the router/policy boundary:

    1. :meth:`select_packages` consumes package scores only.
    2. RTR-005 scores local banks for the selected token rows.
    3. :meth:`select_locals` ranks those rows and reconstructs global routes.

    Package selection is fixed-zero anchored top-1, including anchor-first and
    package-order tie-breaking. Local selection is a stable specialist-only
    descending rank. It does not apply another anchor veto or require positive
    local scores because local calibration is intentionally package-relative.

    This policy is framework infrastructure, not a paper-derived algorithm.
    It implements the hierarchical strategy defined by the architecture
    blueprint and reuses POL-001's package gate plus POL-002's pair-shaped
    route metadata.
    """

    policy_name = "hierarchical_package_top1_local_top2"

    def __init__(self) -> None:
        """Create the stateless policy with an anchored package selector."""

        self._package_selection = AnchoredTop1Selection()

    def select_packages(
        self,
        package_scores: Tensor,
        *,
        package_ids: Sequence[str],
    ) -> RouteDecision:
        """Return the deterministic fixed-zero package decision."""

        return self._package_selection(
            package_scores,
            expert_ids=package_ids,
        )

    def select_locals(
        self,
        package_routes: RouteDecision,
        local_scores: Sequence[HierarchicalLocalScores],
        *,
        expert_ids_by_package: Mapping[str, Sequence[str]],
    ) -> HierarchicalRouteDecision:
        """Rank selected local experts and reconstruct global pair metadata.

        Args:
            package_routes:
                Result previously returned by :meth:`select_packages`.
            local_scores:
                RTR-005 groups for exactly the active packages. Each group's
                flattened indices must exactly equal the rows assigned to that
                package.
            expert_ids_by_package:
                Complete topology mapping in ``package_routes.expert_ids``
                order, including packages with no selected rows.
        """

        if not isinstance(package_routes, RouteDecision):
            raise TypeError("package_routes must be a RouteDecision.")
        if isinstance(local_scores, (str, bytes)) or not isinstance(
            local_scores,
            Sequence,
        ):
            raise TypeError(
                "local_scores must be a sequence of HierarchicalLocalScores."
            )
        topology = _normalize_topology(
            expert_ids_by_package,
            package_ids=package_routes.expert_ids,
        )
        groups = _normalize_groups(
            local_scores,
            package_routes=package_routes,
            topology=topology,
        )

        global_expert_ids = tuple(
            expert_id
            for package_id in package_routes.expert_ids
            for expert_id in topology[package_id]
        )
        expert_package_ids = tuple(
            package_id
            for package_id in package_routes.expert_ids
            for _ in topology[package_id]
        )
        global_offset: dict[str, int] = {}
        offset = 0
        for package_id in package_routes.expert_ids:
            global_offset[package_id] = offset
            offset += len(topology[package_id])

        pair_shape = package_routes.selected_indices.shape + (2,)
        selected_indices = torch.zeros(
            pair_shape,
            dtype=torch.long,
            device=package_routes.selected_indices.device,
        )
        selected_scores = torch.zeros(
            pair_shape,
            dtype=package_routes.selected_scores.dtype,
            device=package_routes.selected_scores.device,
        )
        flat_indices = selected_indices.reshape(-1, 2)
        flat_scores = selected_scores.reshape(-1, 2)
        for group in groups:
            # Stable sorting preserves local expert order for exact score ties.
            ranked_local = torch.argsort(
                group.expert_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[:, :2]
            ranked_scores = torch.gather(
                group.expert_scores,
                dim=-1,
                index=ranked_local,
            )
            candidate_offset = global_offset[group.package_id] + 1
            flat_indices[group.flat_token_indices] = (
                ranked_local + candidate_offset
            )
            flat_scores[group.flat_token_indices] = ranked_scores

        expert_routes = Top2RouteDecision(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            expert_ids=global_expert_ids,
        )
        return HierarchicalRouteDecision(
            package_routes=package_routes,
            expert_routes=expert_routes,
            expert_package_ids=expert_package_ids,
        )


def _normalize_topology(
    topology: Mapping[str, Sequence[str]],
    *,
    package_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Validate the complete package-major expert identity topology."""

    if not isinstance(topology, Mapping):
        raise TypeError("expert_ids_by_package must be a mapping.")
    if set(topology) != set(package_ids):
        raise HierarchicalSelectionError(
            "expert_ids_by_package keys must exactly match package route IDs."
        )
    normalized: dict[str, tuple[str, ...]] = {}
    seen_experts: dict[str, str] = {}
    for package_id in package_ids:
        expert_ids = _normalize_identifiers(
            topology[package_id],
            field_name=f"expert_ids_by_package[{package_id!r}]",
        )
        if len(expert_ids) < 2:
            raise HierarchicalSelectionError(
                f"Package {package_id!r} requires at least two local experts."
            )
        for expert_id in expert_ids:
            incumbent = seen_experts.get(expert_id)
            if incumbent is not None:
                raise HierarchicalSelectionError(
                    f"Expert ID {expert_id!r} is shared by packages "
                    f"{incumbent!r} and {package_id!r}."
                )
            seen_experts[expert_id] = package_id
        normalized[package_id] = expert_ids
    return normalized


def _normalize_groups(
    groups: Sequence[HierarchicalLocalScores],
    *,
    package_routes: RouteDecision,
    topology: Mapping[str, tuple[str, ...]],
) -> tuple[HierarchicalLocalScores, ...]:
    """Validate exact selected-row coverage in stable package order."""

    by_package: dict[str, HierarchicalLocalScores] = {}
    for group in groups:
        if not isinstance(group, HierarchicalLocalScores):
            raise TypeError(
                "local_scores entries must be HierarchicalLocalScores."
            )
        if group.package_id in by_package:
            raise HierarchicalSelectionError(
                f"Duplicate local score group for {group.package_id!r}."
            )
        if group.package_id not in topology:
            raise HierarchicalSelectionError(
                f"Unknown local score package: {group.package_id!r}."
            )
        if group.expert_ids != topology[group.package_id]:
            raise HierarchicalSelectionError(
                f"Local expert order for {group.package_id!r} does not match "
                "the declared topology."
            )
        if group.expert_scores.dtype != package_routes.selected_scores.dtype:
            raise HierarchicalSelectionError(
                "Package and local scores must share one dtype."
            )
        if group.expert_scores.device != package_routes.selected_scores.device:
            raise HierarchicalSelectionError(
                "Package and local scores must share one device."
            )
        by_package[group.package_id] = group

    flat_package_indices = package_routes.selected_indices.reshape(-1)
    ordered: list[HierarchicalLocalScores] = []
    for package_position, package_id in enumerate(
        package_routes.expert_ids,
        start=1,
    ):
        expected = torch.nonzero(
            flat_package_indices == package_position,
            as_tuple=False,
        ).squeeze(-1)
        group = by_package.get(package_id)
        if not expected.numel():
            if group is not None:
                raise HierarchicalSelectionError(
                    f"Inactive package {package_id!r} must not have local scores."
                )
            continue
        if group is None:
            raise HierarchicalSelectionError(
                f"Active package {package_id!r} is missing local scores."
            )
        if not torch.equal(group.flat_token_indices, expected):
            raise HierarchicalSelectionError(
                f"Local token indices for {package_id!r} do not exactly match "
                "the package decision."
            )
        ordered.append(group)
    if set(by_package) != {group.package_id for group in ordered}:
        raise HierarchicalSelectionError(
            "Local score groups contain packages outside the active routes."
        )
    return tuple(ordered)


def _normalize_identifiers(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Return stable unique non-empty identifiers."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise HierarchicalSelectionError(
                f"{field_name} values must be non-empty strings without "
                "surrounding whitespace."
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise HierarchicalSelectionError(
            f"{field_name} must not contain duplicates."
        )
    return tuple(normalized)


def _normalize_owner_ids(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate aligned owner IDs while allowing one package per many experts."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise HierarchicalSelectionError(
                f"{field_name} values must be non-empty strings without "
                "surrounding whitespace."
            )
        normalized.append(value)
    return tuple(normalized)
