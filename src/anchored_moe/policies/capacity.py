"""Capacity policies for deterministic anchored fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anchored_moe.policies.selection import (
    AdaptiveRouteDecision,
    RouteDecision,
    Top2RouteDecision,
)


@dataclass(frozen=True)
class CapacityDecision:
    """Effective routes after applying an expert-capacity policy.

    Attributes:
        routes:
            Route decision after unavailable specialist selections have been
            rewritten. Candidate zero remains the immutable anchor.
        overflow_mask:
            Boolean tensor with the same token shape as ``routes``. ``True``
            marks a token whose originally selected expert was unavailable.

    Capacity decisions contain route metadata only. They neither execute an
    expert nor calculate an expert's serving capacity.
    """

    routes: RouteDecision
    overflow_mask: Tensor

    def __post_init__(self) -> None:
        """Validate the public tensor shape, dtype, and device contract."""

        if not isinstance(self.routes, RouteDecision):
            raise TypeError("routes must be a RouteDecision.")
        if not isinstance(self.overflow_mask, Tensor):
            raise TypeError("overflow_mask must be a torch.Tensor.")
        if self.overflow_mask.dtype != torch.bool:
            raise TypeError("overflow_mask must use torch.bool.")
        if self.overflow_mask.shape != self.routes.selected_indices.shape:
            raise ValueError(
                "overflow_mask must have the same token shape as routes."
            )
        if self.overflow_mask.device != self.routes.selected_indices.device:
            raise ValueError("overflow_mask and routes must share one device.")


@dataclass(frozen=True)
class Top2CapacityDecision:
    """Effective anchor/top-1/top-2 routes after pair capacity handling.

    ``effective_indices`` and ``effective_scores`` have shape
    ``token_shape + (2,)``. Active specialist slots are packed from the left;
    candidate zero and score zero pad inactive slots. ``route_arity`` has
    ``token_shape`` and records whether the anchor, one specialist, or two
    specialists execute.

    ``overflow_mask`` retains the original selected-pair shape. A true entry
    means that an approved specialist occurrence was unavailable. Anchor-vetoed
    pairs never overflow because they did not request specialist execution.

    This dedicated result does not reuse ``AdaptiveRouteDecision``: POL-003
    deliberately retains a complete specialist ranking even for anchor routes,
    while BAL-002 exposes only the capacity-approved execution slots.
    """

    effective_indices: Tensor
    effective_scores: Tensor
    route_arity: Tensor
    expert_ids: tuple[str, ...]
    overflow_mask: Tensor

    def __post_init__(self) -> None:
        """Validate shapes, padding, identifiers, and execution arity."""

        if not isinstance(self.effective_indices, Tensor):
            raise TypeError("effective_indices must be a torch.Tensor.")
        if self.effective_indices.dtype != torch.long:
            raise TypeError("effective_indices must use torch.long.")
        if not isinstance(self.effective_scores, Tensor):
            raise TypeError("effective_scores must be a torch.Tensor.")
        if not self.effective_scores.is_floating_point():
            raise TypeError("effective_scores must use a floating-point dtype.")
        if self.effective_indices.shape != self.effective_scores.shape:
            raise ValueError(
                "effective_indices and effective_scores must share one shape."
            )
        if (
            self.effective_indices.ndim < 1
            or self.effective_indices.shape[-1] != 2
        ):
            raise ValueError(
                "effective_indices and effective_scores must have a final "
                "dimension of size two."
            )
        if not isinstance(self.route_arity, Tensor):
            raise TypeError("route_arity must be a torch.Tensor.")
        if self.route_arity.dtype != torch.long:
            raise TypeError("route_arity must use torch.long.")
        if self.route_arity.shape != self.effective_indices.shape[:-1]:
            raise ValueError(
                "route_arity must have the effective-route token shape."
            )
        if not isinstance(self.overflow_mask, Tensor):
            raise TypeError("overflow_mask must be a torch.Tensor.")
        if self.overflow_mask.dtype != torch.bool:
            raise TypeError("overflow_mask must use torch.bool.")
        if self.overflow_mask.shape != self.effective_indices.shape:
            raise ValueError(
                "overflow_mask must have the effective pair shape."
            )
        if not (
            self.effective_indices.device
            == self.effective_scores.device
            == self.route_arity.device
            == self.overflow_mask.device
        ):
            raise ValueError("All top-2 capacity tensors must share one device.")

        normalized_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", normalized_ids)
        if self.effective_indices.numel():
            minimum = int(self.effective_indices.min().item())
            maximum = int(self.effective_indices.max().item())
            if minimum < 0 or maximum > len(normalized_ids):
                raise ValueError(
                    "effective_indices contains a candidate outside the anchor "
                    f"and {len(normalized_ids)} declared experts."
                )
        if not torch.isfinite(self.effective_scores).all().item():
            raise ValueError("effective_scores must contain only finite values.")
        if self.route_arity.numel():
            minimum_arity = int(self.route_arity.min().item())
            maximum_arity = int(self.route_arity.max().item())
            if minimum_arity < 0 or maximum_arity > 2:
                raise ValueError("route_arity values must be zero, one, or two.")

        slot_indices = torch.arange(
            2,
            dtype=torch.long,
            device=self.route_arity.device,
        )
        active_slots = slot_indices < self.route_arity.unsqueeze(-1)
        if torch.any(self.effective_indices[active_slots] == 0).item():
            raise ValueError("Active capacity routes cannot use the anchor index.")
        if torch.any(self.effective_indices[~active_slots] != 0).item():
            raise ValueError("Inactive capacity slots must use anchor padding.")
        if torch.any(self.effective_scores[~active_slots] != 0).item():
            raise ValueError("Inactive capacity slots must use zero scores.")
        top2_mask = self.route_arity == 2
        if top2_mask.any().item() and torch.any(
            self.effective_indices[..., 0][top2_mask]
            == self.effective_indices[..., 1][top2_mask]
        ).item():
            raise ValueError("Top-2 capacity routes require distinct specialists.")

    @property
    def anchor_mask(self) -> Tensor:
        """Return tokens whose effective route is the immutable anchor."""

        return self.route_arity == 0

    @property
    def top1_mask(self) -> Tensor:
        """Return tokens degraded to one available specialist."""

        return self.route_arity == 1

    @property
    def top2_mask(self) -> Tensor:
        """Return tokens retaining both selected specialists."""

        return self.route_arity == 2

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return tokens where ``expert_id`` is capacity-approved."""

        try:
            candidate_index = self.expert_ids.index(expert_id) + 1
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        return torch.any(self.effective_indices == candidate_index, dim=-1)


class OverflowToAnchorCapacity:
    """Rewrite unavailable top-1 expert routes to the anchor.

    For selected candidate ``r_t`` and expert availability
    ``C[t, k] in {False, True}``, the effective route is

    ``r'_t = r_t`` when ``r_t = 0`` or ``C[t, r_t - 1]`` is true, and
    ``r'_t = 0`` otherwise.

    ``expert_availability`` has shape ``route_shape + (K,)`` in the same stable
    expert order as :class:`~anchored_moe.policies.RouteDecision.expert_ids`.
    Omitting it declares every expert available. The policy never changes an
    anchor route and never drops a token.

    This is infrastructure rather than a paper-derived balancing algorithm.
    It does not estimate capacity, reorder candidates, or fall through to a
    second specialist.
    """

    policy_name = "overflow_to_anchor"

    def __call__(
        self,
        routes: RouteDecision,
        *,
        expert_availability: Tensor | None = None,
    ) -> CapacityDecision:
        """Apply the policy and return effective routes plus overflow metadata.

        Args:
            routes:
                Deterministic top-1 selection to validate against capacity.
            expert_availability:
                Optional boolean tensor with shape
                ``routes.selected_indices.shape + (K,)``.

        Returns:
            A :class:`CapacityDecision`. When no selected expert overflows, the
            original ``RouteDecision`` object is retained.

        Raises:
            TypeError:
                If routes or availability use incompatible public types.
            ValueError:
                If availability has the wrong shape or device.
        """

        if not isinstance(routes, RouteDecision):
            raise TypeError("routes must be a RouteDecision.")
        expert_count = len(routes.expert_ids)
        expected_shape = routes.selected_indices.shape + (expert_count,)

        if expert_availability is None:
            overflow_mask = torch.zeros_like(
                routes.selected_indices,
                dtype=torch.bool,
            )
            return CapacityDecision(routes=routes, overflow_mask=overflow_mask)

        if not isinstance(expert_availability, Tensor):
            raise TypeError("expert_availability must be a torch.Tensor.")
        if expert_availability.dtype != torch.bool:
            raise TypeError("expert_availability must use torch.bool.")
        if expert_availability.shape != expected_shape:
            raise ValueError(
                "expert_availability must have route shape plus one expert "
                f"dimension: {tuple(expert_availability.shape)} != "
                f"{tuple(expected_shape)}."
            )
        if expert_availability.device != routes.selected_indices.device:
            raise ValueError(
                "expert_availability and routes must share one device."
            )

        if expert_count == 0:
            overflow_mask = torch.zeros_like(
                routes.selected_indices,
                dtype=torch.bool,
            )
            return CapacityDecision(routes=routes, overflow_mask=overflow_mask)

        selected_experts = routes.selected_indices - 1
        selected_is_expert = selected_experts >= 0
        # Clamp anchor indices only for the gather; selected_is_expert prevents
        # the synthetic column-zero lookup from affecting anchor decisions.
        availability = torch.gather(
            expert_availability,
            dim=-1,
            index=selected_experts.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
        overflow_mask = selected_is_expert & ~availability
        if not overflow_mask.any().item():
            return CapacityDecision(routes=routes, overflow_mask=overflow_mask)

        effective_indices = routes.selected_indices.masked_fill(overflow_mask, 0)
        effective_scores = routes.selected_scores.masked_fill(overflow_mask, 0.0)
        effective_routes = RouteDecision(
            selected_indices=effective_indices,
            selected_scores=effective_scores,
            expert_ids=routes.expert_ids,
        )
        return CapacityDecision(
            routes=effective_routes,
            overflow_mask=overflow_mask,
        )


class Top2ToTop1Capacity:
    """Degrade unavailable approved pairs to one specialist or the anchor.

    For an approved ranked pair ``(i, j)``, explicit availability produces:

    - ``(i, j)`` with arity two when both specialists are available;
    - the available member with arity one when exactly one is available;
    - the anchor with arity zero when neither is available.

    If only the second-ranked specialist survives, it is promoted to execution
    slot zero without changing its candidate identity or score. The policy
    never searches outside the already selected pair, estimates capacity, or
    changes anchor-vetoed decisions. Omitting availability declares all
    specialists available.

    This is deterministic inference infrastructure rather than a paper-derived
    load-balancing algorithm.
    """

    policy_name = "top2_to_top1"

    def __call__(
        self,
        routes: Top2RouteDecision,
        *,
        expert_availability: Tensor | None = None,
    ) -> Top2CapacityDecision:
        """Return packed effective routes and pair-shaped overflow metadata."""

        if not isinstance(routes, Top2RouteDecision):
            raise TypeError("routes must be a Top2RouteDecision.")
        expert_count = len(routes.expert_ids)
        token_shape = routes.selected_indices.shape[:-1]
        expected_shape = token_shape + (expert_count,)

        if expert_availability is not None:
            if not isinstance(expert_availability, Tensor):
                raise TypeError("expert_availability must be a torch.Tensor.")
            if expert_availability.dtype != torch.bool:
                raise TypeError("expert_availability must use torch.bool.")
            if expert_availability.shape != expected_shape:
                raise ValueError(
                    "expert_availability must have route token shape plus one "
                    f"expert dimension: {tuple(expert_availability.shape)} != "
                    f"{tuple(expected_shape)}."
                )
            if expert_availability.device != routes.selected_indices.device:
                raise ValueError(
                    "expert_availability and routes must share one device."
                )

        approved_pair = routes.expert_pair_mask.unsqueeze(-1)
        if expert_count == 0:
            selected_available = torch.zeros_like(
                routes.selected_indices,
                dtype=torch.bool,
            )
        elif expert_availability is None:
            selected_available = torch.ones_like(
                routes.selected_indices,
                dtype=torch.bool,
            )
        else:
            # Anchor indices are clamped only for a safe gather. approved_pair
            # excludes every anchor-containing decision from capacity handling.
            selected_experts = routes.selected_indices - 1
            selected_available = torch.gather(
                expert_availability,
                dim=-1,
                index=selected_experts.clamp_min(0),
            )

        active_slots = approved_pair & selected_available
        overflow_mask = approved_pair & ~selected_available
        first_active = active_slots[..., 0]
        second_active = active_slots[..., 1]

        effective_indices = torch.zeros_like(routes.selected_indices)
        effective_scores = torch.zeros_like(routes.selected_scores)
        first_survivor_indices = torch.where(
            first_active,
            routes.selected_indices[..., 0],
            routes.selected_indices[..., 1],
        )
        first_survivor_scores = torch.where(
            first_active,
            routes.selected_scores[..., 0],
            routes.selected_scores[..., 1],
        )
        any_active = first_active | second_active
        effective_indices[..., 0] = torch.where(
            any_active,
            first_survivor_indices,
            0,
        )
        effective_scores[..., 0] = torch.where(
            any_active,
            first_survivor_scores,
            0.0,
        )
        both_active = first_active & second_active
        effective_indices[..., 1] = torch.where(
            both_active,
            routes.selected_indices[..., 1],
            0,
        )
        effective_scores[..., 1] = torch.where(
            both_active,
            routes.selected_scores[..., 1],
            0.0,
        )
        route_arity = active_slots.sum(dim=-1, dtype=torch.long)

        return Top2CapacityDecision(
            effective_indices=effective_indices,
            effective_scores=effective_scores,
            route_arity=route_arity,
            expert_ids=routes.expert_ids,
            overflow_mask=overflow_mask,
        )


class AdaptiveToAnchorCapacity:
    """Degrade adaptive top-1/top-2 routes under explicit availability.

    POL-003 already declares whether each token requests zero, one, or two
    ranked specialists. This BAL-002-compatible policy checks only those
    requested slots, packs surviving specialists from the left, and falls back
    to the always-available anchor when none survive. It never activates an
    unrequested expert or searches beyond the retained pair.
    """

    policy_name = "adaptive_to_anchor"

    def __call__(
        self,
        routes: AdaptiveRouteDecision,
        *,
        expert_availability: Tensor | None = None,
    ) -> Top2CapacityDecision:
        """Return capacity-approved pair slots for adaptive selection."""

        if not isinstance(routes, AdaptiveRouteDecision):
            raise TypeError("routes must be an AdaptiveRouteDecision.")
        expert_count = len(routes.expert_ids)
        token_shape = routes.route_arity.shape
        expected_shape = token_shape + (expert_count,)
        if expert_availability is not None:
            if not isinstance(expert_availability, Tensor):
                raise TypeError("expert_availability must be a torch.Tensor.")
            if expert_availability.dtype != torch.bool:
                raise TypeError("expert_availability must use torch.bool.")
            if expert_availability.shape != expected_shape:
                raise ValueError(
                    "expert_availability must have route token shape plus one "
                    f"expert dimension: {tuple(expert_availability.shape)} != "
                    f"{tuple(expected_shape)}."
                )
            if expert_availability.device != routes.ranked_indices.device:
                raise ValueError(
                    "expert_availability and routes must share one device."
                )

        slot_indices = torch.arange(
            2,
            dtype=torch.long,
            device=routes.route_arity.device,
        )
        requested_slots = slot_indices < routes.route_arity.unsqueeze(-1)
        if expert_count == 0:
            selected_available = torch.zeros_like(
                routes.ranked_indices,
                dtype=torch.bool,
            )
        elif expert_availability is None:
            selected_available = torch.ones_like(
                routes.ranked_indices,
                dtype=torch.bool,
            )
        else:
            selected_available = torch.gather(
                expert_availability,
                dim=-1,
                index=(routes.ranked_indices - 1).clamp_min(0),
            )

        active_slots = requested_slots & selected_available
        overflow_mask = requested_slots & ~selected_available
        first_active = active_slots[..., 0]
        second_active = active_slots[..., 1]
        any_active = first_active | second_active
        both_active = first_active & second_active

        effective_indices = torch.zeros_like(routes.ranked_indices)
        effective_scores = torch.zeros_like(routes.ranked_scores)
        first_survivor_indices = torch.where(
            first_active,
            routes.ranked_indices[..., 0],
            routes.ranked_indices[..., 1],
        )
        first_survivor_scores = torch.where(
            first_active,
            routes.ranked_scores[..., 0],
            routes.ranked_scores[..., 1],
        )
        effective_indices[..., 0] = torch.where(
            any_active,
            first_survivor_indices,
            0,
        )
        effective_scores[..., 0] = torch.where(
            any_active,
            first_survivor_scores,
            0.0,
        )
        effective_indices[..., 1] = torch.where(
            both_active,
            routes.ranked_indices[..., 1],
            0,
        )
        effective_scores[..., 1] = torch.where(
            both_active,
            routes.ranked_scores[..., 1],
            0.0,
        )
        return Top2CapacityDecision(
            effective_indices=effective_indices,
            effective_scores=effective_scores,
            route_arity=active_slots.sum(dim=-1, dtype=torch.long),
            expert_ids=routes.expert_ids,
            overflow_mask=overflow_mask,
        )


def _normalize_expert_ids(expert_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Validate stable public expert identifiers without importing internals."""

    if not isinstance(expert_ids, tuple):
        raise TypeError("expert_ids must be a tuple.")
    normalized: list[str] = []
    for expert_id in expert_ids:
        if not isinstance(expert_id, str) or not expert_id.strip():
            raise ValueError("expert_ids must contain non-empty strings.")
        normalized.append(expert_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must not contain duplicates.")
    return tuple(normalized)
