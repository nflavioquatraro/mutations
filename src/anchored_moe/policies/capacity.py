"""Capacity policies for deterministic anchored fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anchored_moe.policies.selection import RouteDecision


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
