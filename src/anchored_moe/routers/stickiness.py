"""Explicit cross-layer stabilization for anchored top-1 routes.

RTR-008 reduces adjacent-layer route churn without storing request state in the
model or router. The caller supplies the previous layer's route decision, and
the stabilizer compares that incumbent with the current layer's calibrated
scores. A route changes only when the current baseline winner exceeds the
incumbent by more than a configured margin.

References:
    Dai et al., "StableMoE: Stable Routing Strategy for Mixture of Experts."
    https://aclanthology.org/2022.acl-long.489/

Differences from the paper:
    StableMoE distills learned assignments into a frozen deployment router.
    RTR-008 instead applies a deterministic, inference-time switching margin
    between adjacent converted layers. It introduces no learned router,
    persistent request state, balancing objective, or expert-training stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, ClassVar, Sequence, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from anchored_moe.policies.selection import RouteDecision


class CrossLayerStickinessError(ValueError):
    """Raised when RTR-008 cannot align adjacent-layer route information."""


@dataclass(frozen=True)
class CrossLayerStickinessConfig:
    """Configuration for deterministic anchored top-1 route stickiness.

    Args:
        switch_margin:
            Finite non-negative evidence gap required to replace an available
            previous route. A value of zero disables stickiness and reproduces
            :class:`AnchoredTop1Selection` exactly.
    """

    strategy: ClassVar[str] = "cross_layer_top1_stickiness"
    switch_margin: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize the public switching margin."""

        if isinstance(self.switch_margin, bool) or not isinstance(
            self.switch_margin,
            (int, float),
        ):
            raise TypeError("switch_margin must be a real number.")
        normalized_margin = float(self.switch_margin)
        if not isfinite(normalized_margin) or normalized_margin < 0:
            raise ValueError(
                "switch_margin must be finite and greater than or equal to zero."
            )
        object.__setattr__(self, "switch_margin", normalized_margin)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible strategy configuration."""

        return {
            "strategy": self.strategy,
            "switch_margin": self.switch_margin,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> CrossLayerStickinessConfig:
        """Strictly reconstruct a serialized strategy configuration.

        Raises:
            TypeError: If ``value`` is not a mapping.
            ValueError: If fields or the strategy name do not match RTR-008.
        """

        if not isinstance(value, dict):
            raise TypeError("Cross-layer stickiness config must be a dictionary.")
        expected_fields = {"strategy", "switch_margin"}
        if set(value) != expected_fields:
            raise ValueError(
                "Cross-layer stickiness config fields must be exactly "
                f"{sorted(expected_fields)}."
            )
        if value["strategy"] != cls.strategy:
            raise ValueError("Unsupported cross-layer stickiness strategy.")
        return cls(switch_margin=value["switch_margin"])


@dataclass(frozen=True)
class CrossLayerStickinessResult:
    """Auditable result of one explicit adjacent-layer stabilization step.

    Attributes:
        routes:
            Effective current-layer route after applying the switching margin.
        baseline_routes:
            Ordinary anchored top-1 decision before stabilization.
        previous_aligned_indices:
            Previous routes expressed in the current candidate order. ``-1``
            marks a previous expert that is absent from the current layer, or
            every token when no previous decision was supplied.
        previous_available_mask:
            Tokens whose previous anchor or expert exists in the current layer.
        retained_previous_mask:
            Tokens for which the effective route equals an available previous
            route. This includes incumbents that also win without stabilization.
    """

    routes: RouteDecision
    baseline_routes: RouteDecision
    previous_aligned_indices: Tensor
    previous_available_mask: Tensor
    retained_previous_mask: Tensor

    def __post_init__(self) -> None:
        """Validate route identity, audit-mask shape, dtype, and device."""

        from anchored_moe.policies.selection import RouteDecision

        if not isinstance(self.routes, RouteDecision):
            raise TypeError("routes must be a RouteDecision.")
        if not isinstance(self.baseline_routes, RouteDecision):
            raise TypeError("baseline_routes must be a RouteDecision.")
        if self.routes.expert_ids != self.baseline_routes.expert_ids:
            raise CrossLayerStickinessError(
                "Effective and baseline routes must share expert_ids."
            )
        if (
            self.routes.selected_indices.shape
            != self.baseline_routes.selected_indices.shape
        ):
            raise CrossLayerStickinessError(
                "Effective and baseline routes must share one token shape."
            )

        token_shape = self.routes.selected_indices.shape
        device = self.routes.selected_indices.device
        if not isinstance(self.previous_aligned_indices, Tensor):
            raise TypeError("previous_aligned_indices must be a torch.Tensor.")
        if self.previous_aligned_indices.dtype != torch.long:
            raise TypeError("previous_aligned_indices must use torch.long.")
        if self.previous_aligned_indices.shape != token_shape:
            raise CrossLayerStickinessError(
                "previous_aligned_indices must have the route token shape."
            )
        for field_name in (
            "previous_available_mask",
            "retained_previous_mask",
        ):
            mask = getattr(self, field_name)
            if not isinstance(mask, Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor.")
            if mask.dtype != torch.bool:
                raise TypeError(f"{field_name} must use torch.bool.")
            if mask.shape != token_shape:
                raise CrossLayerStickinessError(
                    f"{field_name} must have the route token shape."
                )
            if mask.device != device:
                raise CrossLayerStickinessError(
                    f"{field_name} must share the route device."
                )
        if self.previous_aligned_indices.device != device:
            raise CrossLayerStickinessError(
                "previous_aligned_indices must share the route device."
            )
        if torch.any(
            self.retained_previous_mask & ~self.previous_available_mask
        ).item():
            raise CrossLayerStickinessError(
                "An unavailable previous route cannot be retained."
            )
        if torch.any(
            self.previous_available_mask
            & (
                (self.previous_aligned_indices < 0)
                | (self.previous_aligned_indices > len(self.routes.expert_ids))
            )
        ).item():
            raise CrossLayerStickinessError(
                "Available previous routes must use current candidate indices."
            )
        if torch.any(
            ~self.previous_available_mask
            & (self.previous_aligned_indices != -1)
        ).item():
            raise CrossLayerStickinessError(
                "Unavailable previous routes must use index -1."
            )
        expected_retained = self.previous_available_mask & (
            self.routes.selected_indices == self.previous_aligned_indices
        )
        if not torch.equal(self.retained_previous_mask, expected_retained):
            raise CrossLayerStickinessError(
                "retained_previous_mask does not match the effective routes."
            )

    @property
    def stabilized_mask(self) -> Tensor:
        """Return tokens changed relative to ordinary current-layer top-1."""

        return (
            self.routes.selected_indices
            != self.baseline_routes.selected_indices
        )

    @property
    def switch_mask(self) -> Tensor:
        """Return tokens that switched away from an available previous route."""

        return self.previous_available_mask & (
            self.routes.selected_indices != self.previous_aligned_indices
        )

    @property
    def fallback_mask(self) -> Tensor:
        """Return tokens routed normally because no prior candidate is available."""

        return ~self.previous_available_mask


class CrossLayerTop1Stickiness:
    r"""Stabilize anchored top-1 decisions across adjacent converted layers.

    Let ``b_l`` be the ordinary anchored top-1 winner at layer ``l``, ``p`` the
    previous route aligned to the current layer, and ``s_l(c)`` the current
    calibrated score of candidate ``c`` with ``s_l(anchor) = 0``. For positive
    switching margin ``m``, RTR-008 selects

    .. math::

        r_l =
        \begin{cases}
        b_l, & p\ \text{is unavailable},\\
        b_l, & s_l(b_l) - s_l(p) > m,\\
        p,   & \text{otherwise}.
        \end{cases}

    A zero margin is an explicit disabled mode that returns baseline anchored
    top-1 exactly, including anchor-first and append-order tie-breaking. The
    previous decision is never stored; callers own route sequencing and may
    discard it at any request, batch, or generation boundary.

    References:
        Dai et al., "StableMoE: Stable Routing Strategy for Mixture of Experts."
        https://aclanthology.org/2022.acl-long.489/

    Differences from the paper:
        StableMoE learns and distills a deployment router. This class applies a
        project-specific deterministic switching margin to already calibrated
        adjacent-layer scores and has no training or persistent state.
    """

    def __init__(
        self,
        config: CrossLayerStickinessConfig | None = None,
    ) -> None:
        """Create a stabilizer from validated public configuration."""

        from anchored_moe.policies.selection import AnchoredTop1Selection

        self.config = config or CrossLayerStickinessConfig()
        if not isinstance(self.config, CrossLayerStickinessConfig):
            raise TypeError("config must be a CrossLayerStickinessConfig.")
        self._baseline_policy = AnchoredTop1Selection()

    def __call__(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
        previous_routes: RouteDecision | None,
    ) -> CrossLayerStickinessResult:
        """Return stabilized routes and explicit audit metadata.

        Args:
            expert_scores:
                Current-layer calibrated scores with shape
                ``token_shape + (K,)``.
            expert_ids:
                Unique current-layer expert IDs in score-column order.
            previous_routes:
                Previous layer's top-1 decision with the same token shape and
                device, or ``None`` at an explicit sequence boundary.

        Returns:
            A :class:`CrossLayerStickinessResult` containing baseline and
            effective decisions plus availability and retention masks.

        Raises:
            TypeError:
                If the previous decision has the wrong public type.
            CrossLayerStickinessError:
                If adjacent decisions have incompatible shapes or devices.
            ValueError:
                If current scores or expert identifiers are invalid.
        """

        baseline = self._baseline_policy(
            expert_scores,
            expert_ids=expert_ids,
        )
        token_shape = baseline.selected_indices.shape
        device = baseline.selected_indices.device
        if previous_routes is None:
            unavailable_indices = torch.full(
                token_shape,
                -1,
                dtype=torch.long,
                device=device,
            )
            unavailable_mask = torch.zeros(
                token_shape,
                dtype=torch.bool,
                device=device,
            )
            return CrossLayerStickinessResult(
                routes=baseline,
                baseline_routes=baseline,
                previous_aligned_indices=unavailable_indices,
                previous_available_mask=unavailable_mask,
                retained_previous_mask=unavailable_mask,
            )
        from anchored_moe.policies.selection import RouteDecision

        if not isinstance(previous_routes, RouteDecision):
            raise TypeError("previous_routes must be a RouteDecision or None.")
        if previous_routes.selected_indices.shape != token_shape:
            raise CrossLayerStickinessError(
                "previous_routes must have the current route token shape."
            )
        if previous_routes.selected_indices.device != device:
            raise CrossLayerStickinessError(
                "previous_routes and current scores must share one device."
            )

        aligned_previous, previous_available = _align_previous_indices(
            previous_routes,
            current_expert_ids=baseline.expert_ids,
        )
        if self.config.switch_margin == 0:
            retained = previous_available & (
                baseline.selected_indices == aligned_previous
            )
            return CrossLayerStickinessResult(
                routes=baseline,
                baseline_routes=baseline,
                previous_aligned_indices=aligned_previous,
                previous_available_mask=previous_available,
                retained_previous_mask=retained,
            )

        anchor_scores = expert_scores.new_zeros(
            expert_scores.shape[:-1] + (1,)
        )
        candidate_scores = torch.cat((anchor_scores, expert_scores), dim=-1)
        safe_previous_indices = aligned_previous.clamp_min(0)
        incumbent_scores = torch.gather(
            candidate_scores,
            dim=-1,
            index=safe_previous_indices.unsqueeze(-1),
        ).squeeze(-1)
        baseline_scores = baseline.selected_scores

        # Decisions are discrete, so compare detached float64 gaps. Promoting
        # lower-precision scores prevents rounding the configured Python margin
        # to a coarser deployed dtype; an infinite subtraction gap can only
        # arise from two finite extreme float64 scores and correctly forces a
        # switch.
        evidence_gap = (
            baseline_scores.detach().to(dtype=torch.float64)
            - incumbent_scores.detach().to(dtype=torch.float64)
        )
        baseline_beats_margin = (
            evidence_gap > self.config.switch_margin
        )
        keep_previous = previous_available & ~baseline_beats_margin
        selected_indices = torch.where(
            keep_previous,
            aligned_previous,
            baseline.selected_indices,
        )
        selected_scores = torch.gather(
            candidate_scores,
            dim=-1,
            index=selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        routes = RouteDecision(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            expert_ids=baseline.expert_ids,
        )
        retained = previous_available & (
            selected_indices == aligned_previous
        )
        return CrossLayerStickinessResult(
            routes=routes,
            baseline_routes=baseline,
            previous_aligned_indices=aligned_previous,
            previous_available_mask=previous_available,
            retained_previous_mask=retained,
        )


def _align_previous_indices(
    previous_routes: RouteDecision,
    *,
    current_expert_ids: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    """Map previous candidate indices into the current expert order.

    Candidate zero always maps to the immutable anchor. A previous specialist
    missing from the current layer maps to ``-1`` and is explicitly unavailable.
    The lookup is constructed in previous candidate order, so tensor gathering
    supports arbitrary token-leading shapes without inspecting token contents.
    """

    current_positions = {
        expert_id: index + 1
        for index, expert_id in enumerate(current_expert_ids)
    }
    mapping = [0]
    mapping.extend(
        current_positions.get(expert_id, -1)
        for expert_id in previous_routes.expert_ids
    )
    mapping_tensor = torch.tensor(
        mapping,
        dtype=torch.long,
        device=previous_routes.selected_indices.device,
    )
    aligned = mapping_tensor[previous_routes.selected_indices]
    available = aligned >= 0
    return aligned, available
