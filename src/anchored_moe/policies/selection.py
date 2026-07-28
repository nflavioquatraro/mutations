"""Deterministic anchored top-1, top-2, and adaptive selection policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

import torch
from torch import Tensor

from anchored_moe.registry.compatibility import MixtureCompatibilityIndex


@dataclass(frozen=True)
class RouteDecision:
    """Structured per-token result returned by a selection policy.

    Candidate index ``0`` represents the immutable anchor. Candidate index
    ``i + 1`` represents ``expert_ids[i]``. ``selected_indices`` and
    ``selected_scores`` have the same leading shape as the input score tensor,
    excluding its final expert dimension.

    Attributes:
        selected_indices:
            Integer tensor containing one candidate index per token.
        selected_scores:
            Floating-point tensor containing the winning calibrated score.
        expert_ids:
            Expert identifiers in the stable order used by candidate indices.
    """

    selected_indices: Tensor
    selected_scores: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shape, dtype, device, identifier, and index invariants."""

        if not isinstance(self.selected_indices, Tensor):
            raise TypeError("selected_indices must be a torch.Tensor.")
        if self.selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must use torch.long.")
        if not isinstance(self.selected_scores, Tensor):
            raise TypeError("selected_scores must be a torch.Tensor.")
        if not self.selected_scores.is_floating_point():
            raise TypeError("selected_scores must use a floating-point dtype.")
        if self.selected_indices.shape != self.selected_scores.shape:
            raise ValueError(
                "selected_indices and selected_scores must have the same shape."
            )
        if self.selected_indices.device != self.selected_scores.device:
            raise ValueError(
                "selected_indices and selected_scores must share one device."
            )

        normalized_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", normalized_ids)
        if self.selected_indices.numel():
            minimum = int(self.selected_indices.min().item())
            maximum = int(self.selected_indices.max().item())
            if minimum < 0 or maximum > len(normalized_ids):
                raise ValueError(
                    "selected_indices contains a candidate outside the anchor "
                    f"and {len(normalized_ids)} declared experts."
                )
        if not torch.isfinite(self.selected_scores).all().item():
            raise ValueError("selected_scores must contain only finite values.")

    @property
    def anchor_mask(self) -> Tensor:
        """Return a boolean mask for tokens assigned to the anchor."""

        return self.selected_indices == 0

    @property
    def expert_mask(self) -> Tensor:
        """Return a boolean mask for tokens assigned to any expert."""

        return self.selected_indices != 0

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return the token mask for one declared expert identifier.

        Args:
            expert_id: Exact identifier from :attr:`expert_ids`.

        Raises:
            KeyError: If the identifier was not a candidate in this decision.
        """

        try:
            candidate_index = self.expert_ids.index(expert_id) + 1
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        return self.selected_indices == candidate_index


@dataclass(frozen=True)
class Top2RouteDecision:
    """Ranked candidate pair returned by anchor-veto top-2 selection.

    Candidate index ``0`` represents the immutable anchor. Candidate index
    ``i + 1`` represents ``expert_ids[i]``. The final dimension of
    ``selected_indices`` and ``selected_scores`` has size two and stores the
    candidates in descending score order.

    An empty registry has only one distinct candidate. In that case both
    entries are the anchor, which preserves the fixed pair-shaped contract
    while :attr:`expert_pair_mask` remains false.

    Attributes:
        selected_indices:
            Integer tensor with shape ``token_shape + (2,)``.
        selected_scores:
            Floating-point tensor with the same shape and ranked scores.
        expert_ids:
            Expert identifiers in the stable order used by candidate indices.
    """

    selected_indices: Tensor
    selected_scores: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate pair shape, dtype, device, identifier, and index invariants."""

        if not isinstance(self.selected_indices, Tensor):
            raise TypeError("selected_indices must be a torch.Tensor.")
        if self.selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must use torch.long.")
        if not isinstance(self.selected_scores, Tensor):
            raise TypeError("selected_scores must be a torch.Tensor.")
        if not self.selected_scores.is_floating_point():
            raise TypeError("selected_scores must use a floating-point dtype.")
        if self.selected_indices.shape != self.selected_scores.shape:
            raise ValueError(
                "selected_indices and selected_scores must have the same shape."
            )
        if self.selected_indices.ndim < 1 or self.selected_indices.shape[-1] != 2:
            raise ValueError(
                "top-2 selected_indices and selected_scores must have a final "
                "dimension of size two."
            )
        if self.selected_indices.device != self.selected_scores.device:
            raise ValueError(
                "selected_indices and selected_scores must share one device."
            )

        normalized_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", normalized_ids)
        if self.selected_indices.numel():
            minimum = int(self.selected_indices.min().item())
            maximum = int(self.selected_indices.max().item())
            if minimum < 0 or maximum > len(normalized_ids):
                raise ValueError(
                    "selected_indices contains a candidate outside the anchor "
                    f"and {len(normalized_ids)} declared experts."
                )
        if not torch.isfinite(self.selected_scores).all().item():
            raise ValueError("selected_scores must contain only finite values.")

    @property
    def anchor_mask(self) -> Tensor:
        """Return tokens vetoed because the anchor appears in the ranked pair."""

        return torch.any(self.selected_indices == 0, dim=-1)

    @property
    def expert_pair_mask(self) -> Tensor:
        """Return tokens whose two ranked candidates are both specialists."""

        return ~self.anchor_mask

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return tokens whose ranked pair contains one declared expert."""

        try:
            candidate_index = self.expert_ids.index(expert_id) + 1
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        return torch.any(self.selected_indices == candidate_index, dim=-1)


@dataclass(frozen=True)
class AdaptiveRouteDecision:
    """Ranked specialists and per-token anchor/top-1/top-2 route arity.

    ``ranked_indices`` and ``ranked_scores`` have shape
    ``token_shape + (2,)``. Candidate index ``i + 1`` represents
    ``expert_ids[i]``; candidate zero is reserved for padding when fewer than
    two specialists exist. Ranked specialists remain visible even when the
    anchor is selected so threshold behavior is auditable.

    ``route_arity`` has ``token_shape`` and contains:

    - ``0`` when the anchor should execute;
    - ``1`` when only the first ranked specialist should execute;
    - ``2`` when both ranked specialists should execute.

    Attributes:
        ranked_indices:
            Stable specialist ranking padded to a final dimension of size two.
        ranked_scores:
            Calibrated scores aligned with ``ranked_indices``; padding scores
            are zero.
        route_arity:
            Integer tensor containing the active specialist count per token.
        expert_ids:
            Expert identifiers in the stable order used by candidate indices.
    """

    ranked_indices: Tensor
    ranked_scores: Tensor
    route_arity: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate pair shape, route arity, identifiers, and active slots."""

        if not isinstance(self.ranked_indices, Tensor):
            raise TypeError("ranked_indices must be a torch.Tensor.")
        if self.ranked_indices.dtype != torch.long:
            raise TypeError("ranked_indices must use torch.long.")
        if not isinstance(self.ranked_scores, Tensor):
            raise TypeError("ranked_scores must be a torch.Tensor.")
        if not self.ranked_scores.is_floating_point():
            raise TypeError("ranked_scores must use a floating-point dtype.")
        if self.ranked_indices.shape != self.ranked_scores.shape:
            raise ValueError(
                "ranked_indices and ranked_scores must have the same shape."
            )
        if self.ranked_indices.ndim < 1 or self.ranked_indices.shape[-1] != 2:
            raise ValueError(
                "adaptive ranked_indices and ranked_scores must have a final "
                "dimension of size two."
            )
        if not isinstance(self.route_arity, Tensor):
            raise TypeError("route_arity must be a torch.Tensor.")
        if self.route_arity.dtype != torch.long:
            raise TypeError("route_arity must use torch.long.")
        if self.route_arity.shape != self.ranked_indices.shape[:-1]:
            raise ValueError(
                "route_arity must have the ranked-pair token-leading shape."
            )
        if not (
            self.ranked_indices.device
            == self.ranked_scores.device
            == self.route_arity.device
        ):
            raise ValueError(
                "ranked_indices, ranked_scores, and route_arity must share "
                "one device."
            )

        normalized_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", normalized_ids)
        if self.ranked_indices.numel():
            minimum = int(self.ranked_indices.min().item())
            maximum = int(self.ranked_indices.max().item())
            if minimum < 0 or maximum > len(normalized_ids):
                raise ValueError(
                    "ranked_indices contains a candidate outside the padding "
                    f"index and {len(normalized_ids)} declared experts."
                )
        if not torch.isfinite(self.ranked_scores).all().item():
            raise ValueError("ranked_scores must contain only finite values.")
        if self.route_arity.numel():
            minimum_arity = int(self.route_arity.min().item())
            maximum_arity = int(self.route_arity.max().item())
            if minimum_arity < 0 or maximum_arity > 2:
                raise ValueError("route_arity values must be zero, one, or two.")

        ranked_count = min(2, len(normalized_ids))
        if ranked_count:
            if torch.any(self.ranked_indices[..., :ranked_count] == 0).item():
                raise ValueError(
                    "available specialists must occupy every ranked slot "
                    "before padding."
                )
        if ranked_count < 2:
            if torch.any(self.ranked_indices[..., ranked_count:] != 0).item():
                raise ValueError(
                    "missing specialist ranking slots must use padding index zero."
                )
            if torch.any(self.ranked_scores[..., ranked_count:] != 0).item():
                raise ValueError(
                    "missing specialist ranking slots must use padding score zero."
                )
        if ranked_count == 2 and torch.any(
            self.ranked_indices[..., 0] == self.ranked_indices[..., 1]
        ).item():
            raise ValueError(
                "adaptive rankings must contain two distinct specialists."
            )

        active_slots = self._active_slot_mask()
        if torch.any(self.ranked_indices[active_slots] == 0).item():
            raise ValueError("active adaptive routes cannot use padding index zero.")

    @property
    def anchor_mask(self) -> Tensor:
        """Return tokens assigned to the immutable anchor."""

        return self.route_arity == 0

    @property
    def top1_mask(self) -> Tensor:
        """Return tokens assigned to exactly one specialist."""

        return self.route_arity == 1

    @property
    def top2_mask(self) -> Tensor:
        """Return tokens assigned to two specialists."""

        return self.route_arity == 2

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return tokens where one declared expert is an active route."""

        try:
            candidate_index = self.expert_ids.index(expert_id) + 1
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        active_matches = (
            self.ranked_indices == candidate_index
        ) & self._active_slot_mask()
        return torch.any(active_matches, dim=-1)

    def _active_slot_mask(self) -> Tensor:
        """Return pair slots enabled by each token's route arity."""

        slot_indices = torch.arange(
            2,
            dtype=torch.long,
            device=self.route_arity.device,
        )
        return slot_indices < self.route_arity.unsqueeze(-1)


class SelectionPolicy(ABC):
    """Interface for choosing anchor or expert candidates from router scores.

    A policy consumes calibrated expert scores whose final dimension follows a
    declared stable expert order. It returns route metadata only; token
    dispatch, expert execution, capacity handling, and telemetry are separate
    responsibilities.
    """

    policy_name: ClassVar[str]

    def __call__(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> RouteDecision | Top2RouteDecision | AdaptiveRouteDecision:
        """Delegate to :meth:`select` for ordinary callable use."""

        return self.select(expert_scores, expert_ids=expert_ids)

    @abstractmethod
    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> RouteDecision | Top2RouteDecision | AdaptiveRouteDecision:
        """Choose candidate routes per token from calibrated expert scores."""


class AnchoredTop1Selection(SelectionPolicy):
    """Select the anchor or one expert using fixed-zero anchor evidence.

    Given expert scores ``G[..., K]``, the policy prepends the permanent anchor
    score and computes

    ``r(h) = argmax([0, g_1(h), ..., g_K(h)])``.

    The anchor is candidate zero, so PyTorch's first-maximum behavior makes the
    anchor win every tie at zero. Expert columns retain router append order, so
    equal positive expert scores choose the earliest appended expert. The
    implementation rejects NaN and infinity instead of allowing backend- or
    order-dependent routing.

    This class only selects routes. It does not check expert capacity, execute
    MLPs, or attach itself to :class:`anchored_moe.modeling.RoutedMLP`.
    """

    policy_name = "anchored_top1"
    anchor_score = 0.0

    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> RouteDecision:
        """Return deterministic top-1 decisions for every score vector.

        Args:
            expert_scores:
                Floating-point tensor with shape ``[..., K]``. The last
                dimension follows ``expert_ids`` order and may be empty for an
                empty registry.
            expert_ids:
                Unique, stable identifiers for the ``K`` expert columns.

        Returns:
            A :class:`RouteDecision` with shape ``expert_scores.shape[:-1]``.

        Raises:
            TypeError:
                If scores are not a floating-point tensor or identifiers are
                not a sequence of strings.
            ValueError:
                If shapes, identifiers, or score values are invalid.
        """

        if not isinstance(expert_scores, Tensor):
            raise TypeError("expert_scores must be a torch.Tensor.")
        if not expert_scores.is_floating_point():
            raise TypeError("expert_scores must use a floating-point dtype.")
        if expert_scores.ndim < 1:
            raise ValueError(
                "expert_scores must have a final expert-candidate dimension."
            )

        normalized_ids = _normalize_expert_ids(expert_ids)
        expert_count = expert_scores.shape[-1]
        if expert_count != len(normalized_ids):
            raise ValueError(
                "expert_scores final dimension must match expert_ids: "
                f"{expert_count} != {len(normalized_ids)}."
            )
        if not torch.isfinite(expert_scores).all().item():
            raise ValueError("expert_scores must contain only finite values.")

        anchor_scores = expert_scores.new_zeros(
            expert_scores.shape[:-1] + (1,)
        )
        # Candidate order is a semantic invariant: PyTorch argmax returns the
        # first maximum, giving anchor-first and append-order tie-breaking.
        candidate_scores = torch.cat((anchor_scores, expert_scores), dim=-1)
        selected_indices = torch.argmax(candidate_scores, dim=-1)
        selected_scores = torch.gather(
            candidate_scores,
            dim=-1,
            index=selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        return RouteDecision(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            expert_ids=normalized_ids,
        )


class AnchorVetoTop2Selection(SelectionPolicy):
    """Rank two candidates and allow specialists only when anchor is absent.

    Given calibrated expert scores ``G[..., K]``, the policy prepends the
    permanent anchor score and ranks

    ``[0, g_1(h), ..., g_K(h)]``.

    The anchor vetoes specialist execution whenever it appears in the first
    two positions. Consequently :attr:`Top2RouteDecision.expert_pair_mask` is
    true exactly when at least two specialists have scores strictly greater
    than zero. Stable sorting makes the anchor win score ties at zero and
    makes earlier appended experts win equal specialist-score ties.

    This class returns route metadata only. COM-001 computes mixture weights;
    sparse expert execution, capacity handling, pair compatibility, and output
    reconstruction remain separate responsibilities.
    """

    policy_name = "anchor_veto_top2"
    anchor_score = 0.0

    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> Top2RouteDecision:
        """Return deterministic ranked pairs and anchor-veto masks.

        Args:
            expert_scores:
                Floating-point tensor with shape ``[..., K]``. The last
                dimension follows ``expert_ids`` order and may be empty.
            expert_ids:
                Unique, stable identifiers for the ``K`` expert columns.

        Returns:
            A :class:`Top2RouteDecision` with pair tensors of shape
            ``expert_scores.shape[:-1] + (2,)``.

        Raises:
            TypeError:
                If scores are not a floating-point tensor or identifiers are
                not a sequence of strings.
            ValueError:
                If shapes, identifiers, or score values are invalid.
        """

        if not isinstance(expert_scores, Tensor):
            raise TypeError("expert_scores must be a torch.Tensor.")
        if not expert_scores.is_floating_point():
            raise TypeError("expert_scores must use a floating-point dtype.")
        if expert_scores.ndim < 1:
            raise ValueError(
                "expert_scores must have a final expert-candidate dimension."
            )

        normalized_ids = _normalize_expert_ids(expert_ids)
        expert_count = expert_scores.shape[-1]
        if expert_count != len(normalized_ids):
            raise ValueError(
                "expert_scores final dimension must match expert_ids: "
                f"{expert_count} != {len(normalized_ids)}."
            )
        if not torch.isfinite(expert_scores).all().item():
            raise ValueError("expert_scores must contain only finite values.")

        anchor_scores = expert_scores.new_zeros(
            expert_scores.shape[:-1] + (1,)
        )
        candidate_scores = torch.cat((anchor_scores, expert_scores), dim=-1)
        if candidate_scores.shape[-1] == 1:
            # Pair-shaped metadata makes downstream policy inspection uniform.
            # The duplicated anchor is never interpreted as two active routes.
            selected_indices = torch.zeros(
                expert_scores.shape[:-1] + (2,),
                dtype=torch.long,
                device=expert_scores.device,
            )
            selected_scores = anchor_scores.expand(
                expert_scores.shape[:-1] + (2,)
            )
        else:
            # Stable descending order preserves semantic candidate order:
            # anchor first, followed by experts in append order.
            ranked_indices = torch.argsort(
                candidate_scores,
                dim=-1,
                descending=True,
                stable=True,
            )
            selected_indices = ranked_indices[..., :2]
            selected_scores = torch.gather(
                candidate_scores,
                dim=-1,
                index=selected_indices,
            )

        return Top2RouteDecision(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            expert_ids=normalized_ids,
        )


class CompatibilityConstrainedTop2Selection(SelectionPolicy):
    """Choose the strongest explicitly compatible positive specialist pair.

    POL-005 ranks compatible pairs lexicographically by their descending score
    tuple ``(g_(1), g_(2))``. This preserves ordinary top-2 semantics: maximize
    the strongest member first, then the supporting second member. Exact score
    ties preserve expert append order and then pair enumeration order.

    A pair is eligible only when both experts share one explicit non-empty
    PKG-003 group and both calibrated scores are strictly positive. If no pair
    is eligible, the result contains the best specialist and the fixed-zero
    anchor in stable score order, so :attr:`Top2RouteDecision.anchor_mask`
    forces exact anchor execution. Experts with missing group metadata are
    therefore valid top-1 candidates but can never be mixed.

    Pair enumeration is ``O(K^2)`` in the worst case and intentionally targets
    small registries. Hierarchical routing or precomputed candidate
    shortlisting remains separate work.
    """

    policy_name = "global_compatibility_top2"
    anchor_score = 0.0

    def __init__(self, compatibility: MixtureCompatibilityIndex) -> None:
        """Bind selection to one immutable expert compatibility index."""

        if not isinstance(compatibility, MixtureCompatibilityIndex):
            raise TypeError(
                "compatibility must be a MixtureCompatibilityIndex."
            )
        self.compatibility = compatibility

    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> Top2RouteDecision:
        """Return the best compatible pair or an anchor-veto fallback.

        Args:
            expert_scores:
                Floating-point tensor with shape ``[..., K]`` in candidate
                expert order.
            expert_ids:
                Unique stable identifiers exactly matching the bound
                compatibility index. Order may differ because groups are
                realigned by identifier.

        Returns:
            Pair-shaped deterministic route metadata. A specialist pair is
            active only when both selected indices are nonzero.

        Raises:
            TypeError: If scores or identifiers have invalid types.
            ValueError:
                If scores, shapes, values, or the compatibility candidate set
                are invalid.
        """

        if not isinstance(expert_scores, Tensor):
            raise TypeError("expert_scores must be a torch.Tensor.")
        if not expert_scores.is_floating_point():
            raise TypeError("expert_scores must use a floating-point dtype.")
        if expert_scores.ndim < 1:
            raise ValueError(
                "expert_scores must have a final expert-candidate dimension."
            )

        normalized_ids = _normalize_expert_ids(expert_ids)
        expert_count = expert_scores.shape[-1]
        if expert_count != len(normalized_ids):
            raise ValueError(
                "expert_scores final dimension must match expert_ids: "
                f"{expert_count} != {len(normalized_ids)}."
            )
        if not torch.isfinite(expert_scores).all().item():
            raise ValueError("expert_scores must contain only finite values.")
        group_ids = self.compatibility.groups_for(normalized_ids)

        token_shape = expert_scores.shape[:-1]
        if expert_count == 0:
            selected_indices = torch.zeros(
                token_shape + (2,),
                dtype=torch.long,
                device=expert_scores.device,
            )
            selected_scores = expert_scores.new_zeros(token_shape + (2,))
            return Top2RouteDecision(
                selected_indices=selected_indices,
                selected_scores=selected_scores,
                expert_ids=normalized_ids,
            )

        # The fallback remains auditable: retain the strongest specialist next
        # to the anchor, but never expose two nonzero indices without a valid
        # compatibility group.
        ranked_specialists = torch.argsort(
            expert_scores,
            dim=-1,
            descending=True,
            stable=True,
        )
        best_index = ranked_specialists[..., 0]
        best_score = torch.gather(
            expert_scores,
            dim=-1,
            index=best_index.unsqueeze(-1),
        ).squeeze(-1)
        best_candidate = best_index + 1
        best_is_positive = best_score > self.anchor_score
        anchor_indices = torch.zeros_like(best_candidate)
        anchor_scores = torch.zeros_like(best_score)
        selected_indices = torch.stack(
            (
                torch.where(
                    best_is_positive,
                    best_candidate,
                    anchor_indices,
                ),
                torch.where(
                    best_is_positive,
                    anchor_indices,
                    best_candidate,
                ),
            ),
            dim=-1,
        )
        selected_scores = torch.stack(
            (
                torch.where(best_is_positive, best_score, anchor_scores),
                torch.where(best_is_positive, anchor_scores, best_score),
            ),
            dim=-1,
        )

        compatible_pairs = tuple(
            (first, second)
            for first in range(expert_count)
            for second in range(first + 1, expert_count)
            if group_ids[first] is not None
            and group_ids[first] == group_ids[second]
        )
        pair_found = torch.zeros(
            token_shape,
            dtype=torch.bool,
            device=expert_scores.device,
        )
        best_first_score = torch.zeros_like(best_score)
        best_second_score = torch.zeros_like(best_score)

        for first, second in compatible_pairs:
            first_raw = expert_scores[..., first]
            second_raw = expert_scores[..., second]
            # ``first`` precedes ``second`` in candidate order, so an exact
            # score tie keeps it in the first ranked slot.
            first_precedes = first_raw >= second_raw
            pair_first_score = torch.where(
                first_precedes,
                first_raw,
                second_raw,
            )
            pair_second_score = torch.where(
                first_precedes,
                second_raw,
                first_raw,
            )
            pair_first_index = torch.where(
                first_precedes,
                torch.full_like(best_candidate, first + 1),
                torch.full_like(best_candidate, second + 1),
            )
            pair_second_index = torch.where(
                first_precedes,
                torch.full_like(best_candidate, second + 1),
                torch.full_like(best_candidate, first + 1),
            )

            eligible = pair_second_score > self.anchor_score
            lexicographically_better = (
                (pair_first_score > best_first_score)
                | (
                    (pair_first_score == best_first_score)
                    & (pair_second_score > best_second_score)
                )
            )
            replace = eligible & (~pair_found | lexicographically_better)
            selected_indices[..., 0] = torch.where(
                replace,
                pair_first_index,
                selected_indices[..., 0],
            )
            selected_indices[..., 1] = torch.where(
                replace,
                pair_second_index,
                selected_indices[..., 1],
            )
            selected_scores[..., 0] = torch.where(
                replace,
                pair_first_score,
                selected_scores[..., 0],
            )
            selected_scores[..., 1] = torch.where(
                replace,
                pair_second_score,
                selected_scores[..., 1],
            )
            best_first_score = torch.where(
                replace,
                pair_first_score,
                best_first_score,
            )
            best_second_score = torch.where(
                replace,
                pair_second_score,
                best_second_score,
            )
            pair_found = pair_found | eligible

        return Top2RouteDecision(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            expert_ids=normalized_ids,
        )


class AdaptiveTop1Top2Selection(SelectionPolicy):
    """Choose anchor, one specialist, or two specialists by score evidence.

    For descending specialist scores ``g_(1)`` and ``g_(2)``, POL-003 uses

    - anchor when ``g_(1) <= 0``;
    - top-1 when ``g_(1) > 0`` and either no second specialist exists or
      ``g_(2) <= mix_threshold``;
    - top-2 when a second specialist exists and
      ``g_(2) > mix_threshold``.

    ``mix_threshold`` is finite and non-negative, so top-2 always requires two
    strictly positive specialists. An exact threshold tie remains top-1.
    Stable descending sorting preserves expert append order for score ties.

    This class returns selection metadata only. Sparse execution, mixture
    weighting, capacity handling, pair compatibility, and telemetry remain
    separate responsibilities.
    """

    policy_name = "adaptive_top1_top2"
    anchor_score = 0.0

    def __init__(self, mix_threshold: float = 0.0) -> None:
        """Create adaptive selection with a finite non-negative threshold."""

        if isinstance(mix_threshold, bool) or not isinstance(
            mix_threshold,
            (int, float),
        ):
            raise TypeError("mix_threshold must be a real number.")
        normalized_threshold = float(mix_threshold)
        if not isfinite(normalized_threshold) or normalized_threshold < 0:
            raise ValueError(
                "mix_threshold must be finite and greater than or equal to zero."
            )
        self.mix_threshold = normalized_threshold

    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> AdaptiveRouteDecision:
        """Return deterministic adaptive route arity for every score vector.

        Args:
            expert_scores:
                Floating-point tensor with shape ``[..., K]``. The final
                dimension follows ``expert_ids`` order and may be empty.
            expert_ids:
                Unique, stable identifiers for the ``K`` specialist columns.

        Returns:
            An :class:`AdaptiveRouteDecision` with pair-shaped rankings and
            token-shaped route arity.

        Raises:
            TypeError:
                If scores are not a floating-point tensor or identifiers are
                not a sequence of strings.
            ValueError:
                If shapes, identifiers, or score values are invalid.
        """

        if not isinstance(expert_scores, Tensor):
            raise TypeError("expert_scores must be a torch.Tensor.")
        if not expert_scores.is_floating_point():
            raise TypeError("expert_scores must use a floating-point dtype.")
        if expert_scores.ndim < 1:
            raise ValueError(
                "expert_scores must have a final expert-candidate dimension."
            )

        normalized_ids = _normalize_expert_ids(expert_ids)
        expert_count = expert_scores.shape[-1]
        if expert_count != len(normalized_ids):
            raise ValueError(
                "expert_scores final dimension must match expert_ids: "
                f"{expert_count} != {len(normalized_ids)}."
            )
        if not torch.isfinite(expert_scores).all().item():
            raise ValueError("expert_scores must contain only finite values.")

        token_shape = expert_scores.shape[:-1]
        route_arity = torch.zeros(
            token_shape,
            dtype=torch.long,
            device=expert_scores.device,
        )
        if expert_count == 0:
            ranked_indices = torch.zeros(
                token_shape + (2,),
                dtype=torch.long,
                device=expert_scores.device,
            )
            ranked_scores = expert_scores.new_zeros(token_shape + (2,))
        else:
            # Stable specialist-only ranking makes equal scores follow the
            # declared append order independently of the anchor threshold.
            all_ranked_indices = torch.argsort(
                expert_scores,
                dim=-1,
                descending=True,
                stable=True,
            )
            retained_count = min(2, expert_count)
            retained_indices = all_ranked_indices[..., :retained_count]
            retained_scores = torch.gather(
                expert_scores,
                dim=-1,
                index=retained_indices,
            )
            ranked_indices = retained_indices + 1
            ranked_scores = retained_scores
            if retained_count == 1:
                padding_indices = torch.zeros(
                    token_shape + (1,),
                    dtype=torch.long,
                    device=expert_scores.device,
                )
                padding_scores = expert_scores.new_zeros(token_shape + (1,))
                ranked_indices = torch.cat(
                    (ranked_indices, padding_indices),
                    dim=-1,
                )
                ranked_scores = torch.cat(
                    (ranked_scores, padding_scores),
                    dim=-1,
                )

            best_is_positive = ranked_scores[..., 0] > self.anchor_score
            route_arity = best_is_positive.to(dtype=torch.long)
            if expert_count >= 2:
                second_supports_mixing = (
                    ranked_scores[..., 1] > self.mix_threshold
                )
                route_arity = torch.where(
                    best_is_positive & second_supports_mixing,
                    torch.full_like(route_arity, 2),
                    route_arity,
                )

        return AdaptiveRouteDecision(
            ranked_indices=ranked_indices,
            ranked_scores=ranked_scores,
            route_arity=route_arity,
            expert_ids=normalized_ids,
        )


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate and defensively copy stable expert candidate identifiers."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(
        expert_ids,
        Sequence,
    ):
        raise TypeError("expert_ids must be a sequence of strings.")

    normalized: list[str] = []
    for expert_id in expert_ids:
        if (
            not isinstance(expert_id, str)
            or not expert_id
            or expert_id != expert_id.strip()
        ):
            raise ValueError(
                "expert_ids must contain non-empty strings without "
                "surrounding whitespace."
            )
        normalized.append(expert_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must not contain duplicates.")
    return tuple(normalized)
