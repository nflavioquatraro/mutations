"""Deterministic anchored top-1 and anchor-veto top-2 selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor


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
    ) -> RouteDecision | Top2RouteDecision:
        """Delegate to :meth:`select` for ordinary callable use."""

        return self.select(expert_scores, expert_ids=expert_ids)

    @abstractmethod
    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> RouteDecision | Top2RouteDecision:
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
