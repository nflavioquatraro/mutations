"""Batch-level expert-choice routing for training and prompt prefill.

BAL-005 reverses ordinary token-choice routing: every specialist receives a
fixed-capacity bucket and selects its highest-affinity eligible tokens from the
current batch. A token may therefore be selected by zero, one, or several
experts. Valid tokens selected by no specialist are marked for immutable-anchor
fallback so the anchored framework never drops them.

References:
    Zhou et al., "Mixture-of-Experts with Expert Choice Routing."
    https://arxiv.org/abs/2202.09368

Differences from the paper:
    This initial research backend accepts an explicit integer expert capacity
    instead of deriving and rounding ``n * capacity_factor / expert_count``.
    It returns sparse indices rather than materializing the paper's dense
    one-hot permutation tensor. Padding masks and exact anchor-fallback metadata
    are project-specific additions. The backend plans assignments only; it does
    not execute experts, integrate with Trainer, communicate across processes,
    or route single-token autoregressive decoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor


class ExpertChoiceRoutingError(ValueError):
    """Raised when BAL-005 configuration, scores, or assignments are invalid."""


@dataclass(frozen=True)
class ExpertChoiceConfig:
    """Configuration for fixed-bucket expert-choice routing.

    Args:
        expert_capacity:
            Positive number of eligible tokens selected independently by every
            expert. Callers must choose a value no larger than the number of
            eligible tokens in each routed batch.

    The explicit bucket size is the paper's ``k``. Keeping it explicit avoids
    a hidden rounding rule when ``n * capacity_factor / expert_count`` is not
    integral and makes the executed batch budget directly auditable.
    """

    strategy: ClassVar[str] = "expert_choice"

    expert_capacity: int

    def __post_init__(self) -> None:
        """Reject booleans and non-positive bucket capacities."""

        if (
            isinstance(self.expert_capacity, bool)
            or not isinstance(self.expert_capacity, int)
            or self.expert_capacity <= 0
        ):
            raise ExpertChoiceRoutingError(
                "expert_capacity must be a positive integer."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible public configuration."""

        return {
            "expert_capacity": self.expert_capacity,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExpertChoiceConfig:
        """Strictly reconstruct a serialized BAL-005 configuration mapping."""

        if not isinstance(value, dict):
            raise ExpertChoiceRoutingError(
                "Expert-choice config must be a dictionary."
            )
        if set(value) != {"expert_capacity", "strategy"}:
            raise ExpertChoiceRoutingError(
                "Expert-choice config fields are incompatible."
            )
        if value["strategy"] != cls.strategy:
            raise ExpertChoiceRoutingError(
                "Unsupported expert-choice routing strategy."
            )
        return cls(expert_capacity=value["expert_capacity"])


@dataclass(frozen=True)
class ExpertChoiceDecision:
    """Immutable sparse assignment plan produced by BAL-005.

    ``selected_token_indices`` and ``selected_gates`` have shape ``(E, C)``,
    where ``E`` is the number of experts and ``C`` is the fixed expert
    capacity. Indices address the flattened original token dimensions, not a
    padding-compacted intermediate tensor.

    ``token_assignment_counts`` has the original token shape and records a
    value in ``0..E``. ``anchor_fallback_mask`` is true exactly for eligible
    tokens with zero specialist assignments. Padding positions are ineligible,
    receive zero assignments, and are not reported as anchor work.
    """

    selected_token_indices: Tensor
    selected_gates: Tensor
    token_assignment_counts: Tensor
    anchor_fallback_mask: Tensor
    eligible_token_mask: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate sparse capacity, token accounting, and fallback invariants."""

        expert_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", expert_ids)

        if (
            not isinstance(self.selected_token_indices, Tensor)
            or self.selected_token_indices.dtype != torch.long
            or self.selected_token_indices.ndim != 2
        ):
            raise ExpertChoiceRoutingError(
                "selected_token_indices must be a two-dimensional torch.long "
                "tensor."
            )
        if self.selected_token_indices.shape[0] != len(expert_ids):
            raise ExpertChoiceRoutingError(
                "selected_token_indices must contain one row per expert."
            )
        if self.selected_token_indices.shape[1] <= 0:
            raise ExpertChoiceRoutingError(
                "Expert-choice decisions require positive expert capacity."
            )
        if (
            not isinstance(self.selected_gates, Tensor)
            or not self.selected_gates.is_floating_point()
            or self.selected_gates.shape != self.selected_token_indices.shape
        ):
            raise ExpertChoiceRoutingError(
                "selected_gates must be floating point with assignment shape."
            )
        if not torch.isfinite(self.selected_gates).all().item():
            raise ExpertChoiceRoutingError(
                "selected_gates must contain only finite values."
            )
        if (
            torch.any(self.selected_gates < 0).item()
            or torch.any(self.selected_gates > 1).item()
        ):
            raise ExpertChoiceRoutingError(
                "selected_gates must be probabilities in the closed unit "
                "interval."
            )

        if (
            not isinstance(self.token_assignment_counts, Tensor)
            or self.token_assignment_counts.dtype != torch.long
            or self.token_assignment_counts.ndim < 1
            or self.token_assignment_counts.numel() <= 0
        ):
            raise ExpertChoiceRoutingError(
                "token_assignment_counts must be a non-empty torch.long "
                "tensor."
            )
        for field_name in ("anchor_fallback_mask", "eligible_token_mask"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or value.shape != self.token_assignment_counts.shape
            ):
                raise ExpertChoiceRoutingError(
                    f"{field_name} must be boolean with the original token "
                    "shape."
                )
        if not (
            self.selected_token_indices.device
            == self.selected_gates.device
            == self.token_assignment_counts.device
            == self.anchor_fallback_mask.device
            == self.eligible_token_mask.device
        ):
            raise ExpertChoiceRoutingError(
                "All expert-choice decision tensors must share one device."
            )

        token_count = self.token_assignment_counts.numel()
        minimum = int(self.selected_token_indices.min().item())
        maximum = int(self.selected_token_indices.max().item())
        if minimum < 0 or maximum >= token_count:
            raise ExpertChoiceRoutingError(
                "selected_token_indices contains an out-of-range token."
            )

        # An expert's fixed bucket cannot contain the same token twice.
        sorted_indices = torch.sort(
            self.selected_token_indices,
            dim=-1,
        ).values
        if sorted_indices.shape[1] > 1 and torch.any(
            sorted_indices[:, 1:] == sorted_indices[:, :-1]
        ).item():
            raise ExpertChoiceRoutingError(
                "Each expert must select distinct tokens."
            )

        flat_eligible = self.eligible_token_mask.reshape(-1)
        if not flat_eligible.gather(
            0,
            self.selected_token_indices.reshape(-1),
        ).all().item():
            raise ExpertChoiceRoutingError(
                "Experts may select only eligible tokens."
            )

        expected_counts = torch.zeros(
            token_count,
            dtype=torch.long,
            device=self.selected_token_indices.device,
        )
        expected_counts.scatter_add_(
            0,
            self.selected_token_indices.reshape(-1),
            torch.ones(
                self.selected_token_indices.numel(),
                dtype=torch.long,
                device=self.selected_token_indices.device,
            ),
        )
        expected_counts = expected_counts.reshape(
            self.token_assignment_counts.shape
        )
        if not torch.equal(self.token_assignment_counts, expected_counts):
            raise ExpertChoiceRoutingError(
                "token_assignment_counts does not match sparse assignments."
            )
        expected_fallback = self.eligible_token_mask & (
            self.token_assignment_counts == 0
        )
        if not torch.equal(self.anchor_fallback_mask, expected_fallback):
            raise ExpertChoiceRoutingError(
                "anchor_fallback_mask must mark exactly unassigned eligible "
                "tokens."
            )

    @property
    def expert_capacity(self) -> int:
        """Return the exact number of selected tokens per expert."""

        return int(self.selected_token_indices.shape[1])

    @property
    def token_shape(self) -> torch.Size:
        """Return the original, unflattened token shape."""

        return self.token_assignment_counts.shape

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return the original-shape mask selected by one declared expert."""

        try:
            expert_index = self.expert_ids.index(expert_id)
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        mask = torch.zeros(
            self.token_assignment_counts.numel(),
            dtype=torch.bool,
            device=self.selected_token_indices.device,
        )
        mask.scatter_(
            0,
            self.selected_token_indices[expert_index],
            True,
        )
        return mask.reshape(self.token_shape)


class ExpertChoiceRouting:
    r"""Select a fixed token bucket independently for every expert.

    Given finite router logits \(L\) with shape ``token_shape + (E,)``, the
    specialist affinity probabilities are

    \[
    S_{t,e}=\operatorname{softmax}_e(L_{t,:}).
    \]

    Every expert independently selects the ``C`` eligible tokens with largest
    \(S_{t,e}\). Stable descending sorting makes the earliest flattened token
    win an exact affinity tie. Consequently every expert has exactly ``C``
    assignments, while each token has between zero and ``E`` assignments.

    References:
        Zhou et al., "Mixture-of-Experts with Expert Choice Routing."
        https://arxiv.org/abs/2202.09368

    Differences from the paper:
        Capacity is explicit, padding is excluded, sparse indices replace the
        dense permutation tensor, and unselected valid tokens receive auditable
        immutable-anchor fallback metadata. This class is intentionally a
        training/prefill planner rather than an autoregressive dispatcher.
    """

    strategy_name: ClassVar[str] = "expert_choice"

    def __init__(self, config: ExpertChoiceConfig) -> None:
        """Bind one immutable expert-capacity configuration."""

        if not isinstance(config, ExpertChoiceConfig):
            raise TypeError("config must be an ExpertChoiceConfig.")
        self.config = config

    def __call__(
        self,
        router_logits: Tensor,
        *,
        expert_ids: Sequence[str],
        token_mask: Tensor | None = None,
    ) -> ExpertChoiceDecision:
        """Return fixed expert buckets and exact anchor-fallback metadata.

        Args:
            router_logits:
                Finite floating tensor with shape ``token_shape + (E,)``.
                The final dimension follows ``expert_ids`` order.
            expert_ids:
                Stable unique identifiers for at least two specialists.
            token_mask:
                Optional boolean tensor with ``token_shape``. False positions
                are padding: experts cannot select them and they do not request
                anchor execution.

        Raises:
            ExpertChoiceRoutingError:
                If scores, identifiers, masking, or capacity are incompatible.
        """

        normalized_ids = _normalize_expert_ids(expert_ids)
        if (
            not isinstance(router_logits, Tensor)
            or not router_logits.is_floating_point()
            or router_logits.ndim < 2
        ):
            raise ExpertChoiceRoutingError(
                "router_logits must be a floating-point tensor with token and "
                "expert dimensions."
            )
        if router_logits.shape[-1] != len(normalized_ids):
            raise ExpertChoiceRoutingError(
                "Router-logit width must match expert_ids."
            )
        if router_logits.numel() == 0:
            raise ExpertChoiceRoutingError(
                "router_logits must contain at least one token."
            )
        if not torch.isfinite(router_logits).all().item():
            raise ExpertChoiceRoutingError(
                "router_logits must contain only finite values."
            )

        token_shape = router_logits.shape[:-1]
        if token_mask is None:
            eligible_mask = torch.ones(
                token_shape,
                dtype=torch.bool,
                device=router_logits.device,
            )
        else:
            if (
                not isinstance(token_mask, Tensor)
                or token_mask.dtype != torch.bool
            ):
                raise ExpertChoiceRoutingError(
                    "token_mask must be a boolean torch.Tensor or None."
                )
            if token_mask.shape != token_shape:
                raise ExpertChoiceRoutingError(
                    "token_mask must have the router-logit token shape."
                )
            if token_mask.device != router_logits.device:
                raise ExpertChoiceRoutingError(
                    "token_mask and router_logits must share one device."
                )
            # Retain caller storage when possible while guaranteeing that the
            # immutable result cannot be changed through later mask mutation.
            eligible_mask = token_mask.detach().clone()

        flat_eligible = eligible_mask.reshape(-1)
        eligible_positions = torch.nonzero(
            flat_eligible,
            as_tuple=False,
        ).squeeze(-1)
        eligible_count = int(eligible_positions.numel())
        if eligible_count < self.config.expert_capacity:
            raise ExpertChoiceRoutingError(
                "expert_capacity cannot exceed the number of eligible tokens: "
                f"{self.config.expert_capacity} > {eligible_count}."
            )

        probabilities = torch.softmax(router_logits, dim=-1)
        flat_probabilities = probabilities.reshape(
            -1,
            len(normalized_ids),
        )
        eligible_probabilities = flat_probabilities.index_select(
            0,
            eligible_positions,
        ).transpose(0, 1)

        # Sorting along the token axis implements expert choice. Stable sort is
        # required because torch.topk does not promise deterministic tie order.
        ranked_local_positions = torch.argsort(
            eligible_probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )[:, : self.config.expert_capacity]
        candidate_positions = eligible_positions.unsqueeze(0).expand(
            len(normalized_ids),
            -1,
        )
        selected_token_indices = candidate_positions.gather(
            1,
            ranked_local_positions,
        )
        selected_gates = eligible_probabilities.gather(
            1,
            ranked_local_positions,
        )

        flat_counts = torch.zeros(
            flat_probabilities.shape[0],
            dtype=torch.long,
            device=router_logits.device,
        )
        flat_counts.scatter_add_(
            0,
            selected_token_indices.reshape(-1),
            torch.ones(
                selected_token_indices.numel(),
                dtype=torch.long,
                device=router_logits.device,
            ),
        )
        assignment_counts = flat_counts.reshape(token_shape)
        anchor_fallback = eligible_mask & (assignment_counts == 0)

        return ExpertChoiceDecision(
            selected_token_indices=selected_token_indices,
            selected_gates=selected_gates,
            token_assignment_counts=assignment_counts,
            anchor_fallback_mask=anchor_fallback,
            eligible_token_mask=eligible_mask,
            expert_ids=normalized_ids,
        )


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Return at least two unique, non-empty expert identifiers."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(
        expert_ids,
        Sequence,
    ):
        raise ExpertChoiceRoutingError(
            "expert_ids must be a sequence of strings."
        )
    normalized = tuple(expert_ids)
    if len(normalized) < 2:
        raise ExpertChoiceRoutingError(
            "Expert-choice routing requires at least two experts."
        )
    if any(
        not isinstance(expert_id, str)
        or not expert_id
        or expert_id != expert_id.strip()
        for expert_id in normalized
    ):
        raise ExpertChoiceRoutingError(
            "expert_ids must be non-empty strings without surrounding "
            "whitespace."
        )
    if len(set(normalized)) != len(normalized):
        raise ExpertChoiceRoutingError("expert_ids must be unique.")
    return normalized
