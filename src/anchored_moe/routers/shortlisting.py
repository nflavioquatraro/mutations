"""Deterministic prompt-level top-M shortlisting for logical routers.

RTR-006 treats retrieval as an explicit stage before token routing. A caller
supplies one finite retrieval score per prompt and candidate. The shortlister
selects a stable top-M set, and the scoring wrapper evaluates only those
logical-router columns for each prompt before reconstructing the complete
candidate dimension expected by existing anchored selection policies.

The retriever itself is deliberately outside this module. This keeps learned
prompt encoders, product-key retrieval, and package-specific retrieval
training independent from the routing and dispatch contracts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from anchored_moe.routers.logical import (
    LogicalAppendOnlyRouter,
    compile_logical_router,
)


class PromptShortlistingError(ValueError):
    """Raised when retrieval, shortlist, and logical-router metadata conflict."""


@dataclass(frozen=True)
class PromptCandidateShortlist:
    """Auditable prompt-level candidate indices and retrieval scores.

    Attributes:
        selected_indices:
            Long tensor with shape ``(B, M')``, where
            ``M' = min(max_candidates, K)``.
        selected_scores:
            Detached finite retrieval scores aligned with
            ``selected_indices``.
        candidate_ids:
            Complete stable candidate order of length ``K``.

    The tensors are detached defensive copies because RTR-006 consumes an
    already computed shortlist; it does not train the external retriever.
    """

    selected_indices: Tensor
    selected_scores: Tensor
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and defensively copy prompt-level routing metadata."""

        candidate_ids = _normalize_candidate_ids(self.candidate_ids)
        if not isinstance(self.selected_indices, Tensor):
            raise TypeError("selected_indices must be a torch.Tensor.")
        if (
            self.selected_indices.dtype != torch.long
            or self.selected_indices.ndim != 2
        ):
            raise PromptShortlistingError(
                "selected_indices must be a two-dimensional torch.long tensor."
            )
        if not isinstance(self.selected_scores, Tensor):
            raise TypeError("selected_scores must be a torch.Tensor.")
        if (
            not self.selected_scores.is_floating_point()
            or self.selected_scores.ndim != 2
        ):
            raise PromptShortlistingError(
                "selected_scores must be a two-dimensional floating-point tensor."
            )
        if self.selected_scores.shape != self.selected_indices.shape:
            raise PromptShortlistingError(
                "selected_scores and selected_indices must share one shape."
            )
        if self.selected_scores.device != self.selected_indices.device:
            raise PromptShortlistingError(
                "selected_scores and selected_indices must share one device."
            )
        if self.selected_indices.shape[1] > len(candidate_ids):
            raise PromptShortlistingError(
                "The shortlist width cannot exceed the candidate count."
            )
        if not torch.isfinite(self.selected_scores).all().item():
            raise PromptShortlistingError(
                "selected_scores must contain only finite values."
            )
        if self.selected_indices.numel():
            if torch.any(self.selected_indices < 0).item() or torch.any(
                self.selected_indices >= len(candidate_ids)
            ).item():
                raise PromptShortlistingError(
                    "selected_indices contain an out-of-range candidate."
                )
            sorted_indices = torch.sort(self.selected_indices, dim=-1).values
            if sorted_indices.shape[1] > 1 and torch.any(
                sorted_indices[:, 1:] == sorted_indices[:, :-1]
            ).item():
                raise PromptShortlistingError(
                    "Each prompt shortlist must contain unique candidates."
                )

        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(
            self,
            "selected_indices",
            self.selected_indices.detach().clone(),
        )
        object.__setattr__(
            self,
            "selected_scores",
            self.selected_scores.detach().clone(),
        )

    @property
    def batch_size(self) -> int:
        """Return the number of prompt rows."""

        return self.selected_indices.shape[0]

    @property
    def shortlist_size(self) -> int:
        """Return the fixed shortlist width for this batch."""

        return self.selected_indices.shape[1]

    def candidate_mask(self, *, device: torch.device | None = None) -> Tensor:
        """Return a boolean ``(B, K)`` mask in complete candidate order."""

        target_device = self.selected_indices.device if device is None else device
        mask = torch.zeros(
            (self.batch_size, len(self.candidate_ids)),
            dtype=torch.bool,
            device=target_device,
        )
        if self.shortlist_size:
            indices = self.selected_indices.to(device=target_device)
            mask.scatter_(1, indices, True)
        return mask


@dataclass(frozen=True)
class PromptShortlistedScores:
    """Full candidate-aligned token scores plus their prompt shortlist.

    ``scores`` has shape ``(B, ..., K)``. Selected candidates contain ordinary
    calibrated logical-router scores. Unselected candidates contain
    ``torch.finfo(dtype).min``, a finite value strictly below the permanent
    zero anchor for every supported floating dtype.
    """

    scores: Tensor
    shortlist: PromptCandidateShortlist
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate full score alignment and exact masking semantics."""

        if not isinstance(self.shortlist, PromptCandidateShortlist):
            raise TypeError("shortlist must be a PromptCandidateShortlist.")
        expert_ids = _normalize_candidate_ids(self.expert_ids)
        if expert_ids != self.shortlist.candidate_ids:
            raise PromptShortlistingError(
                "expert_ids must exactly match shortlist candidate_ids."
            )
        if not isinstance(self.scores, Tensor) or not self.scores.is_floating_point():
            raise TypeError("scores must be a floating-point torch.Tensor.")
        if self.scores.ndim < 2:
            raise PromptShortlistingError(
                "scores must contain prompt and candidate dimensions."
            )
        if self.scores.shape[0] != self.shortlist.batch_size:
            raise PromptShortlistingError(
                "scores prompt dimension must match the shortlist batch size."
            )
        if self.scores.shape[-1] != len(expert_ids):
            raise PromptShortlistingError(
                "scores final dimension must match expert_ids."
            )
        if not torch.isfinite(self.scores).all().item():
            raise PromptShortlistingError("scores must contain only finite values.")

        mask = self.shortlist.candidate_mask(device=self.scores.device)
        expanded_mask = _expand_prompt_mask(mask, score_shape=self.scores.shape)
        masked_value = torch.finfo(self.scores.dtype).min
        if torch.any(self.scores.masked_select(~expanded_mask) != masked_value).item():
            raise PromptShortlistingError(
                "Unselected candidate scores must equal the finite mask value."
            )
        object.__setattr__(self, "expert_ids", expert_ids)

    @property
    def candidate_mask(self) -> Tensor:
        """Return the prompt-level mask on the score tensor's device."""

        return self.shortlist.candidate_mask(device=self.scores.device)


class PromptTopMShortlister:
    """Select at most ``max_candidates`` per prompt with deterministic ties.

    Retrieval scores have shape ``(B, K)`` and may use any finite floating
    dtype. Stable descending sorting preserves the supplied candidate order
    when scores tie. Negative candidates remain eligible for the shortlist;
    the later token policy still compares their calibrated token evidence with
    the fixed-zero anchor.
    """

    def __init__(self, max_candidates: int) -> None:
        """Configure the positive maximum shortlist width."""

        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise TypeError("max_candidates must be an integer.")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero.")
        self.max_candidates = max_candidates

    def __call__(
        self,
        retrieval_scores: Tensor,
        *,
        candidate_ids: Sequence[str],
    ) -> PromptCandidateShortlist:
        """Return stable prompt-level top-M candidates from external scores."""

        normalized_ids = _normalize_candidate_ids(candidate_ids)
        if not isinstance(retrieval_scores, Tensor):
            raise TypeError("retrieval_scores must be a torch.Tensor.")
        if (
            not retrieval_scores.is_floating_point()
            or retrieval_scores.ndim != 2
        ):
            raise PromptShortlistingError(
                "retrieval_scores must be a two-dimensional floating-point tensor."
            )
        if retrieval_scores.shape[1] != len(normalized_ids):
            raise PromptShortlistingError(
                "retrieval_scores final dimension must match candidate_ids."
            )
        if not torch.isfinite(retrieval_scores).all().item():
            raise PromptShortlistingError(
                "retrieval_scores must contain only finite values."
            )

        shortlist_size = min(self.max_candidates, len(normalized_ids))
        if shortlist_size == 0:
            indices = torch.empty(
                (retrieval_scores.shape[0], 0),
                dtype=torch.long,
                device=retrieval_scores.device,
            )
        else:
            indices = torch.argsort(
                retrieval_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[:, :shortlist_size]
        scores = torch.gather(retrieval_scores, dim=1, index=indices)
        return PromptCandidateShortlist(
            selected_indices=indices,
            selected_scores=scores,
            candidate_ids=normalized_ids,
        )


class PromptShortlistedLogicalRouter(nn.Module):
    """Evaluate only prompt-shortlisted RTR-003 columns.

    Args:
        router:
            Frozen logical append-only router to copy into this deployment
            wrapper. The copy has no trainable parameters or source-storage
            aliases.

    Input features must have shape ``(B, ..., D)`` so the first dimension
    aligns with prompt rows. For each expert column, only prompt batches whose
    shortlist contains that expert are gathered and scored. Complete
    ``(B, ..., K)`` scores are reconstructed with a finite below-anchor mask,
    allowing unchanged anchored top-1, top-2, or dynamic-k policies to consume
    the result.

    This is framework infrastructure implementing the architecture
    blueprint's prompt-shortlisted routing boundary. It does not derive a
    retrieval or optimization algorithm from a paper.
    """

    architecture = "prompt_shortlisted_logical_columns"

    def __init__(self, router: LogicalAppendOnlyRouter) -> None:
        super().__init__()
        if not isinstance(router, LogicalAppendOnlyRouter):
            raise TypeError("router must be a LogicalAppendOnlyRouter.")
        self.router = compile_logical_router(
            router.to_independent_heads(trainable=False)
        )
        self.feature_size = self.router.feature_size

    def expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return the copied logical router's layer-local expert order."""

        return self.router.expert_ids_for_layer(layer_index)

    def forward(
        self,
        features: Tensor,
        *,
        shortlist: PromptCandidateShortlist,
        layer_index: int,
    ) -> PromptShortlistedScores:
        """Score shortlisted columns and reconstruct complete token scores."""

        if not isinstance(shortlist, PromptCandidateShortlist):
            raise TypeError("shortlist must be a PromptCandidateShortlist.")
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating-point torch.Tensor.")
        if features.ndim < 2:
            raise PromptShortlistingError(
                "features must contain prompt and feature dimensions."
            )
        if features.shape[-1] != self.feature_size:
            raise PromptShortlistingError(
                "features final dimension must match router feature_size."
            )
        if features.shape[0] != shortlist.batch_size:
            raise PromptShortlistingError(
                "features prompt dimension must match the shortlist batch size."
            )

        expert_ids = self.expert_ids_for_layer(layer_index)
        if expert_ids != shortlist.candidate_ids:
            raise PromptShortlistingError(
                "Shortlist candidates must exactly match the layer expert order."
            )

        prompt_mask = shortlist.candidate_mask(device=features.device)
        masked_score = torch.finfo(features.dtype).min
        score_columns: list[Tensor] = []
        for candidate_index in range(len(expert_ids)):
            prompt_indices = torch.nonzero(
                prompt_mask[:, candidate_index],
                as_tuple=False,
            ).flatten()
            complete_column = features.new_full(
                features.shape[:-1],
                masked_score,
            )
            if prompt_indices.numel():
                selected_features = features.index_select(0, prompt_indices)
                selected_scores = self.router.score_candidates(
                    selected_features,
                    candidate_indices=(candidate_index,),
                    layer_index=layer_index,
                ).squeeze(-1)
                complete_column = complete_column.index_copy(
                    0,
                    prompt_indices,
                    selected_scores,
                )
            score_columns.append(complete_column)

        if score_columns:
            scores = torch.stack(score_columns, dim=-1)
        else:
            scores = features.new_empty(features.shape[:-1] + (0,))
        return PromptShortlistedScores(
            scores=scores,
            shortlist=shortlist,
            expert_ids=expert_ids,
        )


def _expand_prompt_mask(
    mask: Tensor,
    *,
    score_shape: torch.Size,
) -> Tensor:
    """Broadcast a ``(B, K)`` prompt mask over token-leading dimensions."""

    reshape = (mask.shape[0],) + (1,) * (len(score_shape) - 2) + (mask.shape[1],)
    return mask.reshape(reshape).expand(score_shape)


def _normalize_candidate_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    """Return unique non-empty candidate IDs without changing order."""

    if isinstance(candidate_ids, (str, bytes)) or not isinstance(
        candidate_ids,
        Sequence,
    ):
        raise TypeError("candidate_ids must be a sequence of strings.")
    normalized = tuple(candidate_ids)
    if any(not isinstance(candidate_id, str) for candidate_id in normalized):
        raise TypeError("candidate_ids must contain only strings.")
    if any(not candidate_id for candidate_id in normalized):
        raise PromptShortlistingError("candidate_ids must not contain empty values.")
    if len(set(normalized)) != len(normalized):
        raise PromptShortlistingError("candidate_ids must be unique.")
    return normalized
