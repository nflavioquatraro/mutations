"""Exact product-key retrieval over EXP-004 micro-expert identifiers.

EXP-005 implements the routing-index portion of PEER without executing or
combining the retrieved micro-experts. Two learned subkey tables implicitly
define a square Cartesian product of expert keys. A precomputed split query is
scored against each subkey table, then only the Cartesian product of the best
subkeys is ranked to recover the exact global top-k experts.

References:
    He, "Mixture of A Million Experts."
    https://arxiv.org/abs/2407.04153

Differences from the paper:
    PEER learns a query projection, optionally normalizes queries, performs
    multi-head retrieval, executes singleton experts, and combines their
    outputs with transformed router scores. This retrieval-only primitive
    accepts one already projected query, uses one retrieval head, returns raw
    inner-product scores, and deliberately leaves subset execution,
    combination, training orchestration, and package persistence to separate
    features. EXP-004 micro-experts may contain multiple contiguous channels
    rather than PEER's singleton neurons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isqrt
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from anchored_moe.experts.micro import FineGrainedMicroExpertSet


class ProductKeyRetrievalError(ValueError):
    """Raised when product keys, queries, or expert mappings are incompatible."""


@dataclass(frozen=True)
class ProductKeyRetrievalConfig:
    """Configuration for one exact product-key retrieval head.

    Args:
        top_k:
            Positive number of micro-experts returned for each query. It must
            not exceed the square Cartesian expert count resolved from the two
            subkey tables.
    """

    top_k: int

    def __post_init__(self) -> None:
        """Validate the bounded positive retrieval width."""

        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ProductKeyRetrievalError(
                "top_k must be a positive integer."
            )

    def to_dict(self) -> dict[str, int]:
        """Return the complete JSON-compatible configuration."""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ProductKeyRetrievalConfig:
        """Strictly reconstruct a public retrieval configuration."""

        if not isinstance(values, Mapping):
            raise ProductKeyRetrievalError(
                "Product-key retrieval config must be a mapping."
            )
        if set(values) != {"top_k"}:
            raise ProductKeyRetrievalError(
                "Product-key retrieval config fields are incompatible."
            )
        return cls(top_k=values["top_k"])


@dataclass(frozen=True)
class ProductKeyRetrievalResult:
    """Exact selected micro-expert indices, key pairs, and raw scores.

    ``selected_indices`` and ``selected_scores`` have shape
    ``query_shape + (k,)``. ``selected_pairs`` has shape
    ``query_shape + (k, 2)`` and records the two subkey indices. Expert index
    ``i * C + j`` maps pair ``(i, j)`` into the stable row-major
    ``expert_ids`` order, where ``C`` is ``subkey_count``.

    Selected scores remain graph-connected to the query and subkeys. The
    result is therefore immutable as metadata but does not detach or copy
    tensors behind the caller's back.
    """

    selected_indices: Tensor
    selected_scores: Tensor
    selected_pairs: Tensor
    expert_ids: tuple[str, ...]
    subkey_count: int

    def __post_init__(self) -> None:
        """Validate tensor alignment, pair mapping, and unique selections."""

        expert_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", expert_ids)
        if (
            isinstance(self.subkey_count, bool)
            or not isinstance(self.subkey_count, int)
            or self.subkey_count <= 0
        ):
            raise ProductKeyRetrievalError(
                "subkey_count must be a positive integer."
            )
        if len(expert_ids) != self.subkey_count**2:
            raise ProductKeyRetrievalError(
                "expert_ids must describe the complete square product-key "
                "index."
            )
        if not isinstance(self.selected_indices, Tensor):
            raise TypeError("selected_indices must be a torch.Tensor.")
        if self.selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must use torch.long.")
        if not isinstance(self.selected_scores, Tensor):
            raise TypeError("selected_scores must be a torch.Tensor.")
        if not self.selected_scores.is_floating_point():
            raise TypeError("selected_scores must use a floating-point dtype.")
        if self.selected_scores.shape != self.selected_indices.shape:
            raise ProductKeyRetrievalError(
                "selected_scores and selected_indices must share one shape."
            )
        if self.selected_indices.ndim < 1:
            raise ProductKeyRetrievalError(
                "Retrieval tensors must have a final selected-expert dimension."
            )
        if not isinstance(self.selected_pairs, Tensor):
            raise TypeError("selected_pairs must be a torch.Tensor.")
        if self.selected_pairs.dtype != torch.long:
            raise TypeError("selected_pairs must use torch.long.")
        expected_pair_shape = self.selected_indices.shape + (2,)
        if self.selected_pairs.shape != expected_pair_shape:
            raise ProductKeyRetrievalError(
                "selected_pairs must append a size-two product-key dimension."
            )
        if not (
            self.selected_indices.device
            == self.selected_scores.device
            == self.selected_pairs.device
        ):
            raise ProductKeyRetrievalError(
                "All retrieval result tensors must share one device."
            )
        if not torch.isfinite(self.selected_scores).all().item():
            raise ProductKeyRetrievalError(
                "selected_scores must contain only finite values."
            )
        if self.selected_indices.numel():
            if (
                torch.any(self.selected_indices < 0).item()
                or torch.any(self.selected_indices >= len(expert_ids)).item()
            ):
                raise ProductKeyRetrievalError(
                    "selected_indices contain an expert outside expert_ids."
                )
            if (
                torch.any(self.selected_pairs < 0).item()
                or torch.any(
                    self.selected_pairs >= self.subkey_count
                ).item()
            ):
                raise ProductKeyRetrievalError(
                    "selected_pairs contain an out-of-range subkey index."
                )
            reconstructed = (
                self.selected_pairs[..., 0] * self.subkey_count
                + self.selected_pairs[..., 1]
            )
            if not torch.equal(reconstructed, self.selected_indices):
                raise ProductKeyRetrievalError(
                    "selected_pairs do not map to selected_indices."
                )
            ranked = torch.sort(self.selected_indices, dim=-1).values
            if ranked.shape[-1] > 1 and torch.any(
                ranked[..., 1:] == ranked[..., :-1]
            ).item():
                raise ProductKeyRetrievalError(
                    "Each query must select unique micro-experts."
                )

    @property
    def top_k(self) -> int:
        """Return the fixed number of experts selected per query."""

        return self.selected_indices.shape[-1]

    @property
    def candidate_mask(self) -> Tensor:
        """Return a boolean mask in complete stable expert order."""

        mask = torch.zeros(
            self.selected_indices.shape[:-1] + (len(self.expert_ids),),
            dtype=torch.bool,
            device=self.selected_indices.device,
        )
        return mask.scatter(-1, self.selected_indices, True)

    def mask_for_expert(self, expert_id: str) -> Tensor:
        """Return the query mask selecting one stable expert identifier."""

        try:
            expert_index = self.expert_ids.index(expert_id)
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {expert_id!r}.") from error
        return torch.any(self.selected_indices == expert_index, dim=-1)


class ProductKeyMicroExpertRetriever(nn.Module):
    r"""Retrieve exact top-k micro-expert IDs from a square product-key index.

    Let both subkey tables contain ``C`` rows in
    :math:`\mathbb{R}^{d_s}`. Their Cartesian product defines
    :math:`N=C^2` expert keys. For one query
    :math:`q=[q_1;q_2]\in\mathbb{R}^{2d_s}`, expert pair ``(i, j)`` receives

    .. math::

        s_{i,j}(q)=q_1^\top c_i + q_2^\top c'_j.

    The implementation first keeps the best ``min(k, C)`` rows from each
    subkey table, scores at most ``min(k, C)^2`` pairs, and ranks those pairs
    again. This is mathematically exact because every global top-k Cartesian
    pair must use a subkey from each corresponding top-k list.

    Equal scores use the stable row-major ``expert_ids`` order. The retriever
    owns only the two subkey tensors; it does not own a query projection or
    any EXP-004 weights and never executes a micro-expert.

    References:
        He, "Mixture of A Million Experts."
        https://arxiv.org/abs/2407.04153

    Differences from the paper:
        This class implements one retrieval head over caller-projected
        queries. It omits query BatchNorm, multi-head retrieval, score
        activation, expert execution, and output aggregation. EXP-004 experts
        are contiguous gated-MLP segments rather than singleton neurons.
    """

    architecture = "product_key_micro_expert_retrieval"

    def __init__(
        self,
        first_subkeys: Tensor,
        second_subkeys: Tensor,
        *,
        expert_ids: Sequence[str],
        config: ProductKeyRetrievalConfig,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(config, ProductKeyRetrievalConfig):
            raise TypeError(
                "config must be a ProductKeyRetrievalConfig instance."
            )
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        first, second = _validate_and_copy_subkeys(
            first_subkeys,
            second_subkeys,
        )
        normalized_ids = _normalize_expert_ids(expert_ids)
        expert_count = first.shape[0] ** 2
        if len(normalized_ids) != expert_count:
            raise ProductKeyRetrievalError(
                "expert_ids must contain one identifier for every row-major "
                "product-key pair."
            )
        if config.top_k > expert_count:
            raise ProductKeyRetrievalError(
                "top_k cannot exceed the product-key expert count."
            )

        self.config = config
        self.expert_ids = normalized_ids
        self.first_subkeys = nn.Parameter(first, requires_grad=trainable)
        self.second_subkeys = nn.Parameter(second, requires_grad=trainable)

    @classmethod
    def from_micro_expert_set(
        cls,
        expert_set: FineGrainedMicroExpertSet,
        first_subkeys: Tensor,
        second_subkeys: Tensor,
        *,
        config: ProductKeyRetrievalConfig,
        expert_id_prefix: str = "micro",
        trainable: bool = True,
    ) -> ProductKeyMicroExpertRetriever:
        """Bind a square product-key index to one EXP-004 expert count.

        The method reads only ``expert_set.expert_count``. It does not retain,
        freeze, mutate, execute, or serialize the supplied expert set.
        """

        if not isinstance(expert_set, FineGrainedMicroExpertSet):
            raise TypeError(
                "expert_set must be a FineGrainedMicroExpertSet instance."
            )
        prefix = _normalize_expert_id_prefix(expert_id_prefix)
        subkey_count = isqrt(expert_set.expert_count)
        if subkey_count**2 != expert_set.expert_count:
            raise ProductKeyRetrievalError(
                "EXP-005 version 1 requires a perfect-square micro-expert "
                "count."
            )
        if (
            not isinstance(first_subkeys, Tensor)
            or not isinstance(second_subkeys, Tensor)
            or first_subkeys.ndim != 2
            or second_subkeys.ndim != 2
            or first_subkeys.shape[0] != subkey_count
            or second_subkeys.shape[0] != subkey_count
        ):
            raise ProductKeyRetrievalError(
                "Each subkey table must have sqrt(expert_count) rows."
            )
        expert_ids = tuple(
            f"{prefix}:{index}" for index in range(expert_set.expert_count)
        )
        return cls(
            first_subkeys,
            second_subkeys,
            expert_ids=expert_ids,
            config=config,
            trainable=trainable,
        )

    @property
    def subkey_count(self) -> int:
        """Return the cardinality of each equal-size subkey table."""

        return self.first_subkeys.shape[0]

    @property
    def subkey_size(self) -> int:
        """Return the dimensionality of each subkey and half-query."""

        return self.first_subkeys.shape[1]

    @property
    def expert_count(self) -> int:
        """Return the implicit Cartesian product-key count."""

        return len(self.expert_ids)

    @property
    def query_size(self) -> int:
        """Return the required concatenated query width."""

        return self.subkey_size * 2

    def set_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients for both learned subkey tables."""

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        self.requires_grad_(trainable)

    def forward(self, queries: Tensor) -> ProductKeyRetrievalResult:
        """Return exact stable top-k product-key matches for each query.

        Args:
            queries:
                Finite floating tensor with shape ``query_shape + (2D,)``,
                where ``D`` is :attr:`subkey_size`. Its dtype and device must
                exactly match both subkey tables.

        Returns:
            A graph-connected :class:`ProductKeyRetrievalResult`.

        Raises:
            TypeError:
                If ``queries`` is not a floating-point tensor.
            ProductKeyRetrievalError:
                If shape, dtype, device, or finite-value contracts fail.
        """

        if not isinstance(queries, Tensor):
            raise TypeError("queries must be a torch.Tensor.")
        if not queries.is_floating_point():
            raise TypeError("queries must use a floating-point dtype.")
        if queries.ndim < 1 or queries.shape[-1] != self.query_size:
            raise ProductKeyRetrievalError(
                f"queries must have final dimension {self.query_size}."
            )
        if (
            queries.dtype != self.first_subkeys.dtype
            or queries.device != self.first_subkeys.device
        ):
            raise ProductKeyRetrievalError(
                "queries and subkey tables must share one dtype and device."
            )
        if not torch.isfinite(queries).all().item():
            raise ProductKeyRetrievalError(
                "queries must contain only finite values."
            )

        first_queries, second_queries = queries.split(
            self.subkey_size,
            dim=-1,
        )
        first_scores = first_queries @ self.first_subkeys.transpose(0, 1)
        second_scores = second_queries @ self.second_subkeys.transpose(0, 1)
        candidate_width = min(self.config.top_k, self.subkey_count)

        # Stable subkey ordering makes equal-score behavior independent of the
        # backend top-k kernel. Earlier rows correspond to earlier product
        # expert IDs by the documented row-major mapping.
        first_indices = torch.argsort(
            first_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :candidate_width]
        second_indices = torch.argsort(
            second_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :candidate_width]
        selected_first_scores = torch.gather(
            first_scores,
            dim=-1,
            index=first_indices,
        )
        selected_second_scores = torch.gather(
            second_scores,
            dim=-1,
            index=second_indices,
        )

        pair_scores = (
            selected_first_scores.unsqueeze(-1)
            + selected_second_scores.unsqueeze(-2)
        ).flatten(start_dim=-2)
        pair_indices = (
            first_indices.unsqueeze(-1) * self.subkey_count
            + second_indices.unsqueeze(-2)
        ).flatten(start_dim=-2)

        # Candidate tensors arrive in score-ranked subkey order, not global
        # expert order. Pre-sorting by row-major ID and then applying a stable
        # descending score sort gives exact expert-first tie-breaking.
        id_order = torch.argsort(pair_indices, dim=-1, stable=True)
        pair_indices = torch.gather(pair_indices, dim=-1, index=id_order)
        pair_scores = torch.gather(pair_scores, dim=-1, index=id_order)
        score_order = torch.argsort(
            pair_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., : self.config.top_k]
        selected_indices = torch.gather(
            pair_indices,
            dim=-1,
            index=score_order,
        )
        selected_scores = torch.gather(
            pair_scores,
            dim=-1,
            index=score_order,
        )
        selected_pairs = torch.stack(
            (
                torch.div(
                    selected_indices,
                    self.subkey_count,
                    rounding_mode="floor",
                ),
                selected_indices.remainder(self.subkey_count),
            ),
            dim=-1,
        )
        return ProductKeyRetrievalResult(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            selected_pairs=selected_pairs,
            expert_ids=self.expert_ids,
            subkey_count=self.subkey_count,
        )


def _validate_and_copy_subkeys(
    first_subkeys: Tensor,
    second_subkeys: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate PEER's equal square subkey tables and copy their values."""

    for field_name, value in (
        ("first_subkeys", first_subkeys),
        ("second_subkeys", second_subkeys),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{field_name} must be a torch.Tensor.")
        if not value.is_floating_point() or value.ndim != 2:
            raise ProductKeyRetrievalError(
                f"{field_name} must be a two-dimensional floating tensor."
            )
        if value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ProductKeyRetrievalError(
                f"{field_name} dimensions must be positive."
            )
        if not torch.isfinite(value).all().item():
            raise ProductKeyRetrievalError(
                f"{field_name} must contain only finite values."
            )
    if first_subkeys.shape != second_subkeys.shape:
        raise ProductKeyRetrievalError(
            "Product-key subkey tables must share one shape."
        )
    if (
        first_subkeys.dtype != second_subkeys.dtype
        or first_subkeys.device != second_subkeys.device
    ):
        raise ProductKeyRetrievalError(
            "Product-key subkey tables must share one dtype and device."
        )
    return (
        first_subkeys.detach().clone(),
        second_subkeys.detach().clone(),
    )


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Return a stable unique expert order."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(
        expert_ids,
        Sequence,
    ):
        raise TypeError("expert_ids must be a sequence of strings.")
    normalized = tuple(expert_ids)
    if any(
        not isinstance(expert_id, str)
        or not expert_id
        or expert_id != expert_id.strip()
        for expert_id in normalized
    ):
        raise ProductKeyRetrievalError(
            "expert_ids must be non-empty strings without surrounding "
            "whitespace."
        )
    if len(set(normalized)) != len(normalized):
        raise ProductKeyRetrievalError("expert_ids must be unique.")
    return normalized


def _normalize_expert_id_prefix(prefix: str) -> str:
    """Validate the stable prefix used for derived micro-expert IDs."""

    if (
        not isinstance(prefix, str)
        or not prefix
        or prefix != prefix.strip()
        or ":" in prefix
    ):
        raise ProductKeyRetrievalError(
            "expert_id_prefix must be non-empty, trimmed, and contain no ':'."
        )
    return prefix
