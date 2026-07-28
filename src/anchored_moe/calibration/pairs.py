"""Deterministic sparse expert-pair planning and pair-utility targets."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from anchored_moe.calibration.oracle import (
    CounterfactualRoute,
    CounterfactualRouteLosses,
)


@dataclass(frozen=True)
class PairSamplingPlan:
    """Reproducible bounded or exhaustive set of expert-pair candidates.

    Pair identities use ascending registry order and the final ``pairs`` tuple
    is lexicographically ordered by registry indices. In bounded mode,
    ``max_pairs`` is a positive integer and ``seed`` controls a stable SHA-256
    stream of unique flat pair positions. ``max_pairs=None`` explicitly
    requests exhaustive enumeration.

    Args:
        expert_ids:
            Stable registry ordering containing at least two unique experts.
        pairs:
            Canonical selected pair routes.
        max_pairs:
            Positive pair budget, or ``None`` for exhaustive enumeration.
        seed:
            Non-negative deterministic sampling seed.

    This CAL-003 plan contains identities only. It does not execute experts,
    choose mixture weights, or store losses.
    """

    expert_ids: tuple[str, ...]
    pairs: tuple[CounterfactualRoute, ...]
    max_pairs: int | None
    seed: int

    def __post_init__(self) -> None:
        """Validate registry, budget, coverage, and canonical pair ordering."""

        normalized_ids = _normalize_expert_ids(self.expert_ids)
        if len(normalized_ids) < 2:
            raise ValueError("Pair sampling requires at least two expert_ids.")
        normalized_max_pairs = _normalize_max_pairs(self.max_pairs)
        normalized_seed = _normalize_seed(self.seed)
        if not isinstance(self.pairs, tuple):
            raise TypeError("pairs must be a tuple.")
        if not all(isinstance(pair, CounterfactualRoute) for pair in self.pairs):
            raise TypeError("pairs must contain CounterfactualRoute values.")

        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("pairs must not contain duplicates.")
        registry_indices = {
            expert_id: index for index, expert_id in enumerate(normalized_ids)
        }
        positions = tuple(
            _position_for_pair(
                pair,
                registry_indices=registry_indices,
                expert_count=len(normalized_ids),
            )
            for pair in self.pairs
        )
        if positions != tuple(sorted(positions)):
            raise ValueError("pairs must follow lexicographic registry order.")

        available_pair_count = _available_pair_count(len(normalized_ids))
        expected_count = (
            available_pair_count
            if normalized_max_pairs is None
            else min(normalized_max_pairs, available_pair_count)
        )
        if len(self.pairs) != expected_count:
            raise ValueError(
                "pairs must contain exactly the requested bounded or exhaustive "
                "candidate count."
            )

        object.__setattr__(self, "expert_ids", normalized_ids)
        object.__setattr__(self, "max_pairs", normalized_max_pairs)
        object.__setattr__(self, "seed", normalized_seed)

    @property
    def pair_count(self) -> int:
        """Return the number of selected pair candidates."""

        return len(self.pairs)

    @property
    def available_pair_count(self) -> int:
        """Return the exhaustive pair count ``K * (K - 1) / 2``."""

        return _available_pair_count(len(self.expert_ids))

    @property
    def is_exhaustive(self) -> bool:
        """Return whether every possible pair is included."""

        return self.pair_count == self.available_pair_count

    @property
    def sampling_mode(self) -> str:
        """Return ``"bounded"`` or explicitly requested ``"exhaustive"``."""

        return "exhaustive" if self.max_pairs is None else "bounded"

    def index_for_pair(self, pair: CounterfactualRoute) -> int:
        """Return the pair-loss column for one selected route.

        Raises:
            TypeError: If ``pair`` is not a :class:`CounterfactualRoute`.
            KeyError: If the pair is outside this sampling plan.
        """

        if not isinstance(pair, CounterfactualRoute):
            raise TypeError("pair must be a CounterfactualRoute.")
        try:
            return self.pairs.index(pair)
        except ValueError as error:
            raise KeyError(f"Pair is not in the sampling plan: {pair.expert_ids!r}.") from error

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible reproducibility metadata for the plan."""

        return {
            "available_pair_count": self.available_pair_count,
            "expert_ids": list(self.expert_ids),
            "is_exhaustive": self.is_exhaustive,
            "max_pairs": self.max_pairs,
            "pair_count": self.pair_count,
            "pairs": [list(pair.expert_ids) for pair in self.pairs],
            "sampling_mode": self.sampling_mode,
            "seed": self.seed,
        }


class PairUtilityResult:
    """Immutable pair utilities and best-pair targets from measured losses.

    For token or example ``t`` and sampled pair ``p=(i,j)``, CAL-003 computes

    ``anchor_utility[t,p] = anchor_loss[t] - pair_loss[t,p]``

    and

    ``single_gain[t,p] = best_single_loss[t] - pair_loss[t,p]``.

    Positive anchor utility means the pair improves on the immutable anchor.
    Positive single gain means the pair improves on every individual expert.
    Exact best-pair loss ties select the first pair in lexicographic registry
    order.

    Construction validates a shared token shape, dtype, device, finite values,
    and exact plan dimensions. It then detaches and copies all tensors to
    contiguous CPU storage. Public tensor accessors return defensive copies.

    This class consumes precomputed losses and is not paper-derived. Model
    execution, pair-mixture construction, and serialization are deliberately
    excluded.
    """

    def __init__(
        self,
        plan: PairSamplingPlan,
        *,
        anchor_losses: Tensor,
        single_losses: Tensor,
        pair_losses: Tensor,
        loss_name: str,
    ) -> None:
        if not isinstance(plan, PairSamplingPlan):
            raise TypeError("plan must be a PairSamplingPlan.")
        normalized_anchor, normalized_singles, normalized_pairs = _normalize_losses(
            anchor_losses,
            single_losses=single_losses,
            pair_losses=pair_losses,
            expert_count=len(plan.expert_ids),
            pair_count=plan.pair_count,
        )
        if not isinstance(loss_name, str) or not loss_name.strip():
            raise ValueError("loss_name must be a non-empty string.")

        best_single_losses = torch.min(normalized_singles, dim=-1).values
        best_pair_indices = torch.argmin(normalized_pairs, dim=-1)
        best_pair_losses = torch.gather(
            normalized_pairs,
            dim=-1,
            index=best_pair_indices.unsqueeze(-1),
        ).squeeze(-1)

        self._plan = plan
        self._anchor_losses = normalized_anchor
        self._single_losses = normalized_singles
        self._pair_losses = normalized_pairs
        self._best_single_losses = best_single_losses
        self._best_pair_indices = best_pair_indices
        self._best_pair_losses = best_pair_losses
        self._pair_utilities = normalized_anchor.unsqueeze(-1) - normalized_pairs
        self._pair_improvements = (
            best_single_losses.unsqueeze(-1) - normalized_pairs
        )
        self._loss_name = loss_name

    @property
    def plan(self) -> PairSamplingPlan:
        """Return the immutable pair-sampling plan."""

        return self._plan

    @property
    def anchor_losses(self) -> Tensor:
        """Return a defensive copy of anchor losses."""

        return self._anchor_losses.clone()

    @property
    def single_losses(self) -> Tensor:
        """Return a defensive copy of registry-ordered single-expert losses."""

        return self._single_losses.clone()

    @property
    def pair_losses(self) -> Tensor:
        """Return a defensive copy of plan-ordered pair losses."""

        return self._pair_losses.clone()

    @property
    def pair_utilities(self) -> Tensor:
        """Return pair advantage over the anchor for every sampled pair."""

        return self._pair_utilities.clone()

    @property
    def pair_improvements_over_best_single(self) -> Tensor:
        """Return pair loss reduction relative to the best individual expert."""

        return self._pair_improvements.clone()

    @property
    def best_single_losses(self) -> Tensor:
        """Return the lowest individual-expert loss per token or example."""

        return self._best_single_losses.clone()

    @property
    def best_pair_indices(self) -> Tensor:
        """Return plan-local columns of the lowest-loss sampled pairs."""

        return self._best_pair_indices.clone()

    @property
    def best_pair_losses(self) -> Tensor:
        """Return the lowest sampled-pair loss per token or example."""

        return self._best_pair_losses.clone()

    @property
    def best_pair_utilities(self) -> Tensor:
        """Return best sampled-pair advantage over the anchor."""

        return self._anchor_losses.clone() - self._best_pair_losses

    @property
    def best_pair_improvements_over_best_single(self) -> Tensor:
        """Return best sampled-pair loss reduction over the best single."""

        return self._best_single_losses.clone() - self._best_pair_losses

    @property
    def loss_name(self) -> str:
        """Return the shared loss definition."""

        return self._loss_name

    @property
    def token_shape(self) -> torch.Size:
        """Return the common leading token or example shape."""

        return self._anchor_losses.shape

    def pair_beats_best_single(self, *, minimum_margin: float = 0.0) -> Tensor:
        """Return where each pair improves on the best single by a strict margin.

        Args:
            minimum_margin:
                Finite non-negative required loss reduction. An exact equality
                does not pass the strict acceptance gate.
        """

        if isinstance(minimum_margin, bool) or not isinstance(
            minimum_margin,
            (int, float),
        ):
            raise TypeError("minimum_margin must be a real number.")
        normalized_margin = float(minimum_margin)
        if not isfinite(normalized_margin) or normalized_margin < 0:
            raise ValueError("minimum_margin must be finite and non-negative.")
        return self._pair_improvements > normalized_margin

    def to_counterfactual_route_losses(self) -> CounterfactualRouteLosses:
        """Build the canonical CAL-007 table without recomputing losses."""

        losses = torch.cat(
            (
                self._anchor_losses.unsqueeze(-1),
                self._single_losses,
                self._pair_losses,
            ),
            dim=-1,
        )
        routes = (
            CounterfactualRoute.anchor(),
            *(
                CounterfactualRoute.single(expert_id)
                for expert_id in self._plan.expert_ids
            ),
            *self._plan.pairs,
        )
        return CounterfactualRouteLosses(
            losses,
            expert_ids=self._plan.expert_ids,
            routes=routes,
            loss_name=self._loss_name,
        )


def plan_expert_pairs(
    expert_ids: Sequence[str],
    *,
    max_pairs: int | None = 64,
    seed: int = 0,
) -> PairSamplingPlan:
    """Plan deterministic sparse pair evaluation without exhaustive default.

    The bounded sampler hashes successive ``(seed, counter)`` values with
    SHA-256, maps unbiased digests to unique flat pair positions, then restores
    lexicographic registry order for aligned tensors and stable best-pair
    ties. It does not enumerate the full pair universe when the budget is
    smaller than that universe. This avoids mutable global RNG state and is
    reproducible across processes.

    Args:
        expert_ids:
            Stable expert registry ordering containing at least two experts.
        max_pairs:
            Positive evaluation budget. ``None`` explicitly requests every
            possible pair. The default bounds evaluation to 64 pairs.
        seed:
            Non-negative sampling seed.

    Returns:
        An immutable :class:`PairSamplingPlan`.
    """

    normalized_ids = _normalize_expert_ids(expert_ids)
    if len(normalized_ids) < 2:
        raise ValueError("Pair sampling requires at least two expert_ids.")
    normalized_max_pairs = _normalize_max_pairs(max_pairs)
    normalized_seed = _normalize_seed(seed)
    available_pair_count = _available_pair_count(len(normalized_ids))
    selected_count = (
        available_pair_count
        if normalized_max_pairs is None
        else min(normalized_max_pairs, available_pair_count)
    )
    if selected_count == available_pair_count:
        # Exhaustive construction is allowed only when explicitly requested or
        # when the finite budget already covers the entire small registry.
        selected_positions = tuple(range(available_pair_count))
    else:
        selected_positions = _sample_pair_positions(
            available_pair_count=available_pair_count,
            selected_count=selected_count,
            seed=normalized_seed,
        )
    selected_pairs = tuple(
        _route_for_position(
            position,
            expert_ids=normalized_ids,
        )
        for position in selected_positions
    )

    return PairSamplingPlan(
        expert_ids=normalized_ids,
        pairs=selected_pairs,
        max_pairs=normalized_max_pairs,
        seed=normalized_seed,
    )


def compute_pair_utilities(
    plan: PairSamplingPlan,
    *,
    anchor_losses: Tensor,
    single_losses: Tensor,
    pair_losses: Tensor,
    loss_name: str,
) -> PairUtilityResult:
    """Compute immutable CAL-003 utilities from aligned measured losses."""

    return PairUtilityResult(
        plan,
        anchor_losses=anchor_losses,
        single_losses=single_losses,
        pair_losses=pair_losses,
        loss_name=loss_name,
    )


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze one stable expert registry ordering."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(expert_ids, Sequence):
        raise TypeError("expert_ids must be a sequence of strings.")
    normalized = tuple(expert_ids)
    if any(
        not isinstance(expert_id, str) or not expert_id.strip()
        for expert_id in normalized
    ):
        raise ValueError("expert_ids must contain non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must be unique.")
    return normalized


def _normalize_max_pairs(max_pairs: int | None) -> int | None:
    """Validate a positive bounded budget or explicit exhaustive mode."""

    if max_pairs is None:
        return None
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int):
        raise TypeError("max_pairs must be a positive integer or None.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be greater than zero.")
    return max_pairs


def _normalize_seed(seed: int) -> int:
    """Validate a non-negative deterministic seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return seed


def _available_pair_count(expert_count: int) -> int:
    """Return ``K choose 2`` without enumerating candidates."""

    return expert_count * (expert_count - 1) // 2


def _pairs_before_first(first_index: int, *, expert_count: int) -> int:
    """Return lexicographic pair columns preceding one first index."""

    return first_index * (2 * expert_count - first_index - 1) // 2


def _position_for_pair(
    pair: CounterfactualRoute,
    *,
    registry_indices: dict[str, int],
    expert_count: int,
) -> int:
    """Validate one route and return its lexicographic flat pair position."""

    if pair.arity != 2:
        raise ValueError("pairs must contain only two-expert routes.")
    first, second = pair.expert_ids
    if first not in registry_indices or second not in registry_indices:
        raise ValueError("pairs must reference declared expert_ids.")
    first_index = registry_indices[first]
    second_index = registry_indices[second]
    if first_index >= second_index:
        raise ValueError("pair expert_ids must use ascending registry order.")
    return (
        _pairs_before_first(first_index, expert_count=expert_count)
        + second_index
        - first_index
        - 1
    )


def _route_for_position(
    position: int,
    *,
    expert_ids: tuple[str, ...],
) -> CounterfactualRoute:
    """Invert one lexicographic flat pair position without enumeration."""

    expert_count = len(expert_ids)
    low = 0
    high = expert_count - 2
    while low < high:
        midpoint = (low + high + 1) // 2
        if _pairs_before_first(midpoint, expert_count=expert_count) <= position:
            low = midpoint
        else:
            high = midpoint - 1
    first_index = low
    offset = position - _pairs_before_first(
        first_index,
        expert_count=expert_count,
    )
    second_index = first_index + 1 + offset
    return CounterfactualRoute.pair(
        expert_ids[first_index],
        expert_ids[second_index],
    )


def _sample_pair_positions(
    *,
    available_pair_count: int,
    selected_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Select unique flat positions in expected ``O(selected_count)`` work."""

    selected: set[int] = set()
    counter = 0
    value_range = 1 << 256
    rejection_limit = value_range - (value_range % available_pair_count)
    while len(selected) < selected_count:
        payload = f"{seed}:{counter}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")
        counter += 1
        # Rejection removes modulo bias. With a 256-bit digest, rejection is
        # vanishingly rare for practical registry sizes.
        if value >= rejection_limit:
            continue
        selected.add(value % available_pair_count)
    return tuple(sorted(selected))


def _normalize_losses(
    anchor_losses: Tensor,
    *,
    single_losses: Tensor,
    pair_losses: Tensor,
    expert_count: int,
    pair_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate aligned losses and create owned contiguous CPU copies."""

    tensors = {
        "anchor_losses": anchor_losses,
        "single_losses": single_losses,
        "pair_losses": pair_losses,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype.")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values.")

    expected_single_shape = anchor_losses.shape + (expert_count,)
    expected_pair_shape = anchor_losses.shape + (pair_count,)
    if single_losses.shape != expected_single_shape:
        raise ValueError(
            "single_losses must have anchor token shape plus expert_count."
        )
    if pair_losses.shape != expected_pair_shape:
        raise ValueError(
            "pair_losses must have anchor token shape plus plan pair_count."
        )
    if not (
        anchor_losses.dtype == single_losses.dtype == pair_losses.dtype
    ):
        raise ValueError("All loss tensors must share one dtype.")
    if not (
        anchor_losses.device == single_losses.device == pair_losses.device
    ):
        raise ValueError("All loss tensors must share one device.")

    return tuple(
        tensor.detach().to(device="cpu").contiguous().clone()
        for tensor in (anchor_losses, single_losses, pair_losses)
    )
