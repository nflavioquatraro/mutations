"""Deterministic counterfactual route oracle and router-regret evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CounterfactualRoute:
    """Identity of one evaluated anchor, single-expert, or expert-pair route.

    The anchor is represented by an empty ``expert_ids`` tuple. A single route
    contains one expert identifier, and a pair contains two identifiers in
    registry order. Mixture coefficients are deliberately absent: the aligned
    loss tensor may evaluate different score-derived coefficients for the same
    pair on each token.

    Args:
        expert_ids:
            Zero, one, or two unique non-empty expert identifiers.
    """

    expert_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate route arity and expert identifiers."""

        if not isinstance(self.expert_ids, tuple):
            raise TypeError("expert_ids must be a tuple.")
        if len(self.expert_ids) > 2:
            raise ValueError("A counterfactual route may contain at most two experts.")
        if any(
            not isinstance(expert_id, str) or not expert_id.strip()
            for expert_id in self.expert_ids
        ):
            raise ValueError("Route expert_ids must be non-empty strings.")
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("A counterfactual route cannot repeat an expert.")

    @classmethod
    def anchor(cls) -> CounterfactualRoute:
        """Return the immutable-anchor route identity."""

        return cls()

    @classmethod
    def single(cls, expert_id: str) -> CounterfactualRoute:
        """Return one forced single-expert route identity."""

        return cls((expert_id,))

    @classmethod
    def pair(cls, first_expert_id: str, second_expert_id: str) -> CounterfactualRoute:
        """Return one expert-pair route identity in declared registry order."""

        return cls((first_expert_id, second_expert_id))

    @property
    def arity(self) -> int:
        """Return the number of active specialists in this route."""

        return len(self.expert_ids)

    @property
    def is_anchor(self) -> bool:
        """Return whether this route executes the immutable anchor."""

        return self.arity == 0


class CounterfactualRouteLosses:
    """Immutable aligned losses for anchor, single, and sampled pair routes.

    ``losses`` has shape ``token_shape + (C,)`` where ``C`` follows one
    canonical candidate order:

    1. the anchor;
    2. every single expert in ``expert_ids`` order;
    3. zero or more sampled pairs in lexicographic registry-index order.

    Requiring every single route makes top-1 regret well-defined, while the
    pair suffix may remain sparse to avoid implicit :math:`O(K^2)` evaluation.
    Construction detaches, copies, and moves losses to contiguous CPU storage;
    the accessor returns a defensive copy.

    Args:
        losses:
            Finite floating-point counterfactual losses with a final candidate
            dimension.
        expert_ids:
            Stable registry order shared with router decisions.
        routes:
            Explicit candidates aligned to the final loss dimension.
        loss_name:
            Non-empty identifier for the common loss definition.

    This CAL-007 data contract consumes already measured losses. It does not
    execute model paths, sample expert pairs, or serialize calibration banks.
    It is framework infrastructure and is not derived from a paper.
    """

    def __init__(
        self,
        losses: Tensor,
        *,
        expert_ids: Sequence[str],
        routes: Sequence[CounterfactualRoute],
        loss_name: str,
    ) -> None:
        normalized_expert_ids = _normalize_expert_ids(expert_ids)
        normalized_routes = _normalize_routes(
            routes,
            expert_ids=normalized_expert_ids,
        )
        normalized_losses = _normalize_losses(
            losses,
            candidate_count=len(normalized_routes),
        )
        if not isinstance(loss_name, str) or not loss_name.strip():
            raise ValueError("loss_name must be a non-empty string.")

        self._losses = normalized_losses
        self._expert_ids = normalized_expert_ids
        self._routes = normalized_routes
        self._loss_name = loss_name
        self._route_indices = {
            route: candidate_index
            for candidate_index, route in enumerate(normalized_routes)
        }

    @property
    def losses(self) -> Tensor:
        """Return a defensive copy of the aligned counterfactual losses."""

        return self._losses.clone()

    @property
    def expert_ids(self) -> tuple[str, ...]:
        """Return expert identifiers in stable registry order."""

        return self._expert_ids

    @property
    def routes(self) -> tuple[CounterfactualRoute, ...]:
        """Return candidate routes in deterministic oracle tie-break order."""

        return self._routes

    @property
    def loss_name(self) -> str:
        """Return the common loss definition used by every candidate."""

        return self._loss_name

    @property
    def token_shape(self) -> torch.Size:
        """Return the leading token or example shape."""

        return self._losses.shape[:-1]

    @property
    def token_count(self) -> int:
        """Return the flattened number of evaluated token or example rows."""

        return self._losses[..., 0].numel()

    @property
    def candidate_count(self) -> int:
        """Return the number of explicitly evaluated routes."""

        return len(self._routes)

    @property
    def pair_count(self) -> int:
        """Return the number of explicitly evaluated sparse pair routes."""

        return self.candidate_count - len(self._expert_ids) - 1

    def index_for_route(self, route: CounterfactualRoute) -> int:
        """Return the candidate column for one exact route identity.

        Raises:
            TypeError: If ``route`` is not a :class:`CounterfactualRoute`.
            KeyError: If the route was not evaluated in this sparse table.
        """

        if not isinstance(route, CounterfactualRoute):
            raise TypeError("route must be a CounterfactualRoute.")
        try:
            return self._route_indices[route]
        except KeyError as error:
            raise KeyError(
                f"Counterfactual losses do not contain route {route.expert_ids!r}."
            ) from error


@dataclass(frozen=True)
class CounterfactualOracleResult:
    """Per-token oracle choice, selected-route loss, and non-negative regret.

    Attributes:
        oracle_candidate_indices:
            Candidate columns with minimum loss. Ties use the table's stable
            anchor-first, single-before-pair order.
        oracle_losses:
            Minimum evaluated loss for each token or example.
        selected_candidate_indices:
            Candidate columns chosen by the evaluated router policy.
        selected_losses:
            Counterfactual losses of those selected routes.
        regret:
            ``selected_losses - oracle_losses`` for each token or example.
    """

    oracle_candidate_indices: Tensor
    oracle_losses: Tensor
    selected_candidate_indices: Tensor
    selected_losses: Tensor
    regret: Tensor

    @property
    def exact_oracle_mask(self) -> Tensor:
        """Return where the selected route attains the oracle loss."""

        return self.regret == 0

    @property
    def mean_regret(self) -> float:
        """Return mean selected-route regret, or zero for an empty token shape."""

        return _mean_or_zero(self.regret)

    @property
    def oracle_agreement_fraction(self) -> float:
        """Return the fraction of selected routes matching the tie-broken oracle."""

        matches = self.selected_candidate_indices == self.oracle_candidate_indices
        return _mean_or_zero(matches.to(dtype=torch.float64))


def evaluate_counterfactual_oracle(
    candidate_losses: CounterfactualRouteLosses,
    selected_candidate_indices: Tensor,
) -> CounterfactualOracleResult:
    """Evaluate a router selection against the best explicit candidate route.

    Args:
        candidate_losses:
            Immutable CAL-007 loss table whose final dimension enumerates
            candidate routes.
        selected_candidate_indices:
            ``torch.long`` tensor with ``candidate_losses.token_shape``. Values
            index exact columns of the loss table.

    Returns:
        Oracle assignments, selected losses, and per-token regret on CPU.

    Raises:
        TypeError: If arguments have incompatible types or index dtype.
        ValueError: If shape or candidate-index bounds are invalid.
    """

    if not isinstance(candidate_losses, CounterfactualRouteLosses):
        raise TypeError("candidate_losses must be CounterfactualRouteLosses.")
    normalized_indices = _normalize_selected_indices(
        selected_candidate_indices,
        token_shape=candidate_losses.token_shape,
        candidate_count=candidate_losses.candidate_count,
    )
    losses = candidate_losses.losses

    # torch.argmin returns the first minimum, so the canonical candidate order
    # defines deterministic anchor-first and single-before-pair tie-breaking.
    oracle_indices = torch.argmin(losses, dim=-1)
    oracle_losses = torch.gather(
        losses,
        dim=-1,
        index=oracle_indices.unsqueeze(-1),
    ).squeeze(-1)
    selected_losses = torch.gather(
        losses,
        dim=-1,
        index=normalized_indices.unsqueeze(-1),
    ).squeeze(-1)
    regret = selected_losses - oracle_losses

    return CounterfactualOracleResult(
        oracle_candidate_indices=oracle_indices,
        oracle_losses=oracle_losses,
        selected_candidate_indices=normalized_indices,
        selected_losses=selected_losses,
        regret=regret,
    )


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze one stable registry ordering."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(expert_ids, Sequence):
        raise TypeError("expert_ids must be a sequence of strings.")
    normalized = tuple(expert_ids)
    if any(not isinstance(expert_id, str) or not expert_id.strip() for expert_id in normalized):
        raise ValueError("expert_ids must contain non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must be unique.")
    return normalized


def _normalize_routes(
    routes: Sequence[CounterfactualRoute],
    *,
    expert_ids: tuple[str, ...],
) -> tuple[CounterfactualRoute, ...]:
    """Require canonical complete-single and sparse-pair candidate order."""

    if isinstance(routes, (str, bytes)) or not isinstance(routes, Sequence):
        raise TypeError("routes must be a sequence of CounterfactualRoute values.")
    normalized = tuple(routes)
    if not normalized:
        raise ValueError("routes must contain at least the anchor.")
    if not all(isinstance(route, CounterfactualRoute) for route in normalized):
        raise TypeError("routes must contain only CounterfactualRoute values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("routes must not contain duplicates.")

    required_prefix = (CounterfactualRoute.anchor(),) + tuple(
        CounterfactualRoute.single(expert_id) for expert_id in expert_ids
    )
    if normalized[: len(required_prefix)] != required_prefix:
        raise ValueError(
            "routes must begin with the anchor followed by every single expert "
            "in expert_ids order."
        )

    registry_indices = {
        expert_id: expert_index for expert_index, expert_id in enumerate(expert_ids)
    }
    pair_keys: list[tuple[int, int]] = []
    for route in normalized[len(required_prefix) :]:
        if route.arity != 2:
            raise ValueError("Only expert-pair routes may follow the single routes.")
        first, second = route.expert_ids
        if first not in registry_indices or second not in registry_indices:
            raise ValueError("Pair routes must reference declared expert_ids.")
        first_index = registry_indices[first]
        second_index = registry_indices[second]
        if first_index >= second_index:
            raise ValueError(
                "Pair expert_ids must follow ascending registry order."
            )
        pair_keys.append((first_index, second_index))
    if pair_keys != sorted(pair_keys):
        raise ValueError("Pair routes must use lexicographic registry order.")
    return normalized


def _normalize_losses(losses: Tensor, *, candidate_count: int) -> Tensor:
    """Validate finite candidate losses and create immutable CPU storage."""

    if not isinstance(losses, Tensor):
        raise TypeError("losses must be a torch.Tensor.")
    if not losses.is_floating_point():
        raise TypeError("losses must use a floating-point dtype.")
    if losses.ndim < 1 or losses.shape[-1] != candidate_count:
        raise ValueError(
            "losses must end with one column for every declared route."
        )
    if not torch.isfinite(losses).all().item():
        raise ValueError("losses must contain only finite values.")
    return losses.detach().to(device="cpu").contiguous().clone()


def _normalize_selected_indices(
    selected_candidate_indices: Tensor,
    *,
    token_shape: torch.Size,
    candidate_count: int,
) -> Tensor:
    """Validate selected route columns and copy them to CPU."""

    if not isinstance(selected_candidate_indices, Tensor):
        raise TypeError("selected_candidate_indices must be a torch.Tensor.")
    if selected_candidate_indices.dtype != torch.long:
        raise TypeError("selected_candidate_indices must use torch.long.")
    if selected_candidate_indices.shape != token_shape:
        raise ValueError(
            "selected_candidate_indices must match the counterfactual token shape."
        )
    normalized = (
        selected_candidate_indices.detach().to(device="cpu").contiguous().clone()
    )
    if normalized.numel():
        minimum = int(normalized.min().item())
        maximum = int(normalized.max().item())
        if minimum < 0 or maximum >= candidate_count:
            raise ValueError(
                "selected_candidate_indices contains an unevaluated candidate."
            )
    return normalized


def _mean_or_zero(values: Tensor) -> float:
    """Return a JSON-compatible mean for possibly empty tensors."""

    if values.numel() == 0:
        return 0.0
    return float(values.to(dtype=torch.float64).mean().item())
