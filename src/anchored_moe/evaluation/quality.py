"""Top-1 and top-2 route-quality comparison against counterfactual losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anchored_moe.calibration.oracle import (
    CounterfactualOracleResult,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    evaluate_counterfactual_oracle,
)
from anchored_moe.policies.selection import RouteDecision, Top2RouteDecision


@dataclass(frozen=True)
class Top1Top2OracleComparison:
    """Counterfactual quality report for selected top-1 and top-2 routes.

    Attributes:
        top1:
            CAL-007 oracle and regret result for the top-1 decision.
        top2:
            CAL-007 oracle and regret result for the anchor-veto top-2 decision.
        anchor_losses:
            Immutable-anchor loss for every token or example.
        best_single_losses:
            Lowest specialist-only single-route loss.
        best_pair_losses:
            Lowest loss among explicitly sampled expert pairs.
        second_expert_gain:
            ``top1.selected_losses - top2.selected_losses``. Positive values
            mean the selected top-2 policy improved upon selected top-1.
        oracle_pair_gain:
            ``best_single_losses - best_pair_losses``. Positive values mean
            at least one sampled pair beats every individual specialist.
        top2_pair_mask:
            Tokens where top-2 actually selected two specialists rather than
            being vetoed to the anchor.
        destructive_mixture_mask:
            Executed specialist pairs whose loss is strictly worse than the
            selected top-1 route.
    """

    top1: CounterfactualOracleResult
    top2: CounterfactualOracleResult
    anchor_losses: Tensor
    best_single_losses: Tensor
    best_pair_losses: Tensor
    second_expert_gain: Tensor
    oracle_pair_gain: Tensor
    top2_pair_mask: Tensor
    destructive_mixture_mask: Tensor

    @property
    def mean_top1_regret(self) -> float:
        """Return mean selected top-1 regret."""

        return self.top1.mean_regret

    @property
    def mean_top2_regret(self) -> float:
        """Return mean selected top-2 regret."""

        return self.top2.mean_regret

    @property
    def mean_second_expert_gain(self) -> float:
        """Return mean selected top-2 gain over tokens executing a pair."""

        return _masked_mean(self.second_expert_gain, self.top2_pair_mask)

    @property
    def mean_oracle_pair_gain(self) -> float:
        """Return mean best-sampled-pair gain over the best single expert."""

        return _mean_or_zero(self.oracle_pair_gain)

    @property
    def destructive_mixture_fraction(self) -> float:
        """Return the fraction of executed pairs worse than selected top-1."""

        return _masked_mean(
            self.destructive_mixture_mask.to(dtype=torch.float64),
            self.top2_pair_mask,
        )


def compare_top1_top2_to_oracle(
    candidate_losses: CounterfactualRouteLosses,
    top1_decision: RouteDecision,
    top2_decision: Top2RouteDecision,
) -> Top1Top2OracleComparison:
    """Compare selected anchored top-1 and anchor-veto top-2 route quality.

    The function maps existing policy decisions into explicit CAL-007
    candidate columns. Anchor-vetoed top-2 decisions map to the anchor.
    Specialist pairs are unordered for lookup and must have been included in
    the sparse counterfactual table. The function reads but never mutates its
    inputs and records no token content or activations.

    Args:
        candidate_losses:
            CAL-007 table containing anchor, all singles, and at least one pair.
        top1_decision:
            Anchored top-1 routes with the same token shape and expert order.
        top2_decision:
            Anchor-veto top-2 routes with the same token shape and expert order.

    Returns:
        Per-token losses and aggregate metric properties for EVAL-004.

    Raises:
        TypeError: If inputs do not use the required public contracts.
        ValueError: If shapes, expert order, or top-2 pair coverage differ.
    """

    if not isinstance(candidate_losses, CounterfactualRouteLosses):
        raise TypeError("candidate_losses must be CounterfactualRouteLosses.")
    if not isinstance(top1_decision, RouteDecision):
        raise TypeError("top1_decision must be a RouteDecision.")
    if not isinstance(top2_decision, Top2RouteDecision):
        raise TypeError("top2_decision must be a Top2RouteDecision.")
    if candidate_losses.pair_count == 0:
        raise ValueError("EVAL-004 requires at least one sampled pair route.")
    if top1_decision.expert_ids != candidate_losses.expert_ids:
        raise ValueError(
            "top1_decision and candidate_losses must share expert_ids."
        )
    if top2_decision.expert_ids != candidate_losses.expert_ids:
        raise ValueError(
            "top2_decision and candidate_losses must share expert_ids."
        )
    if top1_decision.selected_indices.shape != candidate_losses.token_shape:
        raise ValueError(
            "top1_decision must match the counterfactual token shape."
        )
    if top2_decision.selected_indices.shape[:-1] != candidate_losses.token_shape:
        raise ValueError(
            "top2_decision must match the counterfactual token shape."
        )

    # The canonical table prefix uses the same numeric candidate semantics as
    # RouteDecision: zero is the anchor and i + 1 is expert_ids[i].
    top1_candidate_indices = (
        top1_decision.selected_indices.detach().to(device="cpu").clone()
    )
    top2_candidate_indices = _top2_candidate_indices(
        candidate_losses,
        top2_decision,
    )
    top1_result = evaluate_counterfactual_oracle(
        candidate_losses,
        top1_candidate_indices,
    )
    top2_result = evaluate_counterfactual_oracle(
        candidate_losses,
        top2_candidate_indices,
    )

    losses = candidate_losses.losses
    expert_count = len(candidate_losses.expert_ids)
    anchor_losses = losses[..., 0]
    best_single_losses = torch.min(
        losses[..., 1 : expert_count + 1],
        dim=-1,
    ).values
    best_pair_losses = torch.min(
        losses[..., expert_count + 1 :],
        dim=-1,
    ).values
    top2_pair_mask = (
        top2_decision.expert_pair_mask.detach().to(device="cpu").clone()
    )
    second_expert_gain = (
        top1_result.selected_losses - top2_result.selected_losses
    )
    oracle_pair_gain = best_single_losses - best_pair_losses
    destructive_mixture_mask = top2_pair_mask & (second_expert_gain < 0)

    return Top1Top2OracleComparison(
        top1=top1_result,
        top2=top2_result,
        anchor_losses=anchor_losses,
        best_single_losses=best_single_losses,
        best_pair_losses=best_pair_losses,
        second_expert_gain=second_expert_gain,
        oracle_pair_gain=oracle_pair_gain,
        top2_pair_mask=top2_pair_mask,
        destructive_mixture_mask=destructive_mixture_mask,
    )


def _top2_candidate_indices(
    candidate_losses: CounterfactualRouteLosses,
    decision: Top2RouteDecision,
) -> Tensor:
    """Map anchor-veto pair decisions to sparse counterfactual columns."""

    selected = decision.selected_indices.detach().to(device="cpu")
    pair_mask = decision.expert_pair_mask.detach().to(device="cpu")
    candidate_indices = torch.zeros(
        candidate_losses.token_shape,
        dtype=torch.long,
    )
    if not pair_mask.any().item():
        return candidate_indices

    flat_selected = selected.reshape(-1, 2)
    flat_pair_mask = pair_mask.reshape(-1)
    flat_candidates = candidate_indices.reshape(-1)
    expert_ids = candidate_losses.expert_ids
    for token_index in torch.nonzero(flat_pair_mask, as_tuple=False).flatten().tolist():
        first_index, second_index = flat_selected[token_index].tolist()
        if first_index == second_index:
            raise ValueError(
                "An active top-2 route must contain two distinct specialists."
            )
        ordered_indices = sorted((first_index, second_index))
        route = CounterfactualRoute.pair(
            expert_ids[ordered_indices[0] - 1],
            expert_ids[ordered_indices[1] - 1],
        )
        try:
            flat_candidates[token_index] = candidate_losses.index_for_route(route)
        except KeyError as error:
            raise ValueError(
                "Selected top-2 pair was not evaluated in candidate_losses: "
                f"{route.expert_ids!r}."
            ) from error
    return candidate_indices


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    """Return a JSON-compatible mean over a boolean mask."""

    if not mask.any().item():
        return 0.0
    return float(values[mask].to(dtype=torch.float64).mean().item())


def _mean_or_zero(values: Tensor) -> float:
    """Return a JSON-compatible mean for possibly empty tensors."""

    if values.numel() == 0:
        return 0.0
    return float(values.to(dtype=torch.float64).mean().item())
