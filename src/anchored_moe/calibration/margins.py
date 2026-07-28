"""Deterministic stability-margin refinement for calibrated router heads.

CAL-006 refines only newly proposed expert heads against scores from an
immutable incumbent router. The fixed-zero anchor is always included in the
incumbent margin, so refinement remains well-defined for an anchor-only
registry.

References:
    The repository architecture blueprint, "Approach E — Margin refinement."
    ../../../anchored-plugin-moe-architecture.md

Differences from the blueprint:
    The two hinge objectives are optimized with deterministic full-batch Adam
    in CPU float64 and an explicit proximal penalty around the CAL-005
    initialization. The blueprint specifies the margin losses but does not
    prescribe an optimizer or drift control.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from anchored_moe.calibration.features import RouterFeatureBank
from anchored_moe.calibration.labels import UtilityLabelBank
from anchored_moe.routers.base import RouterScoreBank
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads


class StabilityMarginRefinementError(ValueError):
    """Raised when CAL-006 cannot produce finite refined router heads."""


@dataclass(frozen=True)
class HeadStabilityMarginReport:
    """Before/after margin and activation metrics for one candidate head.

    Attributes:
        layer_index: Zero-based decoder layer owned by the candidate head.
        expert_id: Stable candidate expert identifier.
        useful_win_row_count:
            Win-partition rows whose candidate utility is strictly positive.
        ignored_win_row_count:
            Win-partition rows excluded because utility is non-positive.
        status: ``"optimized"`` or ``"unchanged"``.
        initial_stability_loss:
            Mean incumbent-protection hinge loss before refinement.
        final_stability_loss:
            Mean incumbent-protection hinge loss after deployed-dtype copying.
        initial_win_loss: Mean useful-candidate win hinge loss before refinement.
        final_win_loss:
            Mean useful-candidate win hinge loss after deployed-dtype copying.
        initial_total_loss: Weighted initial hinge objective.
        final_total_loss:
            Weighted final hinge objective plus proximal parameter drift.
        stability_violation_count_before:
            Stability rows with a strictly positive protection hinge.
        stability_violation_count_after:
            Stability rows with a strictly positive protection hinge afterward.
        win_violation_count_before:
            Useful win rows with a strictly positive win hinge.
        win_violation_count_after:
            Useful win rows with a strictly positive win hinge afterward.
        stability_activation_count_before:
            Stability rows where the candidate strictly beats the incumbent.
        stability_activation_count_after:
            Stability rows where the refined candidate beats the incumbent.
        win_activation_count_before:
            Useful win rows where the candidate strictly beats the incumbent.
        win_activation_count_after:
            Useful win rows where the refined candidate beats the incumbent.
    """

    layer_index: int
    expert_id: str
    useful_win_row_count: int
    ignored_win_row_count: int
    status: str
    initial_stability_loss: float
    final_stability_loss: float
    initial_win_loss: float
    final_win_loss: float
    initial_total_loss: float
    final_total_loss: float
    stability_violation_count_before: int
    stability_violation_count_after: int
    win_violation_count_before: int
    win_violation_count_after: int
    stability_activation_count_before: int
    stability_activation_count_after: int
    win_activation_count_before: int
    win_activation_count_after: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible per-head report mapping."""

        return {
            "expert_id": self.expert_id,
            "final_stability_loss": self.final_stability_loss,
            "final_total_loss": self.final_total_loss,
            "final_win_loss": self.final_win_loss,
            "ignored_win_row_count": self.ignored_win_row_count,
            "initial_stability_loss": self.initial_stability_loss,
            "initial_total_loss": self.initial_total_loss,
            "initial_win_loss": self.initial_win_loss,
            "layer_index": self.layer_index,
            "stability_activation_count_after": (
                self.stability_activation_count_after
            ),
            "stability_activation_count_before": (
                self.stability_activation_count_before
            ),
            "stability_violation_count_after": (
                self.stability_violation_count_after
            ),
            "stability_violation_count_before": (
                self.stability_violation_count_before
            ),
            "status": self.status,
            "useful_win_row_count": self.useful_win_row_count,
            "win_activation_count_after": self.win_activation_count_after,
            "win_activation_count_before": self.win_activation_count_before,
            "win_violation_count_after": self.win_violation_count_after,
            "win_violation_count_before": self.win_violation_count_before,
        }


@dataclass(frozen=True)
class StabilityMarginRefinementReport:
    """Reproducibility metadata and per-head CAL-006 results.

    The report supersedes CAL-005 activation metrics for the returned heads
    because affine refinement changes their deployed scores. Row identifiers
    are canonicalized into feature-bank order and contain no prompt content.
    """

    strategy_name: ClassVar[str] = "stability_margin"
    optimizer_name: ClassVar[str] = "full_batch_adam_cpu_float64"

    stability_row_ids: tuple[int, ...]
    win_row_ids: tuple[int, ...]
    stability_margin: float
    win_margin: float
    stability_weight: float
    win_weight: float
    drift_weight: float
    learning_rate: float
    optimization_steps: int
    head_reports: tuple[HeadStabilityMarginReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible refinement metadata."""

        return {
            "drift_weight": self.drift_weight,
            "head_reports": [report.to_dict() for report in self.head_reports],
            "learning_rate": self.learning_rate,
            "optimization_steps": self.optimization_steps,
            "optimizer": self.optimizer_name,
            "stability_margin": self.stability_margin,
            "stability_row_ids": list(self.stability_row_ids),
            "stability_weight": self.stability_weight,
            "strategy": self.strategy_name,
            "win_margin": self.win_margin,
            "win_row_ids": list(self.win_row_ids),
            "win_weight": self.win_weight,
        }


@dataclass(frozen=True)
class _MarginMetrics:
    """Internal scalar metrics measured with deployed score arithmetic."""

    stability_loss: float
    win_loss: float
    stability_violations: int
    win_violations: int
    stability_activations: int
    win_activations: int


def refine_stability_margins(
    candidate_router: AppendOnlyRouterHeads,
    incumbent_router: RouterScoreBank | None,
    feature_bank: RouterFeatureBank,
    utility_labels: UtilityLabelBank,
    *,
    stability_row_ids: Sequence[int],
    win_row_ids: Sequence[int],
    stability_margin: float = 0.1,
    win_margin: float = 0.1,
    stability_weight: float = 1.0,
    win_weight: float = 1.0,
    drift_weight: float = 1e-3,
    learning_rate: float = 5e-2,
    optimization_steps: int = 256,
) -> tuple[AppendOnlyRouterHeads, StabilityMarginRefinementReport]:
    """Refine calibrated candidate heads without changing incumbent scores.

    For candidate score ``g_new`` and incumbent maximum

    ``m_old = max(0, max_k g_old,k)``,

    the per-head objective is

    ``stability_weight * E[max(0, g_new - m_old + stability_margin)]``

    on ``stability_row_ids`` plus

    ``win_weight * E[max(0, m_old + win_margin - g_new)]``

    on strictly positive-utility rows inside ``win_row_ids``. A proximal
    ``drift_weight`` penalty keeps affine parameters near their CAL-005
    initialization. Candidate heads are optimized independently, so installing
    one package never changes another package's score function.

    Optimization is deterministic and seed-free: every head uses full-batch
    CPU float64 Adam for a fixed step count, and the best observed objective
    (including the initialization) is retained. The result is copied back to
    each source head's dtype/device, re-evaluated with deployed arithmetic, and
    accepted only when that deployed objective does not exceed the original.
    Temperature and threshold metadata are preserved exactly. Source routers,
    feature banks, and utility labels are never mutated or aliased.

    ``incumbent_router=None`` represents an anchor-only registry. Otherwise
    the incumbent may use any frozen :class:`RouterScoreBank` implementation.
    Candidate and incumbent expert identifiers must be disjoint at every
    refined layer. Stability and win row partitions must be non-empty,
    disjoint subsets of the feature bank; unlike CAL-005, they need not cover
    every row.

    Args:
        candidate_router: Frozen CAL-005 candidate affine heads to refine.
        incumbent_router:
            Frozen deployed router or ``None`` for fixed-zero anchor only.
        feature_bank: CAL-001 bank containing refinement features.
        utility_labels:
            CAL-002 candidate utilities exactly aligned to ``feature_bank``.
        stability_row_ids:
            Old/general rows on which candidate route changes are discouraged.
        win_row_ids:
            New-domain rows; only positive-utility rows train each candidate.
        stability_margin:
            Non-negative distance below the incumbent required on stability
            rows.
        win_margin:
            Non-negative distance above the incumbent required on useful win
            rows.
        stability_weight: Positive stability-hinge coefficient.
        win_weight: Positive useful-win-hinge coefficient.
        drift_weight: Non-negative proximal parameter-drift coefficient.
        learning_rate: Positive full-batch Adam learning rate.
        optimization_steps: Positive fixed optimizer step count.

    Returns:
        ``(refined_router, report)`` with frozen independently owned heads and
        JSON-compatible post-refinement metrics.

    Raises:
        TypeError: If inputs, row IDs, or hyperparameters have invalid types.
        ValueError: If routers, banks, partitions, or ownership are incompatible.
        StabilityMarginRefinementError:
            If scoring or refined deployed parameters are non-finite.

    References:
        The repository architecture blueprint, "Approach E — Margin
        refinement." ``anchored-plugin-moe-architecture.md``.

    Differences from the blueprint:
        CAL-006 uses deterministic full-batch Adam and proximal affine drift
        control. It refines package-owned heads independently and reports
        deployed-dtype activation changes rather than mutating a global router.
    """

    _validate_inputs(
        candidate_router,
        incumbent_router,
        feature_bank,
        utility_labels,
    )
    normalized_stability_margin = _validate_non_negative_number(
        stability_margin,
        field_name="stability_margin",
    )
    normalized_win_margin = _validate_non_negative_number(
        win_margin,
        field_name="win_margin",
    )
    normalized_stability_weight = _validate_positive_number(
        stability_weight,
        field_name="stability_weight",
    )
    normalized_win_weight = _validate_positive_number(
        win_weight,
        field_name="win_weight",
    )
    normalized_drift_weight = _validate_non_negative_number(
        drift_weight,
        field_name="drift_weight",
    )
    normalized_learning_rate = _validate_positive_number(
        learning_rate,
        field_name="learning_rate",
    )
    normalized_steps = _validate_positive_integer(
        optimization_steps,
        field_name="optimization_steps",
    )
    stability_ids, stability_indices = _normalize_row_partition(
        stability_row_ids,
        row_ids=feature_bank.row_ids,
        field_name="stability_row_ids",
    )
    win_ids, win_indices = _normalize_row_partition(
        win_row_ids,
        row_ids=feature_bank.row_ids,
        field_name="win_row_ids",
    )
    if set(stability_ids) & set(win_ids):
        raise ValueError("stability_row_ids and win_row_ids must be disjoint.")

    incumbent_margins = {
        layer_index: _incumbent_margin(
            incumbent_router,
            feature_bank.features_for_layer(layer_index),
            layer_index=layer_index,
        )
        for layer_index in feature_bank.layer_indices
    }

    refined_router = AppendOnlyRouterHeads(candidate_router.feature_size)
    head_reports: list[HeadStabilityMarginReport] = []
    for source_head in candidate_router.heads:
        layer_index = source_head.layer_index
        expert_ids = utility_labels.expert_ids_for_layer(layer_index)
        expert_column = expert_ids.index(source_head.expert_id)
        utilities = utility_labels.utilities_for_layer(layer_index).to(
            dtype=torch.float64
        )[:, expert_column]
        useful_win_indices = win_indices[
            utilities[win_indices] > 0.0
        ]
        features = feature_bank.features_for_layer(layer_index)
        incumbent_margin = incumbent_margins[layer_index]
        initial_scores = _score_head(source_head, features)
        initial_metrics = _measure_metrics(
            initial_scores,
            incumbent_margin,
            stability_indices=stability_indices,
            useful_win_indices=useful_win_indices,
            stability_margin=normalized_stability_margin,
            win_margin=normalized_win_margin,
        )

        refined_weight, refined_bias = _optimize_head(
            source_head,
            features,
            incumbent_margin,
            stability_indices=stability_indices,
            useful_win_indices=useful_win_indices,
            stability_margin=normalized_stability_margin,
            win_margin=normalized_win_margin,
            stability_weight=normalized_stability_weight,
            win_weight=normalized_win_weight,
            drift_weight=normalized_drift_weight,
            learning_rate=normalized_learning_rate,
            optimization_steps=normalized_steps,
        )
        target_head = _copy_refined_head(
            source_head,
            refined_weight=refined_weight,
            refined_bias=refined_bias,
        )
        final_scores = _score_head(target_head, features)
        final_metrics = _measure_metrics(
            final_scores,
            incumbent_margin,
            stability_indices=stability_indices,
            useful_win_indices=useful_win_indices,
            stability_margin=normalized_stability_margin,
            win_margin=normalized_win_margin,
        )
        initial_total = (
            normalized_stability_weight * initial_metrics.stability_loss
            + normalized_win_weight * initial_metrics.win_loss
        )
        final_drift = _deployed_drift(source_head, target_head)
        final_total = (
            normalized_stability_weight * final_metrics.stability_loss
            + normalized_win_weight * final_metrics.win_loss
            + normalized_drift_weight * final_drift
        )
        # Dtype conversion can round an optimized float64 solution. Preserve
        # the exact CAL-005 source whenever deployed arithmetic does not improve
        # the complete objective.
        if final_total >= initial_total:
            target_head = _copy_refined_head(
                source_head,
                refined_weight=source_head.linear.weight.detach(),
                refined_bias=source_head.linear.bias.detach(),
            )
            final_metrics = initial_metrics
            final_total = initial_total
            status = "unchanged"
        else:
            status = "optimized"

        refined_router.append(target_head)
        head_reports.append(
            HeadStabilityMarginReport(
                layer_index=layer_index,
                expert_id=source_head.expert_id,
                useful_win_row_count=int(useful_win_indices.numel()),
                ignored_win_row_count=(
                    len(win_ids) - int(useful_win_indices.numel())
                ),
                status=status,
                initial_stability_loss=initial_metrics.stability_loss,
                final_stability_loss=final_metrics.stability_loss,
                initial_win_loss=initial_metrics.win_loss,
                final_win_loss=final_metrics.win_loss,
                initial_total_loss=initial_total,
                final_total_loss=final_total,
                stability_violation_count_before=(
                    initial_metrics.stability_violations
                ),
                stability_violation_count_after=(
                    final_metrics.stability_violations
                ),
                win_violation_count_before=initial_metrics.win_violations,
                win_violation_count_after=final_metrics.win_violations,
                stability_activation_count_before=(
                    initial_metrics.stability_activations
                ),
                stability_activation_count_after=(
                    final_metrics.stability_activations
                ),
                win_activation_count_before=initial_metrics.win_activations,
                win_activation_count_after=final_metrics.win_activations,
            )
        )

    return refined_router, StabilityMarginRefinementReport(
        stability_row_ids=stability_ids,
        win_row_ids=win_ids,
        stability_margin=normalized_stability_margin,
        win_margin=normalized_win_margin,
        stability_weight=normalized_stability_weight,
        win_weight=normalized_win_weight,
        drift_weight=normalized_drift_weight,
        learning_rate=normalized_learning_rate,
        optimization_steps=normalized_steps,
        head_reports=tuple(head_reports),
    )


def _validate_inputs(
    candidate_router: AppendOnlyRouterHeads,
    incumbent_router: RouterScoreBank | None,
    feature_bank: RouterFeatureBank,
    utility_labels: UtilityLabelBank,
) -> None:
    """Validate complete topology and immutability before any optimization."""

    if not isinstance(candidate_router, AppendOnlyRouterHeads):
        raise TypeError("candidate_router must be an AppendOnlyRouterHeads.")
    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")
    if not isinstance(utility_labels, UtilityLabelBank):
        raise TypeError("utility_labels must be a UtilityLabelBank.")
    if incumbent_router is not None and not isinstance(
        incumbent_router, RouterScoreBank
    ):
        raise TypeError("incumbent_router must be a RouterScoreBank or None.")
    if not candidate_router.heads:
        raise ValueError("candidate_router must contain at least one head.")
    if candidate_router.feature_size != feature_bank.feature_size:
        raise ValueError(
            "candidate_router feature size does not match the feature bank."
        )
    if incumbent_router is not None:
        if incumbent_router.feature_size != feature_bank.feature_size:
            raise ValueError(
                "incumbent_router feature size does not match the feature bank."
            )
        if any(parameter.requires_grad for parameter in incumbent_router.parameters()):
            raise ValueError("incumbent_router must be frozen.")
        _validate_finite_module(incumbent_router, field_name="incumbent_router")
    if any(parameter.requires_grad for parameter in candidate_router.parameters()):
        raise ValueError("candidate_router must be frozen CAL-005 heads.")
    _validate_finite_module(candidate_router, field_name="candidate_router")
    utility_labels.assert_aligned(feature_bank)

    candidate_layers = {
        head.layer_index for head in candidate_router.heads
    }
    if candidate_layers != set(feature_bank.layer_indices):
        raise ValueError(
            "candidate_router layers must exactly match the feature bank."
        )
    for layer_index in feature_bank.layer_indices:
        candidate_ids = candidate_router.expert_ids_for_layer(layer_index)
        label_ids = utility_labels.expert_ids_for_layer(layer_index)
        if candidate_ids != label_ids:
            raise ValueError(
                "candidate_router expert order must exactly match utility "
                f"labels at layer {layer_index}: {candidate_ids} != {label_ids}."
            )
        if incumbent_router is not None:
            incumbent_ids = incumbent_router.expert_ids_for_layer(layer_index)
            duplicates = set(candidate_ids) & set(incumbent_ids)
            if duplicates:
                raise ValueError(
                    "candidate and incumbent expert identifiers must be "
                    f"disjoint at layer {layer_index}: {sorted(duplicates)}."
                )


def _validate_finite_module(module: RouterScoreBank, *, field_name: str) -> None:
    """Reject non-finite floating router state.

    Integer buffers may encode harmless structural metadata in future
    ``RouterScoreBank`` implementations and do not participate in scoring
    arithmetic.
    """

    for name, tensor in module.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise StabilityMarginRefinementError(
                f"{field_name} tensor {name!r} must be finite floating-point."
            )


def _incumbent_margin(
    incumbent_router: RouterScoreBank | None,
    features: Tensor,
    *,
    layer_index: int,
) -> Tensor:
    """Return ``max(anchor, incumbent specialists)`` as CPU float64."""

    if incumbent_router is None:
        return torch.zeros(features.shape[0], dtype=torch.float64)
    expert_ids = incumbent_router.expert_ids_for_layer(layer_index)
    if not expert_ids:
        return torch.zeros(features.shape[0], dtype=torch.float64)
    state = next(
        (
            tensor
            for tensor in incumbent_router.state_dict().values()
            if tensor.is_floating_point()
        ),
        None,
    )
    if state is None:
        raise StabilityMarginRefinementError(
            "incumbent_router has experts but no tensor state."
        )
    deployed_features = features.to(device=state.device, dtype=state.dtype)
    with torch.no_grad():
        scores = incumbent_router(
            deployed_features,
            layer_index=layer_index,
        )
    expected_shape = (features.shape[0], len(expert_ids))
    if scores.shape != expected_shape or not scores.is_floating_point():
        raise StabilityMarginRefinementError(
            "incumbent_router returned incompatible scores at layer "
            f"{layer_index}: {tuple(scores.shape)} != {expected_shape}."
        )
    if not torch.isfinite(scores).all().item():
        raise StabilityMarginRefinementError(
            f"incumbent_router produced non-finite scores at layer {layer_index}."
        )
    return (
        scores.detach()
        .to(device="cpu", dtype=torch.float64)
        .max(dim=-1)
        .values.clamp_min(0.0)
    )


def _score_head(head: AffineRouterHead, features: Tensor) -> Tensor:
    """Return deployed head scores as detached CPU float64."""

    deployed_features = features.to(
        device=head.linear.weight.device,
        dtype=head.linear.weight.dtype,
    )
    with torch.no_grad():
        scores = head(deployed_features)
    if scores.shape != (features.shape[0],) or not torch.isfinite(scores).all().item():
        raise StabilityMarginRefinementError(
            f"Candidate head {head.key!r} produced invalid deployed scores."
        )
    return scores.detach().to(device="cpu", dtype=torch.float64)


def _optimize_head(
    head: AffineRouterHead,
    features: Tensor,
    incumbent_margin: Tensor,
    *,
    stability_indices: Tensor,
    useful_win_indices: Tensor,
    stability_margin: float,
    win_margin: float,
    stability_weight: float,
    win_weight: float,
    drift_weight: float,
    learning_rate: float,
    optimization_steps: int,
) -> tuple[Tensor, Tensor]:
    """Return the best deterministic float64 affine state for one head."""

    design = features.to(device="cpu", dtype=torch.float64)
    initial_weight = (
        head.linear.weight.detach().to(device="cpu", dtype=torch.float64).clone()
    )
    initial_bias = (
        head.linear.bias.detach().to(device="cpu", dtype=torch.float64).clone()
    )
    weight = initial_weight.clone().requires_grad_(True)
    bias = initial_bias.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)

    with torch.no_grad():
        initial_total = _objective(
            design,
            weight,
            bias,
            temperature=head.temperature,
            threshold=head.threshold,
            incumbent_margin=incumbent_margin,
            stability_indices=stability_indices,
            useful_win_indices=useful_win_indices,
            stability_margin=stability_margin,
            win_margin=win_margin,
            stability_weight=stability_weight,
            win_weight=win_weight,
            drift_weight=drift_weight,
            initial_weight=initial_weight,
            initial_bias=initial_bias,
        )
    best_value = float(initial_total.item())
    best_weight = initial_weight.clone()
    best_bias = initial_bias.clone()

    for _ in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        total = _objective(
            design,
            weight,
            bias,
            temperature=head.temperature,
            threshold=head.threshold,
            incumbent_margin=incumbent_margin,
            stability_indices=stability_indices,
            useful_win_indices=useful_win_indices,
            stability_margin=stability_margin,
            win_margin=win_margin,
            stability_weight=stability_weight,
            win_weight=win_weight,
            drift_weight=drift_weight,
            initial_weight=initial_weight,
            initial_bias=initial_bias,
        )
        if not torch.isfinite(total).item():
            raise StabilityMarginRefinementError(
                f"CAL-006 objective became non-finite for head {head.key!r}."
            )
        total.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_total = _objective(
                design,
                weight,
                bias,
                temperature=head.temperature,
                threshold=head.threshold,
                incumbent_margin=incumbent_margin,
                stability_indices=stability_indices,
                useful_win_indices=useful_win_indices,
                stability_margin=stability_margin,
                win_margin=win_margin,
                stability_weight=stability_weight,
                win_weight=win_weight,
                drift_weight=drift_weight,
                initial_weight=initial_weight,
                initial_bias=initial_bias,
            )
            candidate_value = float(candidate_total.item())
            if not math.isfinite(candidate_value):
                raise StabilityMarginRefinementError(
                    f"CAL-006 produced non-finite state for head {head.key!r}."
                )
            if candidate_value < best_value:
                best_value = candidate_value
                best_weight = weight.detach().clone()
                best_bias = bias.detach().clone()

    return best_weight, best_bias


def _objective(
    features: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    temperature: float,
    threshold: float,
    incumbent_margin: Tensor,
    stability_indices: Tensor,
    useful_win_indices: Tensor,
    stability_margin: float,
    win_margin: float,
    stability_weight: float,
    win_weight: float,
    drift_weight: float,
    initial_weight: Tensor,
    initial_bias: Tensor,
) -> Tensor:
    """Return weighted hinge plus mean proximal drift."""

    scores = (features @ weight.T + bias).squeeze(-1) / temperature - threshold
    stability_loss = torch.relu(
        scores[stability_indices]
        - incumbent_margin[stability_indices]
        + stability_margin
    ).mean()
    if useful_win_indices.numel():
        win_loss = torch.relu(
            incumbent_margin[useful_win_indices]
            + win_margin
            - scores[useful_win_indices]
        ).mean()
    else:
        win_loss = scores.new_zeros(())
    drift = torch.cat(
        (
            (weight - initial_weight).reshape(-1),
            (bias - initial_bias).reshape(-1),
        )
    ).square().mean()
    return (
        stability_weight * stability_loss
        + win_weight * win_loss
        + drift_weight * drift
    )


def _copy_refined_head(
    source: AffineRouterHead,
    *,
    refined_weight: Tensor,
    refined_bias: Tensor,
) -> AffineRouterHead:
    """Copy affine state to one frozen source-compatible deployed head."""

    target = AffineRouterHead(
        source.feature_size,
        layer_index=source.layer_index,
        expert_id=source.expert_id,
        temperature=source.temperature,
        threshold=source.threshold,
        trainable=False,
    ).to(
        device=source.linear.weight.device,
        dtype=source.linear.weight.dtype,
    )
    converted_weight = refined_weight.detach().to(
        device=source.linear.weight.device,
        dtype=source.linear.weight.dtype,
    )
    converted_bias = refined_bias.detach().to(
        device=source.linear.bias.device,
        dtype=source.linear.bias.dtype,
    )
    if not torch.isfinite(converted_weight).all().item() or not torch.isfinite(
        converted_bias
    ).all().item():
        raise StabilityMarginRefinementError(
            f"Refined head {source.key!r} is non-finite in deployed dtype."
        )
    with torch.no_grad():
        target.linear.weight.copy_(converted_weight)
        target.linear.bias.copy_(converted_bias)
    return target


def _measure_metrics(
    scores: Tensor,
    incumbent_margin: Tensor,
    *,
    stability_indices: Tensor,
    useful_win_indices: Tensor,
    stability_margin: float,
    win_margin: float,
) -> _MarginMetrics:
    """Measure hinge and strict activation counts from deployed scores."""

    stability_hinge = torch.relu(
        scores[stability_indices]
        - incumbent_margin[stability_indices]
        + stability_margin
    )
    if useful_win_indices.numel():
        win_hinge = torch.relu(
            incumbent_margin[useful_win_indices]
            + win_margin
            - scores[useful_win_indices]
        )
        win_loss = float(win_hinge.mean().item())
        win_violations = int((win_hinge > 0.0).sum().item())
        win_activations = int(
            (
                scores[useful_win_indices]
                > incumbent_margin[useful_win_indices]
            )
            .sum()
            .item()
        )
    else:
        win_loss = 0.0
        win_violations = 0
        win_activations = 0
    return _MarginMetrics(
        stability_loss=float(stability_hinge.mean().item()),
        win_loss=win_loss,
        stability_violations=int((stability_hinge > 0.0).sum().item()),
        win_violations=win_violations,
        stability_activations=int(
            (
                scores[stability_indices]
                > incumbent_margin[stability_indices]
            )
            .sum()
            .item()
        ),
        win_activations=win_activations,
    )


def _deployed_drift(
    source: AffineRouterHead,
    target: AffineRouterHead,
) -> float:
    """Return mean squared affine drift in CPU float64."""

    difference = torch.cat(
        (
            (
                target.linear.weight.detach().to(dtype=torch.float64, device="cpu")
                - source.linear.weight.detach().to(
                    dtype=torch.float64, device="cpu"
                )
            ).reshape(-1),
            (
                target.linear.bias.detach().to(dtype=torch.float64, device="cpu")
                - source.linear.bias.detach().to(
                    dtype=torch.float64, device="cpu"
                )
            ).reshape(-1),
        )
    )
    return float(difference.square().mean().item())


def _normalize_row_partition(
    values: Sequence[int],
    *,
    row_ids: Tensor,
    field_name: str,
) -> tuple[tuple[int, ...], Tensor]:
    """Validate IDs and return canonical IDs plus positional indices."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of integers.")
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must contain only integers.")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} contains duplicate row identifiers.")

    positions = {int(row_id): index for index, row_id in enumerate(row_ids.tolist())}
    unknown = sorted(set(normalized) - set(positions))
    if unknown:
        raise ValueError(f"{field_name} contains unknown row identifiers: {unknown}.")
    canonical = tuple(row_id for row_id in positions if row_id in set(normalized))
    indices = torch.tensor(
        [positions[row_id] for row_id in canonical],
        dtype=torch.long,
    )
    return canonical, indices


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_non_negative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _validate_positive_number(value: Any, *, field_name: str) -> float:
    normalized = _validate_non_negative_number(value, field_name=field_name)
    if normalized == 0.0:
        raise ValueError(f"{field_name} must be strictly positive.")
    return normalized
