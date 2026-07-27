"""Post-hoc temperature and threshold calibration for utility router heads.

References:
    Guo et al., "On Calibration of Modern Neural Networks."
    https://arxiv.org/abs/1706.04599

Differences from the paper:
    CAL-005 calibrates one binary positive-utility logit per independent expert
    head rather than one shared temperature for multiclass logits. A separate
    held-out row partition then selects an activation threshold under an
    explicit false-activation budget.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F

from anchored_moe.calibration.features import RouterFeatureBank
from anchored_moe.calibration.labels import UtilityLabelBank
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads


_MIN_TEMPERATURE = 1e-2
_MAX_TEMPERATURE = 1e2
_TEMPERATURE_SEARCH_ITERATIONS = 96
_GOLDEN_RATIO_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0


class TemperatureThresholdCalibrationError(ValueError):
    """Raised when CAL-005 cannot produce a valid calibrated router."""


@dataclass(frozen=True)
class HeadCalibrationReport:
    """Metrics and fitted metadata for one layer-expert router head.

    Attributes:
        layer_index: Zero-based decoder layer owned by the head.
        expert_id: Stable expert identifier.
        temperature: Positive fitted temperature.
        threshold: Finite threshold subtracted from the scaled raw score.
        temperature_status:
            ``"optimized"`` or ``"single_class_fallback"``.
        threshold_status:
            ``"optimized"``, ``"no_positive_fallback"``, or
            ``"no_negative_fallback"``.
        temperature_row_count: Rows used only for temperature fitting.
        threshold_row_count: Held-out rows used only for threshold selection.
        positive_count: Positive-utility rows in the threshold partition.
        negative_count: Non-positive-utility rows in the threshold partition.
        true_positive_count: Positive rows activated by the selected threshold.
        false_positive_count:
            Non-positive rows activated by the selected threshold.
        true_positive_rate:
            ``true_positive_count / positive_count``, or zero when no positive
            threshold rows exist.
        false_activation_rate:
            ``false_positive_count / negative_count``, or zero when no
            negative threshold rows exist.
        binary_log_loss_before:
            Temperature-partition loss at temperature one.
        binary_log_loss_after:
            Temperature-partition loss at the fitted temperature.
    """

    layer_index: int
    expert_id: str
    temperature: float
    threshold: float
    temperature_status: str
    threshold_status: str
    temperature_row_count: int
    threshold_row_count: int
    positive_count: int
    negative_count: int
    true_positive_count: int
    false_positive_count: int
    true_positive_rate: float
    false_activation_rate: float
    binary_log_loss_before: float
    binary_log_loss_after: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report mapping."""

        return {
            "binary_log_loss_after": self.binary_log_loss_after,
            "binary_log_loss_before": self.binary_log_loss_before,
            "expert_id": self.expert_id,
            "false_activation_rate": self.false_activation_rate,
            "false_positive_count": self.false_positive_count,
            "layer_index": self.layer_index,
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "temperature": self.temperature,
            "temperature_row_count": self.temperature_row_count,
            "temperature_status": self.temperature_status,
            "threshold": self.threshold,
            "threshold_row_count": self.threshold_row_count,
            "threshold_status": self.threshold_status,
            "true_positive_count": self.true_positive_count,
            "true_positive_rate": self.true_positive_rate,
        }


@dataclass(frozen=True)
class TemperatureThresholdCalibrationReport:
    """Reproducibility metadata and per-head results for CAL-005.

    Row identifiers are stored in canonical feature-bank order. Together with
    the fixed temperature bounds and iteration count, they make the fitting
    and threshold-selection procedure auditable without storing prompt text.

    Attributes:
        false_activation_budget:
            Maximum permitted false-activation rate for each head.
        temperature_row_ids: Rows used only to fit temperature.
        threshold_row_ids: Disjoint held-out rows used to choose thresholds.
        head_reports: Results in the calibrated router's global append order.
    """

    strategy_name: ClassVar[str] = "temperature_threshold"

    false_activation_budget: float
    temperature_row_ids: tuple[int, ...]
    threshold_row_ids: tuple[int, ...]
    head_reports: tuple[HeadCalibrationReport, ...]
    minimum_temperature: float = _MIN_TEMPERATURE
    maximum_temperature: float = _MAX_TEMPERATURE
    temperature_search_iterations: int = _TEMPERATURE_SEARCH_ITERATIONS

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible calibration metadata."""

        return {
            "false_activation_budget": self.false_activation_budget,
            "head_reports": [report.to_dict() for report in self.head_reports],
            "maximum_temperature": self.maximum_temperature,
            "minimum_temperature": self.minimum_temperature,
            "strategy": self.strategy_name,
            "temperature_row_ids": list(self.temperature_row_ids),
            "temperature_search_iterations": self.temperature_search_iterations,
            "threshold_row_ids": list(self.threshold_row_ids),
        }


def calibrate_temperature_and_thresholds(
    router: AppendOnlyRouterHeads,
    feature_bank: RouterFeatureBank,
    utility_labels: UtilityLabelBank,
    *,
    temperature_row_ids: Sequence[int],
    threshold_row_ids: Sequence[int],
    false_activation_rate: float = 0.01,
) -> tuple[AppendOnlyRouterHeads, TemperatureThresholdCalibrationReport]:
    """Return a calibrated frozen copy of CAL-004 affine heads.

    Raw affine score ``s_(l,k)`` is interpreted as a binary logit for the
    target ``z_(l,t,k) = 1[a_(l,t,k) > 0]``. For each head, temperature fitting
    minimizes binary log loss on ``temperature_row_ids``:

    ``T* = argmin_(T > 0) BCEWithLogits(s / T, z)``.

    The implementation performs a fixed 96-iteration bounded search over
    inverse temperature, which is a convex scalar objective. If the
    temperature partition contains only one target class, temperature one is
    retained and the fallback is reported explicitly.

    On the disjoint ``threshold_row_ids`` partition, candidate thresholds are
    score values under the strict activation rule ``s / T - threshold > 0``.
    Among candidates meeting ``false_activation_rate``, selection maximizes
    true positives, then minimizes false positives, then prefers the higher
    threshold. With no positive rows this yields a conservative no-activation
    threshold. With no negative rows the false-activation rate is defined as
    zero and recall is maximized.

    The two row-ID sequences must be non-empty, disjoint, and together cover
    every feature-bank row. They are canonicalized into feature-bank order.
    The input router must contain uncalibrated CAL-004 heads with temperature
    one and threshold zero. Its parameters, metadata, device, dtype, and
    trainability flags are never mutated. The returned heads preserve the
    original global append order and affine tensors but are frozen.

    Args:
        router: Non-empty RTR-002 head bank produced by CAL-004.
        feature_bank: CAL-001 bank containing held-out calibration features.
        utility_labels: CAL-002 utilities exactly aligned to ``feature_bank``.
        temperature_row_ids:
            Explicit rows used only for binary-log-loss temperature fitting.
        threshold_row_ids:
            Explicit disjoint rows used only for threshold selection.
        false_activation_rate:
            Finite per-head budget in the inclusive interval ``[0, 1]``.

    Returns:
        ``(calibrated_router, report)``. The router is a new frozen
        :class:`~anchored_moe.routers.AppendOnlyRouterHeads`; the report is
        JSON-compatible through :meth:`TemperatureThresholdCalibrationReport.to_dict`.

    Raises:
        TypeError:
            If inputs, row identifiers, or the budget have invalid types.
        ValueError:
            If banks, heads, partitions, or budget values are incompatible.
        TemperatureThresholdCalibrationError:
            If finite calibrated metadata cannot be produced.

    References:
        Guo et al., "On Calibration of Modern Neural Networks."
        https://arxiv.org/abs/1706.04599

    Differences from the paper:
        Each expert uses a binary positive-utility target and independent
        temperature. Threshold selection is a project-specific conservative
        policy on a separate held-out partition, not part of Guo et al.
    """

    _validate_inputs(router, feature_bank, utility_labels)
    budget = _validate_false_activation_rate(false_activation_rate)
    temperature_ids, temperature_indices = _normalize_row_partition(
        temperature_row_ids,
        row_ids=feature_bank.row_ids,
        field_name="temperature_row_ids",
    )
    threshold_ids, threshold_indices = _normalize_row_partition(
        threshold_row_ids,
        row_ids=feature_bank.row_ids,
        field_name="threshold_row_ids",
    )
    _validate_complete_disjoint_partition(
        temperature_ids,
        threshold_ids,
        all_row_ids=tuple(feature_bank.row_ids.tolist()),
    )

    raw_scores_by_key: dict[tuple[int, str], Tensor] = {}
    utilities_by_key: dict[tuple[int, str], Tensor] = {}
    for layer_index in feature_bank.layer_indices:
        features = feature_bank.features_for_layer(layer_index)
        utilities = utility_labels.utilities_for_layer(layer_index).to(
            device="cpu",
            dtype=torch.float64,
        )
        expert_ids = utility_labels.expert_ids_for_layer(layer_index)
        for expert_column, expert_id in enumerate(expert_ids):
            head = _head_for_key(router, (layer_index, expert_id))
            deployed_features = features.to(
                device="cpu",
                dtype=head.linear.weight.dtype,
            )
            deployed_raw_scores = F.linear(
                deployed_features,
                head.linear.weight.detach().to(device="cpu"),
                head.linear.bias.detach().to(device="cpu"),
            ).squeeze(-1)
            if not torch.isfinite(deployed_raw_scores).all().item():
                raise TemperatureThresholdCalibrationError(
                    f"Router head {head.key!r} produced non-finite raw scores."
                )
            # Keep native scores for threshold selection so strict comparisons
            # exactly match deployed head arithmetic and dtype. Temperature
            # optimization below converts its selected rows to float64.
            raw_scores_by_key[head.key] = deployed_raw_scores
            utilities_by_key[head.key] = utilities[:, expert_column]

    calibrated = AppendOnlyRouterHeads(feature_size=router.feature_size)
    reports: list[HeadCalibrationReport] = []
    for source_head in router.heads:
        raw_scores = raw_scores_by_key[source_head.key]
        utilities = utilities_by_key[source_head.key]
        temperature_targets = utilities[temperature_indices] > 0.0
        temperature, status, loss_before, loss_after = _fit_temperature(
            raw_scores[temperature_indices].to(dtype=torch.float64),
            temperature_targets,
        )
        threshold_targets = utilities[threshold_indices] > 0.0
        threshold, threshold_status, counts = _select_threshold(
            raw_scores[threshold_indices] / temperature,
            threshold_targets,
            false_activation_budget=budget,
        )

        target_head = AffineRouterHead(
            router.feature_size,
            layer_index=source_head.layer_index,
            expert_id=source_head.expert_id,
            temperature=temperature,
            threshold=threshold,
            trainable=False,
        ).to(
            device=source_head.linear.weight.device,
            dtype=source_head.linear.weight.dtype,
        )
        with torch.no_grad():
            target_head.linear.weight.copy_(source_head.linear.weight)
            target_head.linear.bias.copy_(source_head.linear.bias)
        calibrated.append(target_head)

        positive_count, negative_count, true_positives, false_positives = counts
        reports.append(
            HeadCalibrationReport(
                layer_index=source_head.layer_index,
                expert_id=source_head.expert_id,
                temperature=temperature,
                threshold=threshold,
                temperature_status=status,
                threshold_status=threshold_status,
                temperature_row_count=len(temperature_ids),
                threshold_row_count=len(threshold_ids),
                positive_count=positive_count,
                negative_count=negative_count,
                true_positive_count=true_positives,
                false_positive_count=false_positives,
                true_positive_rate=(
                    true_positives / positive_count if positive_count else 0.0
                ),
                false_activation_rate=(
                    false_positives / negative_count if negative_count else 0.0
                ),
                binary_log_loss_before=loss_before,
                binary_log_loss_after=loss_after,
            )
        )

    return calibrated, TemperatureThresholdCalibrationReport(
        false_activation_budget=budget,
        temperature_row_ids=temperature_ids,
        threshold_row_ids=threshold_ids,
        head_reports=tuple(reports),
    )


def _validate_inputs(
    router: Any,
    feature_bank: Any,
    utility_labels: Any,
) -> None:
    """Validate exact CAL-004 head and CAL-001/CAL-002 alignment contracts."""

    if not isinstance(router, AppendOnlyRouterHeads):
        raise TypeError("router must be an AppendOnlyRouterHeads.")
    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")
    if not isinstance(utility_labels, UtilityLabelBank):
        raise TypeError("utility_labels must be a UtilityLabelBank.")
    if not len(router):
        raise ValueError("router must contain at least one affine head.")
    if router.feature_size != feature_bank.feature_size:
        raise ValueError(
            "Router and feature-bank sizes differ: "
            f"{router.feature_size} != {feature_bank.feature_size}."
        )
    utility_labels.assert_aligned(feature_bank)
    router_layers = tuple(sorted({head.layer_index for head in router.heads}))
    if router_layers != feature_bank.layer_indices:
        raise ValueError(
            "Router layers do not match calibration-bank layers: "
            f"{router_layers} != {feature_bank.layer_indices}."
        )
    for layer_index in feature_bank.layer_indices:
        router_ids = router.expert_ids_for_layer(layer_index)
        utility_ids = utility_labels.expert_ids_for_layer(layer_index)
        if router_ids != utility_ids:
            raise ValueError(
                f"Router expert order for layer {layer_index} does not match "
                f"utility labels: {router_ids} != {utility_ids}."
            )
    for head in router.heads:
        if head.temperature != 1.0 or head.threshold != 0.0:
            raise ValueError(
                "CAL-005 requires uncalibrated CAL-004 heads with temperature "
                f"1 and threshold 0; got {head.key!r} with temperature "
                f"{head.temperature} and threshold {head.threshold}."
            )
        if not torch.isfinite(head.linear.weight).all().item() or not torch.isfinite(
            head.linear.bias
        ).all().item():
            raise ValueError(f"Router head {head.key!r} contains non-finite values.")


def _normalize_row_partition(
    values: Sequence[int],
    *,
    row_ids: Tensor,
    field_name: str,
) -> tuple[tuple[int, ...], Tensor]:
    """Validate row identifiers and return bank-ordered IDs and indices."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of integers.")
    requested: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must contain only integers.")
        requested.append(value)
    if not requested:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{field_name} must not contain duplicate row IDs.")

    bank_ids = tuple(row_ids.tolist())
    unknown = sorted(set(requested) - set(bank_ids))
    if unknown:
        raise ValueError(f"{field_name} contains unknown row IDs: {unknown}.")
    requested_set = set(requested)
    canonical_ids = tuple(row_id for row_id in bank_ids if row_id in requested_set)
    index_by_id = {row_id: index for index, row_id in enumerate(bank_ids)}
    indices = torch.tensor(
        [index_by_id[row_id] for row_id in canonical_ids],
        dtype=torch.long,
    )
    return canonical_ids, indices


def _validate_complete_disjoint_partition(
    temperature_ids: tuple[int, ...],
    threshold_ids: tuple[int, ...],
    *,
    all_row_ids: tuple[int, ...],
) -> None:
    """Require two non-overlapping partitions that cover the calibration bank."""

    overlap = sorted(set(temperature_ids) & set(threshold_ids))
    if overlap:
        raise ValueError(
            "temperature_row_ids and threshold_row_ids must be disjoint; "
            f"overlap={overlap}."
        )
    covered = set(temperature_ids) | set(threshold_ids)
    missing = [row_id for row_id in all_row_ids if row_id not in covered]
    if missing:
        raise ValueError(
            "Temperature and threshold row IDs must cover the feature bank; "
            f"missing={missing}."
        )


def _fit_temperature(
    raw_scores: Tensor,
    targets: Tensor,
) -> tuple[float, str, float, float]:
    """Fit one temperature with deterministic convex inverse-scale search."""

    loss_before = _binary_log_loss(raw_scores, targets)
    if bool(torch.all(targets == targets[0]).item()):
        # One-class calibration cannot identify a meaningful confidence scale.
        # Retaining the neutral temperature avoids manufacturing certainty.
        return 1.0, "single_class_fallback", loss_before, loss_before

    lower = 1.0 / _MAX_TEMPERATURE
    upper = 1.0 / _MIN_TEMPERATURE
    left = upper - _GOLDEN_RATIO_CONJUGATE * (upper - lower)
    right = lower + _GOLDEN_RATIO_CONJUGATE * (upper - lower)
    left_loss = _binary_log_loss(raw_scores * left, targets)
    right_loss = _binary_log_loss(raw_scores * right, targets)
    for _ in range(_TEMPERATURE_SEARCH_ITERATIONS):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - _GOLDEN_RATIO_CONJUGATE * (upper - lower)
            left_loss = _binary_log_loss(raw_scores * left, targets)
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + _GOLDEN_RATIO_CONJUGATE * (upper - lower)
            right_loss = _binary_log_loss(raw_scores * right, targets)

    candidates = (lower, left, (lower + upper) / 2.0, right, upper, 1.0)
    inverse_temperature, loss_after = min(
        (
            (candidate, _binary_log_loss(raw_scores * candidate, targets))
            for candidate in candidates
            if 1.0 / _MAX_TEMPERATURE <= candidate <= 1.0 / _MIN_TEMPERATURE
        ),
        key=lambda item: (item[1], abs(item[0] - 1.0), item[0]),
    )
    temperature = 1.0 / inverse_temperature
    if not math.isfinite(temperature) or not math.isfinite(loss_after):
        raise TemperatureThresholdCalibrationError(
            "Temperature fitting produced non-finite calibration metadata."
        )
    return temperature, "optimized", loss_before, loss_after


def _select_threshold(
    calibrated_scores: Tensor,
    targets: Tensor,
    *,
    false_activation_budget: float,
) -> tuple[float, str, tuple[int, int, int, int]]:
    """Select a deterministic strict-activation threshold under an FPR budget."""

    positive_count = int(targets.sum().item())
    negative_count = targets.numel() - positive_count
    feasible: list[tuple[tuple[int, int, float], float, int, int]] = []

    def consider(
        threshold: float,
        true_positives: int,
        false_positives: int,
    ) -> None:
        """Record one finite threshold when it satisfies the FPR budget."""

        observed_rate = (
            false_positives / negative_count if negative_count else 0.0
        )
        if observed_rate <= false_activation_budget + 1e-15:
            # Maximum utility recall is primary. For equal recall, fewer false
            # activations and then a higher threshold give conservative ties.
            key = (-true_positives, false_positives, -threshold)
            feasible.append((key, threshold, true_positives, false_positives))

    # Evaluate score groups in descending order. At threshold q, strict
    # activation includes only groups already traversed with score > q.
    # Cumulative counts avoid an O(N^2) mask for every unique threshold.
    order = torch.argsort(calibrated_scores, descending=True, stable=True)
    sorted_scores = calibrated_scores[order]
    sorted_targets = targets[order]
    true_positives = 0
    false_positives = 0
    position = 0
    while position < sorted_scores.numel():
        score = sorted_scores[position]
        threshold = float(score.item())
        consider(threshold, true_positives, false_positives)
        group_end = position + 1
        while (
            group_end < sorted_scores.numel()
            and bool((sorted_scores[group_end] == score).item())
        ):
            group_end += 1
        group_targets = sorted_targets[position:group_end]
        group_positives = int(group_targets.sum().item())
        true_positives += group_positives
        false_positives += group_targets.numel() - group_positives
        position = group_end

    minimum_tensor = sorted_scores[-1]
    below_minimum = float(
        torch.nextafter(
            minimum_tensor,
            calibrated_scores.new_tensor(-math.inf),
        ).item()
    )
    if math.isfinite(below_minimum):
        consider(below_minimum, true_positives, false_positives)

    if not feasible:
        raise TemperatureThresholdCalibrationError(
            "No finite threshold satisfies the false-activation budget."
        )
    _, threshold, true_positives, false_positives = min(feasible)
    if not math.isfinite(threshold):
        raise TemperatureThresholdCalibrationError(
            "Threshold selection produced a non-finite value."
        )
    if not positive_count:
        status = "no_positive_fallback"
    elif not negative_count:
        status = "no_negative_fallback"
    else:
        status = "optimized"
    return threshold, status, (
        positive_count,
        negative_count,
        true_positives,
        false_positives,
    )


def _binary_log_loss(scores: Tensor, targets: Tensor) -> float:
    """Return stable mean binary log loss for float64 logits."""

    loss = F.binary_cross_entropy_with_logits(
        scores,
        targets.to(dtype=torch.float64),
        reduction="mean",
    )
    value = float(loss.item())
    if not math.isfinite(value):
        raise TemperatureThresholdCalibrationError(
            "Binary log loss is not finite."
        )
    return value


def _head_for_key(
    router: AppendOnlyRouterHeads,
    key: tuple[int, str],
) -> AffineRouterHead:
    """Return the uniquely owned head for a validated layer-expert key."""

    for head in router.heads:
        if head.key == key:
            return head
    raise TemperatureThresholdCalibrationError(
        f"Router head disappeared during calibration: {key!r}."
    )


def _validate_false_activation_rate(value: Any) -> float:
    """Return a finite false-activation budget in ``[0, 1]``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("false_activation_rate must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(
            "false_activation_rate must be finite and between 0 and 1."
        )
    return normalized
