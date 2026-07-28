"""Deterministic registry-wide recalibration of a frozen logical router.

RTR-004 jointly refits every affine specialist column against individual
utility replay targets while the fixed-zero anchor participates as class zero.
The source RTR-003 router is never mutated. The returned router is a new
buffer-only deployment object bound by its report to one exact registry hash.

This is project-specific optimization infrastructure rather than an
implementation copied from a paper. The anchored classification objective is
the explicit router-only realization of Mode B in the architecture blueprint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F

from anchored_moe.calibration.features import RouterFeatureBank
from anchored_moe.calibration.labels import UtilityLabelBank
from anchored_moe.routers.heads import AppendOnlyRouterHeads
from anchored_moe.routers.logical import (
    LogicalAppendOnlyRouter,
    compile_logical_router,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class GlobalRouterRecalibrationError(ValueError):
    """Raised when RTR-004 cannot produce a finite registry-bound router."""


@dataclass(frozen=True)
class LayerGlobalRouterRecalibrationReport:
    """Before/after route-objective metrics for one decoder layer."""

    layer_index: int
    expert_ids: tuple[str, ...]
    row_count: int
    anchor_target_count: int
    initial_cross_entropy: float
    final_cross_entropy: float
    initial_total_loss: float
    final_total_loss: float
    initial_correct_count: int
    final_correct_count: int
    changed_route_count: int

    def __post_init__(self) -> None:
        """Validate public report construction and finite scalar metrics."""

        _validate_non_negative_integer(self.layer_index, field_name="layer_index")
        _validate_expert_ids(self.expert_ids)
        _validate_positive_integer(self.row_count, field_name="row_count")
        _validate_count(
            self.anchor_target_count,
            row_count=self.row_count,
            field_name="anchor_target_count",
        )
        for field_name in (
            "initial_cross_entropy",
            "final_cross_entropy",
            "initial_total_loss",
            "final_total_loss",
        ):
            _validate_non_negative_number(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in (
            "initial_correct_count",
            "final_correct_count",
            "changed_route_count",
        ):
            _validate_count(
                getattr(self, field_name),
                row_count=self.row_count,
                field_name=field_name,
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible layer report."""

        return {
            "anchor_target_count": self.anchor_target_count,
            "changed_route_count": self.changed_route_count,
            "expert_ids": list(self.expert_ids),
            "final_correct_count": self.final_correct_count,
            "final_cross_entropy": self.final_cross_entropy,
            "final_total_loss": self.final_total_loss,
            "initial_correct_count": self.initial_correct_count,
            "initial_cross_entropy": self.initial_cross_entropy,
            "initial_total_loss": self.initial_total_loss,
            "layer_index": self.layer_index,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> LayerGlobalRouterRecalibrationReport:
        """Strictly reconstruct one serialized layer report."""

        expected = {
            "anchor_target_count",
            "changed_route_count",
            "expert_ids",
            "final_correct_count",
            "final_cross_entropy",
            "final_total_loss",
            "initial_correct_count",
            "initial_cross_entropy",
            "initial_total_loss",
            "layer_index",
            "row_count",
        }
        _require_exact_fields(value, expected, field_name="layer report")
        expert_ids = value["expert_ids"]
        if not isinstance(expert_ids, list):
            raise TypeError("layer report expert_ids must be a list.")
        return cls(
            layer_index=value["layer_index"],
            expert_ids=tuple(expert_ids),
            row_count=value["row_count"],
            anchor_target_count=value["anchor_target_count"],
            initial_cross_entropy=value["initial_cross_entropy"],
            final_cross_entropy=value["final_cross_entropy"],
            initial_total_loss=value["initial_total_loss"],
            final_total_loss=value["final_total_loss"],
            initial_correct_count=value["initial_correct_count"],
            final_correct_count=value["final_correct_count"],
            changed_route_count=value["changed_route_count"],
        )


@dataclass(frozen=True)
class GlobalRouterRecalibrationReport:
    """Registry binding, inputs, hyperparameters, and RTR-004 outcomes."""

    strategy_name: ClassVar[str] = "router_only_global"
    optimizer_name: ClassVar[str] = "full_batch_adam_cpu_float64"
    objective_name: ClassVar[str] = "anchored_best_route_cross_entropy"

    registry_sha256: str
    compatibility_sha256: str
    feature_bank_metadata_sha256: str
    utility_label_metadata_sha256: str
    loss_name: str
    learning_rate: float
    optimization_steps: int
    drift_weight: float
    layer_reports: tuple[LayerGlobalRouterRecalibrationReport, ...]

    def __post_init__(self) -> None:
        """Validate exact binding, hyperparameters, and layer uniqueness."""

        _validate_sha256(self.registry_sha256, field_name="registry_sha256")
        _validate_sha256(
            self.compatibility_sha256,
            field_name="compatibility_sha256",
        )
        _validate_sha256(
            self.feature_bank_metadata_sha256,
            field_name="feature_bank_metadata_sha256",
        )
        _validate_sha256(
            self.utility_label_metadata_sha256,
            field_name="utility_label_metadata_sha256",
        )
        _validate_identifier(self.loss_name, field_name="loss_name")
        _validate_positive_number(self.learning_rate, field_name="learning_rate")
        _validate_positive_integer(
            self.optimization_steps,
            field_name="optimization_steps",
        )
        _validate_non_negative_number(
            self.drift_weight,
            field_name="drift_weight",
        )
        if (
            not isinstance(self.layer_reports, tuple)
            or not self.layer_reports
            or not all(
                isinstance(report, LayerGlobalRouterRecalibrationReport)
                for report in self.layer_reports
            )
        ):
            raise TypeError(
                "layer_reports must be a non-empty tuple of layer reports."
            )
        layer_indices = tuple(report.layer_index for report in self.layer_reports)
        if layer_indices != tuple(sorted(set(layer_indices))):
            raise ValueError(
                "layer_reports must have unique ascending layer indices."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible reproducibility metadata."""

        return {
            "compatibility_sha256": self.compatibility_sha256,
            "drift_weight": self.drift_weight,
            "feature_bank_metadata_sha256": self.feature_bank_metadata_sha256,
            "layer_reports": [
                report.to_dict() for report in self.layer_reports
            ],
            "learning_rate": self.learning_rate,
            "loss_name": self.loss_name,
            "objective": self.objective_name,
            "optimization_steps": self.optimization_steps,
            "optimizer": self.optimizer_name,
            "registry_sha256": self.registry_sha256,
            "strategy": self.strategy_name,
            "utility_label_metadata_sha256": (
                self.utility_label_metadata_sha256
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GlobalRouterRecalibrationReport:
        """Strictly reconstruct serialized RTR-004 report metadata."""

        expected = {
            "compatibility_sha256",
            "drift_weight",
            "feature_bank_metadata_sha256",
            "layer_reports",
            "learning_rate",
            "loss_name",
            "objective",
            "optimization_steps",
            "optimizer",
            "registry_sha256",
            "strategy",
            "utility_label_metadata_sha256",
        }
        _require_exact_fields(value, expected, field_name="recalibration report")
        if value["strategy"] != cls.strategy_name:
            raise ValueError("Unsupported global recalibration strategy.")
        if value["optimizer"] != cls.optimizer_name:
            raise ValueError("Unsupported global recalibration optimizer.")
        if value["objective"] != cls.objective_name:
            raise ValueError("Unsupported global recalibration objective.")
        raw_layers = value["layer_reports"]
        if not isinstance(raw_layers, list):
            raise TypeError("recalibration report layer_reports must be a list.")
        return cls(
            registry_sha256=value["registry_sha256"],
            compatibility_sha256=value["compatibility_sha256"],
            feature_bank_metadata_sha256=value[
                "feature_bank_metadata_sha256"
            ],
            utility_label_metadata_sha256=value[
                "utility_label_metadata_sha256"
            ],
            loss_name=value["loss_name"],
            learning_rate=value["learning_rate"],
            optimization_steps=value["optimization_steps"],
            drift_weight=value["drift_weight"],
            layer_reports=tuple(
                LayerGlobalRouterRecalibrationReport.from_dict(layer)
                for layer in raw_layers
            ),
        )


def recalibrate_global_router(
    source_router: LogicalAppendOnlyRouter,
    feature_bank: RouterFeatureBank,
    utility_labels: UtilityLabelBank,
    *,
    registry_sha256: str,
    compatibility_sha256: str,
    learning_rate: float = 5e-2,
    optimization_steps: int = 256,
    drift_weight: float = 1e-3,
) -> tuple[LogicalAppendOnlyRouter, GlobalRouterRecalibrationReport]:
    """Jointly recalibrate every specialist score against anchored route labels.

    For layer utilities ``U_l in R^(N x K_l)``, class zero represents the
    fixed-zero anchor and the deterministic target is

    ``z_l = argmax([0, U_l])``.

    The deployed score matrix ``G_l`` supplies specialist logits, so the
    optimized classification loss is

    ``CE([0, G_l], z_l) + lambda * ||theta_l - theta_l,0||^2 / |theta_l|``.

    All affine weights and biases at a layer are optimized jointly. Existing
    temperatures and thresholds remain fixed because they are part of the
    deployed score definition. Full-batch Adam runs on CPU in float64 without
    random sampling, and the best total-loss iterate (including the source
    initialization) is retained. The returned logical router is frozen and
    storage-independent.

    The utility bank must use ``incumbent_route="anchor"``. Otherwise a zero
    utility means equality with an arbitrary incumbent rather than equality
    with class zero, making the anchored target semantically invalid.

    Args:
        source_router: Frozen RTR-003 logical router to copy and recalibrate.
        feature_bank: Exact CAL-001 replay feature bank.
        utility_labels: Exact aligned CAL-002 individual utilities.
        registry_sha256: Active ordered registry identity for this router.
        compatibility_sha256: Immutable anchor/feature compatibility identity.
        learning_rate: Positive deterministic Adam learning rate.
        optimization_steps: Positive fixed full-batch step count.
        drift_weight: Non-negative proximal coefficient around source state.

    Returns:
        A new frozen logical router and immutable audit report.

    Raises:
        TypeError: If inputs or hyperparameters have invalid types.
        ValueError: If banks, layers, expert order, or hashes are incompatible.
        GlobalRouterRecalibrationError: If optimization produces invalid state.

    References:
        The repository architecture blueprint, "Mode B — Router-only
        recalibration."

    Differences from the blueprint:
        RTR-004 makes individual utility into an anchored best-route class and
        uses deterministic full-batch Adam with proximal drift. Pair utilities
        and top-2-specific objectives remain outside this initial strategy.
    """

    if not isinstance(source_router, LogicalAppendOnlyRouter):
        raise TypeError("source_router must be a LogicalAppendOnlyRouter.")
    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")
    if not isinstance(utility_labels, UtilityLabelBank):
        raise TypeError("utility_labels must be a UtilityLabelBank.")
    normalized_registry_hash = _validate_sha256(
        registry_sha256,
        field_name="registry_sha256",
    )
    normalized_compatibility_hash = _validate_sha256(
        compatibility_sha256,
        field_name="compatibility_sha256",
    )
    normalized_learning_rate = _validate_positive_number(
        learning_rate,
        field_name="learning_rate",
    )
    normalized_steps = _validate_positive_integer(
        optimization_steps,
        field_name="optimization_steps",
    )
    normalized_drift = _validate_non_negative_number(
        drift_weight,
        field_name="drift_weight",
    )

    utility_labels.assert_aligned(feature_bank)
    if utility_labels.incumbent_route != "anchor":
        raise ValueError(
            "utility_labels.incumbent_route must be 'anchor' for the "
            "fixed-zero anchored objective."
        )
    if source_router.feature_size != feature_bank.feature_size:
        raise ValueError(
            "source router feature size does not match the feature bank."
        )
    source_heads = source_router.to_independent_heads(trainable=False)
    if not source_heads.heads:
        raise ValueError("source_router must contain at least one expert head.")
    first_weight = source_heads.heads[0].linear.weight
    if first_weight.device.type != "cpu":
        raise ValueError(
            "RTR-004 requires a CPU source router for deterministic fitting."
        )
    if first_weight.dtype != feature_bank.dtype:
        raise ValueError(
            "source router dtype must match the feature-bank dtype: "
            f"{first_weight.dtype} != {feature_bank.dtype}."
        )
    source_layers = tuple(
        sorted({layer_index for layer_index, _ in source_heads.head_keys})
    )
    if source_layers != feature_bank.layer_indices:
        raise ValueError(
            "source router layers must exactly match feature-bank layers: "
            f"{source_layers} != {feature_bank.layer_indices}."
        )

    optimized_state: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}
    layer_reports: list[LayerGlobalRouterRecalibrationReport] = []
    for layer_index in source_layers:
        expected_expert_ids = source_router.expert_ids_for_layer(layer_index)
        if utility_labels.expert_ids_for_layer(layer_index) != expected_expert_ids:
            raise ValueError(
                "utility expert order does not match source router at layer "
                f"{layer_index}."
            )
        layer_heads = [
            head
            for head in source_heads.heads
            if head.layer_index == layer_index
        ]
        features = feature_bank.features_for_layer(layer_index).to(
            dtype=torch.float64,
            device="cpu",
        )
        utilities = utility_labels.utilities_for_layer(layer_index).to(
            dtype=torch.float64,
            device="cpu",
        )
        initial_weights = torch.cat(
            [head.linear.weight.detach().to(torch.float64) for head in layer_heads],
            dim=0,
        )
        initial_biases = torch.cat(
            [head.linear.bias.detach().to(torch.float64) for head in layer_heads],
            dim=0,
        )
        temperatures = torch.tensor(
            [head.temperature for head in layer_heads],
            dtype=torch.float64,
        )
        thresholds = torch.tensor(
            [head.threshold for head in layer_heads],
            dtype=torch.float64,
        )
        targets = torch.argmax(
            torch.cat(
                (
                    torch.zeros(
                        utilities.shape[0],
                        1,
                        dtype=torch.float64,
                    ),
                    utilities,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        weights, biases, report = _optimize_layer(
            features,
            targets,
            initial_weights,
            initial_biases,
            temperatures,
            thresholds,
            layer_index=layer_index,
            expert_ids=expected_expert_ids,
            learning_rate=normalized_learning_rate,
            optimization_steps=normalized_steps,
            drift_weight=normalized_drift,
            deployed_dtype=feature_bank.dtype,
        )
        for expert_column, expert_id in enumerate(expected_expert_ids):
            optimized_state[(layer_index, expert_id)] = (
                weights[expert_column : expert_column + 1],
                biases[expert_column : expert_column + 1],
            )
        layer_reports.append(report)

    output_heads = AppendOnlyRouterHeads(source_router.feature_size)
    for source_head in source_heads.heads:
        output_head = type(source_head)(
            source_router.feature_size,
            layer_index=source_head.layer_index,
            expert_id=source_head.expert_id,
            temperature=source_head.temperature,
            threshold=source_head.threshold,
            trainable=False,
        ).to(dtype=feature_bank.dtype, device="cpu")
        weight, bias = optimized_state[source_head.key]
        with torch.no_grad():
            output_head.linear.weight.copy_(weight)
            output_head.linear.bias.copy_(bias)
        output_heads.append(output_head)
    output_router = compile_logical_router(output_heads)
    output_router.requires_grad_(False)

    report = GlobalRouterRecalibrationReport(
        registry_sha256=normalized_registry_hash,
        compatibility_sha256=normalized_compatibility_hash,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        utility_label_metadata_sha256=utility_labels.metadata_sha256,
        loss_name=utility_labels.loss_name,
        learning_rate=normalized_learning_rate,
        optimization_steps=normalized_steps,
        drift_weight=normalized_drift,
        layer_reports=tuple(layer_reports),
    )
    return output_router, report


def _optimize_layer(
    features: Tensor,
    targets: Tensor,
    initial_weights: Tensor,
    initial_biases: Tensor,
    temperatures: Tensor,
    thresholds: Tensor,
    *,
    layer_index: int,
    expert_ids: tuple[str, ...],
    learning_rate: float,
    optimization_steps: int,
    drift_weight: float,
    deployed_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, LayerGlobalRouterRecalibrationReport]:
    """Optimize one complete layer and validate deployed-dtype improvement."""

    weights = initial_weights.clone().requires_grad_(True)
    biases = initial_biases.clone().requires_grad_(True)
    optimizer = torch.optim.Adam((weights, biases), lr=learning_rate)

    with torch.no_grad():
        initial_logits = _anchored_logits(
            features,
            initial_weights,
            initial_biases,
            temperatures,
            thresholds,
        )
        initial_ce = F.cross_entropy(initial_logits, targets)
        best_total = float(initial_ce.item())
        best_weights = initial_weights.clone()
        best_biases = initial_biases.clone()

    parameter_count = initial_weights.numel() + initial_biases.numel()
    for _ in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = _anchored_logits(
            features,
            weights,
            biases,
            temperatures,
            thresholds,
        )
        cross_entropy = F.cross_entropy(logits, targets)
        drift = (
            (weights - initial_weights).square().sum()
            + (biases - initial_biases).square().sum()
        ) / parameter_count
        total = cross_entropy + drift_weight * drift
        if not torch.isfinite(total).item():
            raise GlobalRouterRecalibrationError(
                f"Non-finite RTR-004 objective at layer {layer_index}."
            )
        total.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_logits = _anchored_logits(
                features,
                weights,
                biases,
                temperatures,
                thresholds,
            )
            candidate_ce = F.cross_entropy(candidate_logits, targets)
            candidate_drift = (
                (weights - initial_weights).square().sum()
                + (biases - initial_biases).square().sum()
            ) / parameter_count
            candidate_total = candidate_ce + drift_weight * candidate_drift
            candidate_value = float(candidate_total.item())
            if math.isfinite(candidate_value) and candidate_value < best_total:
                best_total = candidate_value
                best_weights = weights.detach().clone()
                best_biases = biases.detach().clone()

    # Metrics are intentionally recomputed after conversion to the deployed
    # feature dtype. If reduced precision loses the improvement, retain the
    # exact source state instead of publishing a worse router.
    deployed_features = features.to(dtype=deployed_dtype)
    deployed_temperatures = temperatures.to(dtype=deployed_dtype)
    deployed_thresholds = thresholds.to(dtype=deployed_dtype)
    deployed_initial_weights = initial_weights.to(dtype=deployed_dtype)
    deployed_initial_biases = initial_biases.to(dtype=deployed_dtype)
    deployed_weights = best_weights.to(dtype=deployed_dtype)
    deployed_biases = best_biases.to(dtype=deployed_dtype)
    initial_logits = _anchored_logits(
        deployed_features,
        deployed_initial_weights,
        deployed_initial_biases,
        deployed_temperatures,
        deployed_thresholds,
    )
    final_logits = _anchored_logits(
        deployed_features,
        deployed_weights,
        deployed_biases,
        deployed_temperatures,
        deployed_thresholds,
    )
    initial_cross_entropy = float(F.cross_entropy(initial_logits, targets).item())
    final_cross_entropy = float(F.cross_entropy(final_logits, targets).item())
    deployed_drift = float(
        (
            (deployed_weights - deployed_initial_weights).float().square().sum()
            + (deployed_biases - deployed_initial_biases)
            .float()
            .square()
            .sum()
        ).item()
        / parameter_count
    )
    final_total = final_cross_entropy + drift_weight * deployed_drift
    if not math.isfinite(final_total) or final_total > initial_cross_entropy:
        deployed_weights = deployed_initial_weights
        deployed_biases = deployed_initial_biases
        final_logits = initial_logits
        final_cross_entropy = initial_cross_entropy
        final_total = initial_cross_entropy

    initial_routes = torch.argmax(initial_logits, dim=-1)
    final_routes = torch.argmax(final_logits, dim=-1)
    report = LayerGlobalRouterRecalibrationReport(
        layer_index=layer_index,
        expert_ids=expert_ids,
        row_count=features.shape[0],
        anchor_target_count=int((targets == 0).sum().item()),
        initial_cross_entropy=initial_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        initial_total_loss=initial_cross_entropy,
        final_total_loss=final_total,
        initial_correct_count=int((initial_routes == targets).sum().item()),
        final_correct_count=int((final_routes == targets).sum().item()),
        changed_route_count=int((initial_routes != final_routes).sum().item()),
    )
    return deployed_weights, deployed_biases, report


def _anchored_logits(
    features: Tensor,
    weights: Tensor,
    biases: Tensor,
    temperatures: Tensor,
    thresholds: Tensor,
) -> Tensor:
    """Return ``[anchor, specialists]`` logits with fixed-zero class zero."""

    specialist_scores = (
        F.linear(features, weights, biases) / temperatures - thresholds
    )
    return torch.cat(
        (
            torch.zeros(
                *specialist_scores.shape[:-1],
                1,
                dtype=specialist_scores.dtype,
                device=specialist_scores.device,
            ),
            specialist_scores,
        ),
        dim=-1,
    )


def _require_exact_fields(
    value: Any,
    expected: set[str],
    *,
    field_name: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field_name} fields do not match the schema: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _validate_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _validate_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace."
        )
    return value


def _validate_expert_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("expert_ids must be a non-empty tuple.")
    normalized = tuple(
        _validate_identifier(expert_id, field_name="expert_id")
        for expert_id in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must not contain duplicates.")
    return normalized


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _validate_non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_count(value: Any, *, row_count: int, field_name: str) -> int:
    count = _validate_non_negative_integer(value, field_name=field_name)
    if count > row_count:
        raise ValueError(f"{field_name} cannot exceed row_count.")
    return count


def _validate_positive_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _validate_non_negative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized
