"""Deterministic hard-route distillation into a frozen logical router.

RTR-007 learns an affine deployment router from the anchored hard assignments
of an immutable teacher score bank. The teacher, cached features, experts, and
model are never mutated. Full-batch CPU float64 optimization makes fitting
repeatable, while the published router uses the feature bank's deployed dtype
and contains buffers rather than trainable parameters.

References:
    Dai et al., "StableMoE: Stable Routing Strategy for Mixture of Experts."
    https://aclanthology.org/2022.acl-long.489/

Differences from the paper:
    StableMoE distills a jointly learned balanced routing strategy and freezes
    assignments during a later expert-training stage. RTR-007 instead distills
    the fixed-zero-anchor top-1 decisions of any frozen ``RouterScoreBank``
    from an offline CAL-001 feature bank. It does not train experts, reproduce
    StableMoE's load-balancing objective, or implement top-2 distillation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from anchored_moe.routers.base import RouterScoreBank
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads
from anchored_moe.routers.logical import (
    LogicalAppendOnlyRouter,
    compile_logical_router,
)
from anchored_moe.utils.hashing import sha256_json

if TYPE_CHECKING:
    from anchored_moe.calibration.features import RouterFeatureBank


class RouterDistillationError(ValueError):
    """Raised when RTR-007 cannot produce a finite frozen student router."""


@dataclass(frozen=True)
class HardRouteDistillationConfig:
    """Public configuration for deterministic anchored hard-route distillation.

    Args:
        learning_rate: Positive learning rate for full-batch Adam.
        optimization_steps: Positive fixed number of optimizer steps.
        l2_weight: Non-negative normalized penalty on student coefficients.
    """

    strategy: ClassVar[str] = "hard_route_distillation"
    optimizer: ClassVar[str] = "full_batch_adam_cpu_float64"
    objective: ClassVar[str] = "anchored_teacher_route_cross_entropy"

    learning_rate: float = 5e-2
    optimization_steps: int = 256
    l2_weight: float = 1e-4

    def __post_init__(self) -> None:
        """Validate deterministic optimizer hyperparameters."""

        object.__setattr__(
            self,
            "learning_rate",
            _validate_positive_number(
                self.learning_rate,
                field_name="learning_rate",
            ),
        )
        _validate_positive_integer(
            self.optimization_steps,
            field_name="optimization_steps",
        )
        object.__setattr__(
            self,
            "l2_weight",
            _validate_non_negative_number(
                self.l2_weight,
                field_name="l2_weight",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible public configuration."""

        return {
            "l2_weight": self.l2_weight,
            "learning_rate": self.learning_rate,
            "objective": self.objective,
            "optimization_steps": self.optimization_steps,
            "optimizer": self.optimizer,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HardRouteDistillationConfig:
        """Strictly reconstruct a serialized public configuration."""

        expected = {
            "l2_weight",
            "learning_rate",
            "objective",
            "optimization_steps",
            "optimizer",
            "strategy",
        }
        _require_exact_fields(value, expected, field_name="distillation config")
        if value["strategy"] != cls.strategy:
            raise ValueError("Unsupported router distillation strategy.")
        if value["optimizer"] != cls.optimizer:
            raise ValueError("Unsupported router distillation optimizer.")
        if value["objective"] != cls.objective:
            raise ValueError("Unsupported router distillation objective.")
        return cls(
            learning_rate=value["learning_rate"],
            optimization_steps=value["optimization_steps"],
            l2_weight=value["l2_weight"],
        )


@dataclass(frozen=True)
class LayerRouterDistillationReport:
    """Teacher-target and student-agreement metrics for one decoder layer."""

    layer_index: int
    expert_ids: tuple[str, ...]
    row_count: int
    target_counts: tuple[int, ...]
    teacher_scores_sha256: str
    teacher_routes_sha256: str
    initial_cross_entropy: float
    final_cross_entropy: float
    final_total_loss: float
    initial_agreement_count: int
    final_agreement_count: int

    def __post_init__(self) -> None:
        """Validate immutable layer metrics and anchored class counts."""

        _validate_non_negative_integer(self.layer_index, field_name="layer_index")
        _validate_expert_ids(self.expert_ids)
        _validate_positive_integer(self.row_count, field_name="row_count")
        if (
            not isinstance(self.target_counts, tuple)
            or len(self.target_counts) != len(self.expert_ids) + 1
        ):
            raise ValueError(
                "target_counts must contain anchor plus one count per expert."
            )
        for count in self.target_counts:
            _validate_count(count, row_count=self.row_count, field_name="target_count")
        if sum(self.target_counts) != self.row_count:
            raise ValueError("target_counts must sum to row_count.")
        _validate_sha256(
            self.teacher_scores_sha256,
            field_name="teacher_scores_sha256",
        )
        _validate_sha256(
            self.teacher_routes_sha256,
            field_name="teacher_routes_sha256",
        )
        for field_name in (
            "initial_cross_entropy",
            "final_cross_entropy",
            "final_total_loss",
        ):
            _validate_non_negative_number(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in (
            "initial_agreement_count",
            "final_agreement_count",
        ):
            _validate_count(
                getattr(self, field_name),
                row_count=self.row_count,
                field_name=field_name,
            )

    @property
    def anchor_target_count(self) -> int:
        """Return the number of teacher assignments selecting the anchor."""

        return self.target_counts[0]

    @property
    def final_route_agreement(self) -> float:
        """Return final hard-route agreement as a fraction in ``[0, 1]``."""

        return self.final_agreement_count / self.row_count

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible layer-level audit metadata."""

        return {
            "expert_ids": list(self.expert_ids),
            "final_agreement_count": self.final_agreement_count,
            "final_cross_entropy": self.final_cross_entropy,
            "final_total_loss": self.final_total_loss,
            "initial_agreement_count": self.initial_agreement_count,
            "initial_cross_entropy": self.initial_cross_entropy,
            "layer_index": self.layer_index,
            "row_count": self.row_count,
            "target_counts": list(self.target_counts),
            "teacher_routes_sha256": self.teacher_routes_sha256,
            "teacher_scores_sha256": self.teacher_scores_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> LayerRouterDistillationReport:
        """Strictly reconstruct one serialized layer report."""

        expected = {
            "expert_ids",
            "final_agreement_count",
            "final_cross_entropy",
            "final_total_loss",
            "initial_agreement_count",
            "initial_cross_entropy",
            "layer_index",
            "row_count",
            "target_counts",
            "teacher_routes_sha256",
            "teacher_scores_sha256",
        }
        _require_exact_fields(value, expected, field_name="layer report")
        if not isinstance(value["expert_ids"], list):
            raise TypeError("layer report expert_ids must be a list.")
        if not isinstance(value["target_counts"], list):
            raise TypeError("layer report target_counts must be a list.")
        return cls(
            layer_index=value["layer_index"],
            expert_ids=tuple(value["expert_ids"]),
            row_count=value["row_count"],
            target_counts=tuple(value["target_counts"]),
            teacher_scores_sha256=value["teacher_scores_sha256"],
            teacher_routes_sha256=value["teacher_routes_sha256"],
            initial_cross_entropy=value["initial_cross_entropy"],
            final_cross_entropy=value["final_cross_entropy"],
            final_total_loss=value["final_total_loss"],
            initial_agreement_count=value["initial_agreement_count"],
            final_agreement_count=value["final_agreement_count"],
        )


@dataclass(frozen=True)
class RouterDistillationReport:
    """Complete reproducibility record for one frozen distilled router."""

    strategy_name: ClassVar[str] = HardRouteDistillationConfig.strategy
    teacher_policy: ClassVar[str] = "fixed_zero_anchor_argmax"

    teacher_architecture: str
    feature_bank_metadata_sha256: str
    teacher_assignments_sha256: str
    config: HardRouteDistillationConfig
    layer_reports: tuple[LayerRouterDistillationReport, ...]

    def __post_init__(self) -> None:
        """Validate report identity, configuration, and layer ordering."""

        _validate_identifier(
            self.teacher_architecture,
            field_name="teacher_architecture",
        )
        _validate_sha256(
            self.feature_bank_metadata_sha256,
            field_name="feature_bank_metadata_sha256",
        )
        _validate_sha256(
            self.teacher_assignments_sha256,
            field_name="teacher_assignments_sha256",
        )
        if not isinstance(self.config, HardRouteDistillationConfig):
            raise TypeError("config must be a HardRouteDistillationConfig.")
        if (
            not isinstance(self.layer_reports, tuple)
            or not self.layer_reports
            or not all(
                isinstance(report, LayerRouterDistillationReport)
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
        row_counts = {report.row_count for report in self.layer_reports}
        if len(row_counts) != 1:
            raise ValueError(
                "layer_reports must share one feature-bank row count."
            )

    @property
    def row_count(self) -> int:
        """Return the common number of feature rows distilled at each layer."""

        return self.layer_reports[0].row_count

    @property
    def final_route_agreement(self) -> float:
        """Return micro-averaged agreement over all layer-row decisions."""

        correct = sum(report.final_agreement_count for report in self.layer_reports)
        total = sum(report.row_count for report in self.layer_reports)
        return correct / total

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible distillation metadata."""

        return {
            "config": self.config.to_dict(),
            "feature_bank_metadata_sha256": self.feature_bank_metadata_sha256,
            "layer_reports": [report.to_dict() for report in self.layer_reports],
            "strategy": self.strategy_name,
            "teacher_architecture": self.teacher_architecture,
            "teacher_assignments_sha256": self.teacher_assignments_sha256,
            "teacher_policy": self.teacher_policy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RouterDistillationReport:
        """Strictly reconstruct serialized RTR-007 report metadata."""

        expected = {
            "config",
            "feature_bank_metadata_sha256",
            "layer_reports",
            "strategy",
            "teacher_architecture",
            "teacher_assignments_sha256",
            "teacher_policy",
        }
        _require_exact_fields(value, expected, field_name="distillation report")
        if value["strategy"] != cls.strategy_name:
            raise ValueError("Unsupported router distillation report strategy.")
        if value["teacher_policy"] != cls.teacher_policy:
            raise ValueError("Unsupported router distillation teacher policy.")
        if not isinstance(value["config"], dict):
            raise TypeError("distillation report config must be a mapping.")
        if not isinstance(value["layer_reports"], list):
            raise TypeError("distillation report layer_reports must be a list.")
        return cls(
            teacher_architecture=value["teacher_architecture"],
            feature_bank_metadata_sha256=value[
                "feature_bank_metadata_sha256"
            ],
            teacher_assignments_sha256=value["teacher_assignments_sha256"],
            config=HardRouteDistillationConfig.from_dict(value["config"]),
            layer_reports=tuple(
                LayerRouterDistillationReport.from_dict(report)
                for report in value["layer_reports"]
            ),
        )


def distill_frozen_router(
    teacher_router: RouterScoreBank,
    feature_bank: RouterFeatureBank,
    *,
    config: HardRouteDistillationConfig | None = None,
) -> tuple[LogicalAppendOnlyRouter, RouterDistillationReport]:
    """Distill anchored hard assignments into an affine frozen score bank.

    For teacher specialist scores ``G_l^teacher in R^(N x K_l)``, the target
    class includes the permanent anchor at index zero:

    ``z_l = argmax([0, G_l^teacher])``.

    Ties choose the anchor first, then the earliest expert in the teacher's
    stable layer-local order. The student minimizes

    ``CE([0, X_l W_l^T + b_l], z_l) + lambda ||(W_l, b_l)||^2 / P_l``

    independently at each layer. Zero initialization, full-batch CPU float64
    Adam, and best-iterate retention avoid random state. The accepted state is
    checked again in the feature bank's dtype and falls back to the zero
    student if reduced precision makes its total objective worse.

    Args:
        teacher_router:
            Frozen evaluation-mode score bank. It may be more expressive than
            the affine student but must produce finite CPU scores aligned with
            the feature bank.
        feature_bank:
            Immutable CAL-001 pre-MLP feature bank whose layer set defines the
            distillation scope.
        config:
            Optional validated optimizer configuration.

    Returns:
        A new buffer-only logical router and immutable audit report.

    Raises:
        TypeError: If public inputs do not implement the declared contracts.
        ValueError: If feature, layer, expert, dtype, device, or teacher state
            is incompatible.
        RouterDistillationError: If fitting produces a non-finite objective.

    References:
        Dai et al., "StableMoE: Stable Routing Strategy for Mixture of
        Experts." https://aclanthology.org/2022.acl-long.489/

    Differences from the paper:
        This function uses offline anchor-aware hard labels and does not jointly
        train a balanced teacher, experts, or the transformer backbone.
    """

    # Local import avoids making the routers package depend on calibration's
    # package-level exports during import initialization.
    from anchored_moe.calibration.features import RouterFeatureBank

    if not isinstance(teacher_router, RouterScoreBank):
        raise TypeError("teacher_router must implement RouterScoreBank.")
    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")
    if config is None:
        resolved_config = HardRouteDistillationConfig()
    elif isinstance(config, HardRouteDistillationConfig):
        resolved_config = config
    else:
        raise TypeError("config must be a HardRouteDistillationConfig or None.")
    if teacher_router.feature_size != feature_bank.feature_size:
        raise ValueError(
            "teacher router feature size does not match the feature bank."
        )
    if teacher_router.training:
        raise ValueError("teacher_router must be in evaluation mode.")
    if any(parameter.requires_grad for parameter in teacher_router.parameters()):
        raise ValueError("teacher_router parameters must be frozen.")
    for name, value in teacher_router.state_dict().items():
        if value.device.type != "cpu":
            raise ValueError(
                f"teacher_router state {name!r} must reside on CPU."
            )

    teacher_architecture = getattr(
        teacher_router,
        "architecture",
        type(teacher_router).__name__,
    )
    _validate_identifier(
        teacher_architecture,
        field_name="teacher_architecture",
    )

    fitted_state: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}
    layer_reports: list[LayerRouterDistillationReport] = []
    assignment_identities: list[dict[str, Any]] = []
    for layer_index in feature_bank.layer_indices:
        expert_ids = _validate_expert_ids(
            teacher_router.expert_ids_for_layer(layer_index)
        )
        features = feature_bank.features_for_layer(layer_index)
        with torch.no_grad():
            teacher_scores = teacher_router(
                features,
                layer_index=layer_index,
            )
        _validate_teacher_scores(
            teacher_scores,
            row_count=feature_bank.row_count,
            expert_count=len(expert_ids),
            dtype=feature_bank.dtype,
            layer_index=layer_index,
        )
        targets = torch.argmax(
            torch.cat(
                (
                    torch.zeros(
                        feature_bank.row_count,
                        1,
                        dtype=teacher_scores.dtype,
                    ),
                    teacher_scores,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        weights, biases, report = _fit_layer(
            features,
            teacher_scores,
            targets,
            layer_index=layer_index,
            expert_ids=expert_ids,
            config=resolved_config,
        )
        for expert_column, expert_id in enumerate(expert_ids):
            fitted_state[(layer_index, expert_id)] = (
                weights[expert_column : expert_column + 1],
                biases[expert_column : expert_column + 1],
            )
        layer_reports.append(report)
        assignment_identities.append(
            {
                "expert_ids": list(expert_ids),
                "layer_index": layer_index,
                "teacher_routes_sha256": report.teacher_routes_sha256,
            }
        )

    output_heads = AppendOnlyRouterHeads(feature_bank.feature_size)
    for report in layer_reports:
        for expert_id in report.expert_ids:
            output_head = AffineRouterHead(
                feature_bank.feature_size,
                layer_index=report.layer_index,
                expert_id=expert_id,
                temperature=1.0,
                threshold=0.0,
                trainable=False,
            ).to(device="cpu", dtype=feature_bank.dtype)
            weight, bias = fitted_state[(report.layer_index, expert_id)]
            with torch.no_grad():
                output_head.linear.weight.copy_(weight)
                output_head.linear.bias.copy_(bias)
            output_heads.append(output_head)

    output_router = compile_logical_router(output_heads)
    output_router.requires_grad_(False)
    output_router.eval()
    report = RouterDistillationReport(
        teacher_architecture=teacher_architecture,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        teacher_assignments_sha256=sha256_json(assignment_identities),
        config=resolved_config,
        layer_reports=tuple(layer_reports),
    )
    return output_router, report


def _fit_layer(
    features: Tensor,
    teacher_scores: Tensor,
    targets: Tensor,
    *,
    layer_index: int,
    expert_ids: tuple[str, ...],
    config: HardRouteDistillationConfig,
) -> tuple[Tensor, Tensor, LayerRouterDistillationReport]:
    """Fit one affine anchored classifier and audit deployed-dtype behavior."""

    training_features = features.to(device="cpu", dtype=torch.float64)
    weights = torch.zeros(
        len(expert_ids),
        features.shape[1],
        dtype=torch.float64,
        requires_grad=True,
    )
    biases = torch.zeros(
        len(expert_ids),
        dtype=torch.float64,
        requires_grad=True,
    )
    initial_weights = weights.detach().clone()
    initial_biases = biases.detach().clone()
    optimizer = torch.optim.Adam((weights, biases), lr=config.learning_rate)

    with torch.no_grad():
        initial_logits = _anchored_logits(
            training_features,
            initial_weights,
            initial_biases,
        )
        initial_ce = F.cross_entropy(initial_logits, targets)
        best_total = float(initial_ce.item())
        best_weights = initial_weights.clone()
        best_biases = initial_biases.clone()

    parameter_count = weights.numel() + biases.numel()
    for _ in range(config.optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = _anchored_logits(training_features, weights, biases)
        cross_entropy = F.cross_entropy(logits, targets)
        normalized_l2 = (
            weights.square().sum() + biases.square().sum()
        ) / parameter_count
        total = cross_entropy + config.l2_weight * normalized_l2
        if not torch.isfinite(total).item():
            raise RouterDistillationError(
                f"Non-finite RTR-007 objective at layer {layer_index}."
            )
        total.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_logits = _anchored_logits(
                training_features,
                weights,
                biases,
            )
            candidate_ce = F.cross_entropy(candidate_logits, targets)
            candidate_l2 = (
                weights.square().sum() + biases.square().sum()
            ) / parameter_count
            candidate_total = candidate_ce + config.l2_weight * candidate_l2
            candidate_value = float(candidate_total.item())
            if math.isfinite(candidate_value) and candidate_value < best_total:
                best_total = candidate_value
                best_weights = weights.detach().clone()
                best_biases = biases.detach().clone()

    deployed_weights = best_weights.to(dtype=features.dtype)
    deployed_biases = best_biases.to(dtype=features.dtype)
    zero_weights = initial_weights.to(dtype=features.dtype)
    zero_biases = initial_biases.to(dtype=features.dtype)
    initial_logits = _anchored_logits(features, zero_weights, zero_biases)
    final_logits = _anchored_logits(
        features,
        deployed_weights,
        deployed_biases,
    )
    initial_cross_entropy = float(F.cross_entropy(initial_logits, targets).item())
    final_cross_entropy = float(F.cross_entropy(final_logits, targets).item())
    deployed_l2 = float(
        (
            deployed_weights.float().square().sum()
            + deployed_biases.float().square().sum()
        ).item()
        / parameter_count
    )
    final_total = final_cross_entropy + config.l2_weight * deployed_l2
    if not math.isfinite(final_total) or final_total > initial_cross_entropy:
        deployed_weights = zero_weights
        deployed_biases = zero_biases
        final_logits = initial_logits
        final_cross_entropy = initial_cross_entropy
        final_total = initial_cross_entropy

    initial_routes = torch.argmax(initial_logits, dim=-1)
    final_routes = torch.argmax(final_logits, dim=-1)
    target_counts = tuple(
        int((targets == class_index).sum().item())
        for class_index in range(len(expert_ids) + 1)
    )
    report = LayerRouterDistillationReport(
        layer_index=layer_index,
        expert_ids=expert_ids,
        row_count=features.shape[0],
        target_counts=target_counts,
        teacher_scores_sha256=_sha256_tensor(teacher_scores),
        teacher_routes_sha256=_sha256_tensor(targets),
        initial_cross_entropy=initial_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        final_total_loss=final_total,
        initial_agreement_count=int((initial_routes == targets).sum().item()),
        final_agreement_count=int((final_routes == targets).sum().item()),
    )
    return deployed_weights, deployed_biases, report


def _anchored_logits(features: Tensor, weights: Tensor, biases: Tensor) -> Tensor:
    """Return fixed-zero anchor plus affine specialist logits."""

    specialist_scores = F.linear(features, weights, biases)
    return torch.cat(
        (
            torch.zeros(
                specialist_scores.shape[0],
                1,
                dtype=specialist_scores.dtype,
                device=specialist_scores.device,
            ),
            specialist_scores,
        ),
        dim=-1,
    )


def _validate_teacher_scores(
    scores: Tensor,
    *,
    row_count: int,
    expert_count: int,
    dtype: torch.dtype,
    layer_index: int,
) -> None:
    if not isinstance(scores, Tensor) or not scores.is_floating_point():
        raise TypeError(
            f"teacher scores at layer {layer_index} must be floating-point."
        )
    if scores.shape != (row_count, expert_count):
        raise ValueError(
            f"teacher scores at layer {layer_index} must have shape "
            f"{(row_count, expert_count)}, got {tuple(scores.shape)}."
        )
    if scores.device.type != "cpu":
        raise ValueError(
            f"teacher scores at layer {layer_index} must reside on CPU."
        )
    if scores.dtype != dtype:
        raise ValueError(
            f"teacher scores at layer {layer_index} must match feature dtype: "
            f"{scores.dtype} != {dtype}."
        )
    if not torch.isfinite(scores).all().item():
        raise ValueError(
            f"teacher scores at layer {layer_index} must be finite."
        )


def _sha256_tensor(value: Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    portable = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(portable.dtype).encode("ascii"))
    digest.update(str(tuple(portable.shape)).encode("ascii"))
    # Byte views support every PyTorch dtype, including bfloat16, without
    # relying on NumPy to understand the tensor's scalar type.
    digest.update(portable.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


def _validate_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


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
