"""Deterministic fitting for COM-003 learned pair-mixture weights."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F

from anchored_moe.calibration.pairs import PairSamplingPlan

if TYPE_CHECKING:
    from anchored_moe.policies.learned_combination import LearnedMixtureWeightHead


class MixtureWeightFittingError(ValueError):
    """Raised when COM-003 fitting cannot produce a finite frozen head."""


@dataclass(frozen=True)
class MixtureWeightFittingConfig:
    """Configuration for deterministic soft-oracle mixture-weight fitting.

    Args:
        learning_rate: Positive learning rate for full-batch Adam.
        optimization_steps: Positive fixed number of optimizer steps.
        l2_weight: Non-negative normalized penalty on the two affine weights.
    """

    strategy: ClassVar[str] = "learned_pair_weight_affine"
    optimizer: ClassVar[str] = "full_batch_adam_cpu_float64"
    objective: ClassVar[str] = "soft_target_binary_cross_entropy"

    learning_rate: float = 5e-2
    optimization_steps: int = 256
    l2_weight: float = 1e-4

    def __post_init__(self) -> None:
        """Validate public deterministic optimizer settings."""

        object.__setattr__(
            self,
            "learning_rate",
            _positive_number(self.learning_rate, field_name="learning_rate"),
        )
        if (
            isinstance(self.optimization_steps, bool)
            or not isinstance(self.optimization_steps, int)
            or self.optimization_steps <= 0
        ):
            raise ValueError("optimization_steps must be a positive integer.")
        object.__setattr__(
            self,
            "l2_weight",
            _non_negative_number(self.l2_weight, field_name="l2_weight"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible configuration metadata."""

        return {
            "l2_weight": self.l2_weight,
            "learning_rate": self.learning_rate,
            "objective": self.objective,
            "optimization_steps": self.optimization_steps,
            "optimizer": self.optimizer,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MixtureWeightFittingConfig:
        """Strictly reconstruct serialized fitting configuration."""

        expected = {
            "l2_weight",
            "learning_rate",
            "objective",
            "optimization_steps",
            "optimizer",
            "strategy",
        }
        _require_exact_fields(value, expected, field_name="fitting config")
        if value["strategy"] != cls.strategy:
            raise ValueError("Unsupported mixture-weight fitting strategy.")
        if value["optimizer"] != cls.optimizer:
            raise ValueError("Unsupported mixture-weight fitting optimizer.")
        if value["objective"] != cls.objective:
            raise ValueError("Unsupported mixture-weight fitting objective.")
        return cls(
            learning_rate=value["learning_rate"],
            optimization_steps=value["optimization_steps"],
            l2_weight=value["l2_weight"],
        )


@dataclass(frozen=True)
class MixtureWeightFittingReport:
    """Auditable inputs and metrics for one learned mixture head."""

    strategy_name: ClassVar[str] = MixtureWeightFittingConfig.strategy

    config: MixtureWeightFittingConfig
    row_count: int
    pair_counts: tuple[int, ...]
    selected_scores_sha256: str
    target_weights_sha256: str
    pair_indices_sha256: str
    initial_cross_entropy: float
    final_cross_entropy: float
    final_total_loss: float
    initial_mean_absolute_error: float
    final_mean_absolute_error: float

    def __post_init__(self) -> None:
        """Validate report counts, digests, and finite metrics."""

        if not isinstance(self.config, MixtureWeightFittingConfig):
            raise TypeError("config must be a MixtureWeightFittingConfig.")
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer.")
        if (
            not isinstance(self.pair_counts, tuple)
            or not self.pair_counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for count in self.pair_counts
            )
        ):
            raise ValueError(
                "pair_counts must be a non-empty tuple of non-negative integers."
            )
        if sum(self.pair_counts) != self.row_count:
            raise ValueError("pair_counts must sum to row_count.")
        for field_name in (
            "selected_scores_sha256",
            "target_weights_sha256",
            "pair_indices_sha256",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
        for field_name in (
            "initial_cross_entropy",
            "final_cross_entropy",
            "final_total_loss",
            "initial_mean_absolute_error",
            "final_mean_absolute_error",
        ):
            _non_negative_number(getattr(self, field_name), field_name=field_name)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fitting provenance and metrics."""

        return {
            "config": self.config.to_dict(),
            "final_cross_entropy": self.final_cross_entropy,
            "final_mean_absolute_error": self.final_mean_absolute_error,
            "final_total_loss": self.final_total_loss,
            "initial_cross_entropy": self.initial_cross_entropy,
            "initial_mean_absolute_error": self.initial_mean_absolute_error,
            "pair_counts": list(self.pair_counts),
            "pair_indices_sha256": self.pair_indices_sha256,
            "row_count": self.row_count,
            "selected_scores_sha256": self.selected_scores_sha256,
            "target_weights_sha256": self.target_weights_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MixtureWeightFittingReport:
        """Strictly reconstruct one serialized fitting report."""

        expected = {
            "config",
            "final_cross_entropy",
            "final_mean_absolute_error",
            "final_total_loss",
            "initial_cross_entropy",
            "initial_mean_absolute_error",
            "pair_counts",
            "pair_indices_sha256",
            "row_count",
            "selected_scores_sha256",
            "target_weights_sha256",
        }
        _require_exact_fields(value, expected, field_name="fitting report")
        if not isinstance(value["config"], dict):
            raise TypeError("fitting report config must be a mapping.")
        if not isinstance(value["pair_counts"], list):
            raise TypeError("fitting report pair_counts must be a list.")
        return cls(
            config=MixtureWeightFittingConfig.from_dict(value["config"]),
            row_count=value["row_count"],
            pair_counts=tuple(value["pair_counts"]),
            selected_scores_sha256=value["selected_scores_sha256"],
            target_weights_sha256=value["target_weights_sha256"],
            pair_indices_sha256=value["pair_indices_sha256"],
            initial_cross_entropy=value["initial_cross_entropy"],
            final_cross_entropy=value["final_cross_entropy"],
            final_total_loss=value["final_total_loss"],
            initial_mean_absolute_error=value["initial_mean_absolute_error"],
            final_mean_absolute_error=value["final_mean_absolute_error"],
        )


def fit_learned_mixture_head(
    plan: PairSamplingPlan,
    *,
    selected_scores: Tensor,
    target_weights: Tensor,
    pair_indices: Tensor,
    config: MixtureWeightFittingConfig | None = None,
) -> tuple[LearnedMixtureWeightHead, MixtureWeightFittingReport]:
    """Fit a frozen affine pair-weight head from explicit oracle weights.

    Args:
        plan:
            CAL-003 pair plan defining stable pair identities.
        selected_scores:
            Already ordered selection scores with shape ``(N, 2)``.
        target_weights:
            Oracle mixture weights with shape ``(N, 2)`` in the same pair
            order. Values must be non-negative and sum to one.
        pair_indices:
            Integer tensor with shape ``(N,)`` indexing ``plan.pairs``.
        config:
            Optional deterministic fitting configuration.

    Returns:
        A storage-independent CPU head in the input score dtype and an
        immutable fitting report.

    COM-003 deliberately accepts measured oracle weights rather than deriving
    them from CAL-003 pair utility. One loss value per pair cannot identify an
    optimal interpolation coefficient. Counterfactual model execution and
    weight-grid sweeps remain caller-owned.
    """

    if not isinstance(plan, PairSamplingPlan):
        raise TypeError("plan must be a PairSamplingPlan.")
    normalized_config = config or MixtureWeightFittingConfig()
    if not isinstance(normalized_config, MixtureWeightFittingConfig):
        raise TypeError("config must be a MixtureWeightFittingConfig.")
    scores, weights, indices = _normalize_training_data(
        selected_scores,
        target_weights=target_weights,
        pair_indices=pair_indices,
        pair_count=plan.pair_count,
    )
    target_first = weights[:, 0]

    coefficient = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(
        (coefficient, bias),
        lr=normalized_config.learning_rate,
    )

    initial_cross_entropy = _cross_entropy(
        scores @ coefficient + bias,
        target_first,
    )
    initial_mae = _mean_absolute_error(
        scores @ coefficient + bias,
        target_first,
    )
    best_key = (
        initial_cross_entropy,
        initial_cross_entropy,
        0,
    )
    best_weight = coefficient.detach().clone()
    best_bias = bias.detach().clone()

    for step in range(1, normalized_config.optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = scores @ coefficient + bias
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits,
            target_first,
            reduction="mean",
        )
        # Normalize by the two learned score coefficients. Bias is left
        # unregularized so a globally asymmetric oracle remains representable.
        penalty = coefficient.square().mean()
        total_loss = cross_entropy + normalized_config.l2_weight * penalty
        if not torch.isfinite(total_loss).item():
            raise MixtureWeightFittingError(
                "Mixture-weight optimization produced a non-finite loss."
            )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_logits = scores @ coefficient + bias
            candidate_cross_entropy = _cross_entropy(
                candidate_logits,
                target_first,
            )
            candidate_total = candidate_cross_entropy + (
                normalized_config.l2_weight
                * float(coefficient.square().mean().item())
            )
            candidate_key = (
                candidate_total,
                candidate_cross_entropy,
                step,
            )
            if candidate_key < best_key:
                best_key = candidate_key
                best_weight = coefficient.detach().clone()
                best_bias = bias.detach().clone()

    output_dtype = selected_scores.dtype
    deployed_scores = selected_scores.detach().to(device="cpu").contiguous()
    deployed_weight = best_weight.to(dtype=output_dtype)
    deployed_bias = best_bias.to(dtype=output_dtype)
    deployed_logits = (
        torch.sum(deployed_scores * deployed_weight, dim=-1) + deployed_bias
    ).to(dtype=torch.float64)
    final_cross_entropy = _cross_entropy(deployed_logits, target_first)
    final_mae = _mean_absolute_error(deployed_logits, target_first)
    final_total = final_cross_entropy + (
        normalized_config.l2_weight
        * float(deployed_weight.to(dtype=torch.float64).square().mean().item())
    )
    if final_total > initial_cross_entropy + 1e-12:
        # Reduced-precision publication must preserve the same acceptance gate
        # as the float64 fit. The neutral head is exactly representable in all
        # supported floating dtypes and remains the conservative fallback.
        deployed_weight = torch.zeros(2, dtype=output_dtype)
        deployed_bias = torch.zeros((), dtype=output_dtype)
        deployed_logits = torch.zeros(scores.shape[0], dtype=torch.float64)
        final_cross_entropy = initial_cross_entropy
        final_mae = initial_mae
        final_total = initial_cross_entropy

    # Import only when fitting is invoked. Importing the policy package while
    # calibration's public namespace is still initializing would traverse the
    # registry export namespace back into calibration.
    from anchored_moe.policies.learned_combination import LearnedMixtureWeightHead

    head = LearnedMixtureWeightHead(
        weight=deployed_weight,
        bias=deployed_bias,
    ).eval()
    pair_counts = tuple(
        int((indices == pair_index).sum().item())
        for pair_index in range(plan.pair_count)
    )
    report = MixtureWeightFittingReport(
        config=normalized_config,
        row_count=scores.shape[0],
        pair_counts=pair_counts,
        selected_scores_sha256=_sha256_tensor(scores),
        target_weights_sha256=_sha256_tensor(weights),
        pair_indices_sha256=_sha256_tensor(indices),
        initial_cross_entropy=initial_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        final_total_loss=final_total,
        initial_mean_absolute_error=initial_mae,
        final_mean_absolute_error=final_mae,
    )
    return head, report


def _normalize_training_data(
    selected_scores: Tensor,
    *,
    target_weights: Tensor,
    pair_indices: Tensor,
    pair_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate alignment and copy fitting data to deterministic CPU storage."""

    if not isinstance(selected_scores, Tensor):
        raise TypeError("selected_scores must be a torch.Tensor.")
    if not selected_scores.is_floating_point():
        raise TypeError("selected_scores must use a floating-point dtype.")
    if selected_scores.ndim != 2 or selected_scores.shape[1] != 2:
        raise ValueError("selected_scores must have shape (N, 2).")
    if selected_scores.shape[0] == 0:
        raise ValueError("selected_scores must contain at least one row.")
    if not isinstance(target_weights, Tensor):
        raise TypeError("target_weights must be a torch.Tensor.")
    if not target_weights.is_floating_point():
        raise TypeError("target_weights must use a floating-point dtype.")
    if target_weights.shape != selected_scores.shape:
        raise ValueError("target_weights must match selected_scores shape.")
    if target_weights.dtype != selected_scores.dtype:
        raise ValueError("target_weights and selected_scores must share one dtype.")
    if target_weights.device != selected_scores.device:
        raise ValueError("target_weights and selected_scores must share one device.")
    if not isinstance(pair_indices, Tensor):
        raise TypeError("pair_indices must be a torch.Tensor.")
    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must use torch.long.")
    if pair_indices.shape != selected_scores.shape[:1]:
        raise ValueError("pair_indices must have shape (N,).")
    if pair_indices.device != selected_scores.device:
        raise ValueError("pair_indices and selected_scores must share one device.")
    if (
        not torch.isfinite(selected_scores).all().item()
        or not torch.isfinite(target_weights).all().item()
    ):
        raise ValueError("selected_scores and target_weights must be finite.")
    if torch.any(target_weights < 0).item():
        raise ValueError("target_weights must be non-negative.")
    tolerance = max(1e-6, 8 * torch.finfo(target_weights.dtype).eps)
    if not torch.allclose(
        target_weights.sum(dim=-1),
        torch.ones_like(target_weights.sum(dim=-1)),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("target_weights must sum to one per row.")
    if torch.any(pair_indices < 0).item() or torch.any(
        pair_indices >= pair_count
    ).item():
        raise ValueError("pair_indices contains an index outside plan.pairs.")

    return (
        selected_scores.detach().to(device="cpu", dtype=torch.float64).contiguous(),
        target_weights.detach().to(device="cpu", dtype=torch.float64).contiguous(),
        pair_indices.detach().to(device="cpu").contiguous(),
    )


def _cross_entropy(logits: Tensor, targets: Tensor) -> float:
    """Return finite soft-target binary cross entropy."""

    value = float(
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
        ).item()
    )
    if not math.isfinite(value):
        raise MixtureWeightFittingError("Cross entropy is not finite.")
    return value


def _mean_absolute_error(logits: Tensor, targets: Tensor) -> float:
    """Return finite mean absolute first-weight error."""

    value = float(torch.mean(torch.abs(torch.sigmoid(logits) - targets)).item())
    if not math.isfinite(value):
        raise MixtureWeightFittingError("Mean absolute error is not finite.")
    return value


def _sha256_tensor(value: Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    portable = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(portable.dtype).encode("ascii"))
    digest.update(str(tuple(portable.shape)).encode("ascii"))
    digest.update(portable.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _positive_number(value: Any, *, field_name: str) -> float:
    """Return one finite positive public numeric value."""

    normalized = _non_negative_number(value, field_name=field_name)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return normalized


def _non_negative_number(value: Any, *, field_name: str) -> float:
    """Return one finite non-negative public numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite real number.")
    normalized = float(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    """Require a strict JSON mapping schema."""

    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping.")
    if set(value) != expected:
        raise ValueError(
            f"{field_name} fields are incompatible: "
            f"expected={sorted(expected)}, actual={sorted(value)}."
        )
