"""Replay-preservation and expert-initialization drift objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ExpertObjectiveError(ValueError):
    """Raised when TRN-005 objective inputs or snapshots are invalid."""


@dataclass(frozen=True)
class ExpertObjectiveConfig:
    """Configuration for optional forced-path preservation objectives.

    Attributes:
        replay_kl_weight:
            Non-negative coefficient multiplying token-level KL preservation.
            Zero disables the objective and does not require teacher logits.
        drift_weight:
            Non-negative coefficient multiplying parameter-count-normalized
            squared drift from expert initialization. Zero disables snapshots.
        replay_temperature:
            Positive softmax temperature used for both student and frozen
            anchor distributions. The KL is multiplied by ``temperature**2``.
        teacher_logits_key:
            Batch field containing precomputed frozen-anchor logits with shape
            ``(..., V)`` equal to the forced-path model logits.
        replay_token_mask_key:
            Optional batch field containing non-negative replay weights with
            shape ``(...)``. Missing masks fall back to ``attention_mask`` and
            then to all token positions. An all-zero mask contributes zero.
    """

    replay_kl_weight: float = 0.0
    drift_weight: float = 0.0
    replay_temperature: float = 1.0
    teacher_logits_key: str = "anchor_logits"
    replay_token_mask_key: str | None = "replay_token_mask"

    def __post_init__(self) -> None:
        """Validate finite coefficients, temperature, and batch-field names."""

        _validate_nonnegative_finite(
            self.replay_kl_weight,
            field_name="replay_kl_weight",
        )
        _validate_nonnegative_finite(
            self.drift_weight,
            field_name="drift_weight",
        )
        if (
            isinstance(self.replay_temperature, bool)
            or not isinstance(self.replay_temperature, (int, float))
            or not math.isfinite(float(self.replay_temperature))
            or self.replay_temperature <= 0
        ):
            raise ValueError("replay_temperature must be finite and positive.")
        _validate_field_name(
            self.teacher_logits_key,
            field_name="teacher_logits_key",
        )
        if self.replay_token_mask_key is not None:
            _validate_field_name(
                self.replay_token_mask_key,
                field_name="replay_token_mask_key",
            )
            if self.replay_token_mask_key == self.teacher_logits_key:
                raise ValueError(
                    "teacher_logits_key and replay_token_mask_key must differ."
                )

    @property
    def replay_kl_enabled(self) -> bool:
        """Return whether batches must supply frozen-anchor logits."""

        return self.replay_kl_weight > 0

    @property
    def drift_enabled(self) -> bool:
        """Return whether immutable expert initialization snapshots are needed."""

        return self.drift_weight > 0


@dataclass(frozen=True)
class ExpertObjectiveReport:
    """Detached loss components from the most recent trainer computation.

    Attributes:
        specialization_loss: Unweighted base model training loss.
        replay_kl_loss: Unweighted token-masked replay KL, or zero when disabled.
        drift_loss: Unweighted normalized expert drift, or zero when disabled.
        total_loss: Weighted sum used for backpropagation.
    """

    specialization_loss: float
    replay_kl_loss: float
    drift_loss: float
    total_loss: float

    def __post_init__(self) -> None:
        """Reject non-finite diagnostics before they enter training telemetry."""

        for field_name in (
            "specialization_loss",
            "replay_kl_loss",
            "drift_loss",
            "total_loss",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise ExpertObjectiveError(
                    f"{field_name} must be a finite float."
                )


class ReplayKLPreservation:
    r"""Token-masked KL preservation against frozen-anchor logits.

    For student logits \(z\), frozen-anchor logits \(z^A\), temperature \(T\),
    and non-negative replay weights \(m_t\), the objective is

    \[
    L_{KD}
    =
    T^2
    \frac{\sum_t m_t
    KL(p^A_T(\cdot|t)\Vert p_T(\cdot|t))}
    {\sum_t m_t}.
    \]

    The anchor logits are always detached. Calculations use float32 unless the
    student is float64, preventing half-precision softmax underflow while
    preserving gradient flow to student logits. An all-zero mask returns a
    differentiable zero so specialization-only batches can coexist with replay
    batches.

    References:
        Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural
        Network." https://arxiv.org/abs/1503.02531

    Differences from the paper:
        The teacher is the immutable anchor, logits are precomputed, masking is
        token-level, and the hard-label causal loss remains a separate term.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive.")
        self.temperature = float(temperature)

    def __call__(
        self,
        student_logits: Tensor,
        anchor_logits: Tensor,
        *,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Return scalar replay KL for logits shaped ``(..., vocabulary)``.

        Args:
            student_logits: Floating forced-path logits receiving gradients.
            anchor_logits: Floating frozen-anchor logits with identical shape.
            token_mask:
                Optional non-negative weights matching all dimensions except
                vocabulary. Boolean masks are accepted.

        Raises:
            ExpertObjectiveError:
                If shapes, devices, dtypes, values, or mask weights are unsafe.
        """

        _validate_logits(
            student_logits,
            anchor_logits,
        )
        token_shape = student_logits.shape[:-1]
        if token_mask is None:
            weights = torch.ones(
                token_shape,
                dtype=torch.float32,
                device=student_logits.device,
            )
        else:
            if not isinstance(token_mask, Tensor):
                raise ExpertObjectiveError("token_mask must be a Tensor or None.")
            if token_mask.shape != token_shape:
                raise ExpertObjectiveError(
                    "token_mask shape must equal logits.shape[:-1]: "
                    f"{tuple(token_mask.shape)} != {tuple(token_shape)}."
                )
            if token_mask.device != student_logits.device:
                raise ExpertObjectiveError(
                    "token_mask and logits must be on the same device."
                )
            if token_mask.is_complex():
                raise ExpertObjectiveError("token_mask must not be complex.")
            if not torch.isfinite(token_mask).all():
                raise ExpertObjectiveError("token_mask must contain finite values.")
            if (token_mask < 0).any():
                raise ExpertObjectiveError(
                    "token_mask weights must be non-negative."
                )
            weights = token_mask

        compute_dtype = (
            torch.float64
            if student_logits.dtype == torch.float64
            else torch.float32
        )
        weights = weights.to(dtype=compute_dtype)
        weight_sum = weights.sum()
        if weight_sum.item() == 0:
            return student_logits.sum() * 0.0

        temperature = self.temperature
        student = student_logits.to(dtype=compute_dtype) / temperature
        # Detachment is unconditional even if a caller accidentally supplies a
        # differentiable teacher tensor.
        anchor = (
            anchor_logits.detach().to(dtype=compute_dtype) / temperature
        )
        per_token = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(anchor, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        loss = (per_token * weights).sum() / weight_sum
        loss = loss * (temperature**2)
        if not torch.isfinite(loss):
            raise ExpertObjectiveError("Replay KL produced a non-finite loss.")
        return loss


class ExpertWeightDrift:
    r"""Squared expert drift from immutable initialization snapshots.

    For expert parameters \(\theta\), construction-time snapshots
    \(\theta_0\), and total scalar parameter count \(N\), the objective is

    \[
    L_{drift}=\frac{1}{N}\sum_i\|\theta_i-\theta_{0,i}\|_2^2.
    \]

    Snapshots are detached CPU tensors and are not registered on the model, so
    they cannot enter the optimizer or package state. Live parameter objects
    retain ownership of the gradient path.

    References:
        Li et al., "Explicit Inductive Bias for Transfer Learning with
        Convolutional Networks." https://arxiv.org/abs/1802.01483

    Differences from the paper:
        L2-SP is applied only to independently owned expert parameters, uses
        their exact construction-time values as the start point, and is
        normalized by scalar parameter count.
    """

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
    ) -> None:
        if isinstance(named_parameters, (str, bytes)) or not isinstance(
            named_parameters,
            Sequence,
        ):
            raise TypeError(
                "named_parameters must be a sequence of (name, Parameter)."
            )
        if not named_parameters:
            raise ExpertObjectiveError(
                "named_parameters must contain at least one parameter."
            )

        entries: list[tuple[str, nn.Parameter, Tensor]] = []
        names: set[str] = set()
        for item in named_parameters:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], nn.Parameter)
            ):
                raise TypeError(
                    "named_parameters must contain (str, nn.Parameter) pairs."
                )
            name, parameter = item
            _validate_field_name(name, field_name="parameter name")
            if name in names:
                raise ExpertObjectiveError(
                    f"Duplicate drift parameter name: {name!r}."
                )
            if not parameter.is_floating_point():
                raise ExpertObjectiveError(
                    f"Drift parameter {name!r} must be floating point."
                )
            if parameter.device.type == "meta":
                raise ExpertObjectiveError(
                    f"Drift parameter {name!r} cannot be on the meta device."
                )
            if not torch.isfinite(parameter.detach()).all():
                raise ExpertObjectiveError(
                    f"Drift parameter {name!r} must start finite."
                )
            names.add(name)
            entries.append(
                (
                    name,
                    parameter,
                    parameter.detach().cpu().clone(),
                )
            )

        self._entries = tuple(entries)
        self._parameter_count = sum(
            parameter.numel() for _, parameter, _ in self._entries
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return snapshot parameter names in deterministic input order."""

        return tuple(name for name, _, _ in self._entries)

    def snapshot_state(self) -> dict[str, Tensor]:
        """Return independent CPU initialization tensors for safe checkpointing."""

        return {name: initial.clone() for name, _, initial in self._entries}

    def load_snapshot_state(self, state: dict[str, Tensor]) -> None:
        """Restore validated initialization snapshots without changing live weights.

        Keys, shapes, dtypes, and finite values must exactly match the original
        schema. Validation finishes before any snapshot is replaced.
        """

        if set(state) != set(self.parameter_names):
            raise ExpertObjectiveError("Drift snapshot keys do not match parameters.")
        entries = []
        for name, parameter, initial in self._entries:
            tensor = state[name]
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != initial.shape
                or tensor.dtype != initial.dtype
            ):
                raise ExpertObjectiveError("Drift snapshot schema mismatch.")
            if not torch.isfinite(tensor).all():
                raise ExpertObjectiveError("Drift snapshots must be finite.")
            entries.append((name, parameter, tensor.detach().cpu().clone()))
        self._entries = tuple(entries)

    @property
    def parameter_count(self) -> int:
        """Return the scalar count used to normalize squared drift."""

        return self._parameter_count

    def __call__(self) -> Tensor:
        """Return scalar normalized drift with gradients to live parameters."""

        devices = {parameter.device for _, parameter, _ in self._entries}
        if len(devices) != 1:
            raise ExpertObjectiveError(
                "Drift parameters must share one device in this implementation."
            )

        total: Tensor | None = None
        for name, parameter, initial in self._entries:
            if parameter.shape != initial.shape:
                raise ExpertObjectiveError(
                    f"Drift parameter {name!r} changed shape after snapshot."
                )
            if not torch.isfinite(parameter.detach()).all():
                raise ExpertObjectiveError(
                    f"Drift parameter {name!r} became non-finite."
                )
            compute_dtype = (
                torch.float64
                if parameter.dtype == torch.float64
                else torch.float32
            )
            difference = parameter.to(dtype=compute_dtype) - initial.to(
                device=parameter.device,
                dtype=compute_dtype,
            )
            squared = difference.square().sum()
            total = squared if total is None else total + squared

        if total is None:
            raise ExpertObjectiveError("Drift snapshot is unexpectedly empty.")
        loss = total / self._parameter_count
        if not torch.isfinite(loss):
            raise ExpertObjectiveError("Weight drift produced a non-finite loss.")
        return loss


def _validate_logits(student_logits: Tensor, anchor_logits: Tensor) -> None:
    """Validate common KL tensor invariants before numerical operations."""

    if not isinstance(student_logits, Tensor) or not isinstance(
        anchor_logits,
        Tensor,
    ):
        raise ExpertObjectiveError(
            "student_logits and anchor_logits must be Tensor values."
        )
    if student_logits.ndim < 2 or student_logits.shape[-1] < 2:
        raise ExpertObjectiveError(
            "Logits must have shape (..., vocabulary) with vocabulary >= 2."
        )
    if student_logits.shape != anchor_logits.shape:
        raise ExpertObjectiveError(
            "Student and anchor logits must have identical shapes: "
            f"{tuple(student_logits.shape)} != {tuple(anchor_logits.shape)}."
        )
    if student_logits.device != anchor_logits.device:
        raise ExpertObjectiveError(
            "Student and anchor logits must be on the same device."
        )
    if not student_logits.is_floating_point() or not (
        anchor_logits.is_floating_point()
    ):
        raise ExpertObjectiveError("Student and anchor logits must be floating.")
    if not torch.isfinite(student_logits.detach()).all() or not torch.isfinite(
        anchor_logits.detach()
    ).all():
        raise ExpertObjectiveError(
            "Student and anchor logits must contain finite values."
        )


def _validate_nonnegative_finite(value: float, *, field_name: str) -> None:
    """Validate one scalar objective coefficient."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{field_name} must be finite and non-negative.")


def _validate_field_name(value: str, *, field_name: str) -> None:
    """Validate stable non-empty field or parameter names."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace."
        )
