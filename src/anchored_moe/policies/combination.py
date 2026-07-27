"""Activation-space combination policies for selected specialist pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActivationCombinationResult:
    """Combined activation and auditable per-specialist mixture weights.

    Attributes:
        output:
            Weighted activation with shape ``token_shape + output_shape``.
        weights:
            Mixture weights with shape ``token_shape + (2,)``. The final
            dimension follows the already selected specialist order.
    """

    output: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        """Validate activation, pair-weight, dtype, device, and sum invariants."""

        if not isinstance(self.output, Tensor):
            raise TypeError("output must be a torch.Tensor.")
        if not self.output.is_floating_point():
            raise TypeError("output must use a floating-point dtype.")
        if not isinstance(self.weights, Tensor):
            raise TypeError("weights must be a torch.Tensor.")
        if not self.weights.is_floating_point():
            raise TypeError("weights must use a floating-point dtype.")
        if self.weights.ndim < 1 or self.weights.shape[-1] != 2:
            raise ValueError("weights must have a final dimension of size two.")
        if self.output.shape[: self.weights.ndim - 1] != self.weights.shape[:-1]:
            raise ValueError(
                "output must preserve the token-leading shape of weights."
            )
        if self.output.dtype != self.weights.dtype:
            raise ValueError("output and weights must share one dtype.")
        if self.output.device != self.weights.device:
            raise ValueError("output and weights must share one device.")
        if not torch.isfinite(self.weights).all().item():
            raise ValueError("weights must contain only finite values.")
        if torch.any(self.weights < 0).item():
            raise ValueError("weights must be non-negative.")

        weight_sums = self.weights.sum(dim=-1)
        tolerance = max(1e-6, 8 * torch.finfo(self.weights.dtype).eps)
        if not torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("weights must sum to one along the selected pair.")


class CombinationPolicy(ABC):
    """Interface for combining already selected specialist activations."""

    policy_name: ClassVar[str]

    def __call__(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Delegate to :meth:`combine` for ordinary callable use."""

        return self.combine(
            selected_outputs,
            selected_scores=selected_scores,
        )

    @abstractmethod
    def combine(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Combine selected specialist outputs using policy-specific weights."""


class ScoreSoftmaxCombination(CombinationPolicy):
    """Mix two selected specialist outputs with temperature-scaled softmax.

    For selected specialist scores ``[g_i, g_j]`` and positive temperature
    ``T_mix``, COM-001 computes

    ``alpha = softmax([g_i, g_j] / T_mix)``

    and returns ``alpha_i * E_i(h) + alpha_j * E_j(h)``. The candidate
    dimension is the last dimension of ``selected_scores`` and the matching
    prefix dimension of ``selected_outputs``.

    Selection and combination remain intentionally separate: callers must
    pass only specialist pairs approved by POL-002. This policy does not
    interpret anchor indices, execute experts, check pair compatibility, or
    handle capacity.
    """

    policy_name = "score_softmax"

    def __init__(self, mix_temperature: float = 1.0) -> None:
        """Create a stateless combiner with a finite positive temperature."""

        if isinstance(mix_temperature, bool) or not isinstance(
            mix_temperature,
            (int, float),
        ):
            raise TypeError("mix_temperature must be a real number.")
        normalized_temperature = float(mix_temperature)
        if not isfinite(normalized_temperature) or normalized_temperature <= 0:
            raise ValueError("mix_temperature must be finite and greater than zero.")
        self.mix_temperature = normalized_temperature

    def combine(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Return the weighted activation and the exact deployed weights.

        Args:
            selected_outputs:
                Floating-point tensor with shape
                ``token_shape + (2,) + output_shape``.
            selected_scores:
                Finite floating-point tensor with shape
                ``token_shape + (2,)``.

        Returns:
            An :class:`ActivationCombinationResult` preserving output dtype,
            device, token order, and trailing output shape.

        Raises:
            TypeError:
                If inputs are not floating-point tensors.
            ValueError:
                If pair shapes, dtypes, devices, or scores are inconsistent.
        """

        _validate_combination_inputs(
            selected_outputs,
            selected_scores=selected_scores,
        )

        # torch.softmax subtracts the maximum internally, so even extreme
        # calibrated scores produce finite normalized weights. Keeping the
        # router and activation dtype equal avoids silent output promotion.
        weights = torch.softmax(
            selected_scores / self.mix_temperature,
            dim=-1,
        )
        return _combine_with_weights(
            selected_outputs,
            selected_scores=selected_scores,
            weights=weights,
        )


class EqualWeightCombination(CombinationPolicy):
    """Mix two selected specialist outputs with fixed equal weights.

    COM-002's diagnostic baseline computes

    ``y = 0.5 * E_i(h) + 0.5 * E_j(h)``

    for every already approved specialist pair. Scores remain a required input
    so all combination policies share one interface and validate the same
    ranked-pair contract, but the output intentionally does not depend on
    their magnitudes.

    Selection and combination remain separate: callers must pass only
    specialist pairs approved by a selection policy. This policy does not
    interpret anchor indices, execute experts, check pair compatibility, or
    handle capacity.
    """

    policy_name = "equal_weight"

    def combine(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Return the arithmetic mean and exact one-half mixture weights."""

        _validate_combination_inputs(
            selected_outputs,
            selected_scores=selected_scores,
        )
        weights = torch.full_like(selected_scores, 0.5)
        return _combine_with_weights(
            selected_outputs,
            selected_scores=selected_scores,
            weights=weights,
        )


class NormalizedPositiveEvidenceCombination(CombinationPolicy):
    """Mix selected outputs in proportion to non-negative router evidence.

    For calibrated selected scores ``[g_i, g_j]``, COM-002 computes

    ``e_k = max(g_k, 0)`` and ``alpha_k = e_k / (e_i + e_j)``.

    At least one selected score must therefore be strictly positive. A pair
    with zero total positive evidence is rejected instead of receiving a
    hidden fallback policy. POL-002-approved specialist pairs always satisfy
    the stronger condition that both selected scores are positive.

    The piecewise-linear weighting is useful as a non-exponential baseline.
    It preserves gradients through positive scores and selected activations;
    non-positive scores have zero evidence and therefore zero local gradient.
    """

    policy_name = "normalized_positive_evidence"

    def combine(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Return positive-evidence-weighted activations and deployed weights.

        Raises:
            ValueError:
                If any selected pair has no strictly positive score and its
                normalized evidence would therefore be undefined.
        """

        _validate_combination_inputs(
            selected_outputs,
            selected_scores=selected_scores,
        )
        evidence = torch.clamp_min(selected_scores, 0)
        total_evidence = evidence.sum(dim=-1, keepdim=True)
        if torch.any(total_evidence <= 0).item():
            raise ValueError(
                "normalized positive evidence requires at least one strictly "
                "positive selected score per pair."
            )
        weights = evidence / total_evidence
        return _combine_with_weights(
            selected_outputs,
            selected_scores=selected_scores,
            weights=weights,
        )


def _validate_combination_inputs(
    selected_outputs: Tensor,
    *,
    selected_scores: Tensor,
) -> None:
    """Validate the shared pair-shaped activation-combination contract."""

    if not isinstance(selected_outputs, Tensor):
        raise TypeError("selected_outputs must be a torch.Tensor.")
    if not selected_outputs.is_floating_point():
        raise TypeError("selected_outputs must use a floating-point dtype.")
    if not isinstance(selected_scores, Tensor):
        raise TypeError("selected_scores must be a torch.Tensor.")
    if not selected_scores.is_floating_point():
        raise TypeError("selected_scores must use a floating-point dtype.")
    if selected_scores.ndim < 1 or selected_scores.shape[-1] != 2:
        raise ValueError(
            "selected_scores must have a final dimension of size two."
        )
    if selected_outputs.ndim <= selected_scores.ndim:
        raise ValueError(
            "selected_outputs must include at least one trailing output "
            "dimension after the selected pair."
        )
    if selected_outputs.shape[: selected_scores.ndim] != selected_scores.shape:
        raise ValueError(
            "selected_outputs must begin with selected_scores shape."
        )
    if selected_outputs.dtype != selected_scores.dtype:
        raise ValueError(
            "selected_outputs and selected_scores must share one dtype."
        )
    if selected_outputs.device != selected_scores.device:
        raise ValueError(
            "selected_outputs and selected_scores must share one device."
        )
    if not torch.isfinite(selected_scores).all().item():
        raise ValueError("selected_scores must contain only finite values.")


def _combine_with_weights(
    selected_outputs: Tensor,
    *,
    selected_scores: Tensor,
    weights: Tensor,
) -> ActivationCombinationResult:
    """Broadcast validated pair weights and combine selected activations."""

    trailing_rank = selected_outputs.ndim - selected_scores.ndim
    broadcast_weights = weights.reshape(
        *weights.shape,
        *((1,) * trailing_rank),
    )
    selected_dimension = selected_scores.ndim - 1
    output = torch.sum(
        broadcast_weights * selected_outputs,
        dim=selected_dimension,
    )
    return ActivationCombinationResult(output=output, weights=weights)
