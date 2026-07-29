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


@dataclass(frozen=True)
class AnchorDeltaCombinationResult:
    """Always-on anchor output and auditable specialist-delta weights.

    ``weights`` has shape ``token_shape + (2,)``. Anchor routes contain
    ``[0, 0]``, top-1 routes contain ``[1, 0]``, and top-2 routes contain
    normalized score-softmax weights. ``output`` preserves the anchor's full
    shape, dtype, and device.
    """

    output: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        """Validate the public tensor and weight contract."""

        if not isinstance(self.output, Tensor) or not self.output.is_floating_point():
            raise TypeError("output must be a floating-point torch.Tensor.")
        if not isinstance(self.weights, Tensor) or not self.weights.is_floating_point():
            raise TypeError("weights must be a floating-point torch.Tensor.")
        if self.weights.ndim < 1 or self.weights.shape[-1] != 2:
            raise ValueError("weights must have a final dimension of size two.")
        if self.output.shape[: self.weights.ndim - 1] != self.weights.shape[:-1]:
            raise ValueError("output must preserve the weights token-leading shape.")
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
        valid_sums = torch.isclose(
            weight_sums,
            torch.zeros_like(weight_sums),
            rtol=0.0,
            atol=tolerance,
        ) | torch.isclose(
            weight_sums,
            torch.ones_like(weight_sums),
            rtol=0.0,
            atol=tolerance,
        )
        if not valid_sums.all().item():
            raise ValueError("delta weights must sum to zero or one per token.")


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


class AlwaysOnAnchorDeltaCombination:
    r"""Add zero, one, or two routed expert deltas to an immutable anchor.

    For anchor output ``A(h)``, selected deltas ``Delta_i(h)`` and effective
    route arity ``k in {0, 1, 2}``, COM-004 computes

    \[
    y(h) =
    \begin{cases}
    A(h), & k=0,\\
    A(h)+\Delta_i(h), & k=1,\\
    A(h)+\alpha_i\Delta_i(h)+\alpha_j\Delta_j(h), & k=2,
    \end{cases}
    \]

    where the top-2 weights are score-softmax weights using
    ``mix_temperature``. Anchor rows bypass addition entirely, preserving the
    exact already-computed dense activation rather than relying on arithmetic
    with a synthetic zero.

    References:
        Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in
        Mixture-of-Experts Language Models." https://arxiv.org/abs/2401.06066

    Differences from the paper:
        DeepSeekMoE jointly trains shared and routed experts. This policy keeps
        one pretrained dense MLP as an immutable always-on anchor and combines
        independently trainable anchor-relative plug-in deltas selected by the
        framework's existing anchored policies.
    """

    policy_name = "always_on_anchor_deltas"

    def __init__(self, mix_temperature: float = 1.0) -> None:
        """Create the delta combiner with a finite positive temperature."""

        if isinstance(mix_temperature, bool) or not isinstance(
            mix_temperature,
            (int, float),
        ):
            raise TypeError("mix_temperature must be a real number.")
        normalized_temperature = float(mix_temperature)
        if not isfinite(normalized_temperature) or normalized_temperature <= 0:
            raise ValueError("mix_temperature must be finite and greater than zero.")
        self.mix_temperature = normalized_temperature

    def __call__(
        self,
        anchor_output: Tensor,
        selected_deltas: Tensor,
        *,
        selected_scores: Tensor,
        route_arity: Tensor,
    ) -> AnchorDeltaCombinationResult:
        """Delegate to :meth:`combine` for ordinary callable use."""

        return self.combine(
            anchor_output,
            selected_deltas,
            selected_scores=selected_scores,
            route_arity=route_arity,
        )

    def combine(
        self,
        anchor_output: Tensor,
        selected_deltas: Tensor,
        *,
        selected_scores: Tensor,
        route_arity: Tensor,
    ) -> AnchorDeltaCombinationResult:
        """Return the anchor plus capacity-approved routed deltas.

        Args:
            anchor_output:
                Floating tensor with shape ``token_shape + output_shape``.
            selected_deltas:
                Floating tensor with shape
                ``token_shape + (2,) + output_shape``. Inactive slots must be
                zero.
            selected_scores:
                Finite tensor with shape ``token_shape + (2,)``.
            route_arity:
                Long tensor with ``token_shape`` and values zero, one, or two.
        """

        _validate_anchor_delta_inputs(
            anchor_output,
            selected_deltas,
            selected_scores=selected_scores,
            route_arity=route_arity,
        )
        token_shape = route_arity.shape
        token_count = route_arity.numel()
        flat_arity = route_arity.reshape(token_count)
        flat_scores = selected_scores.reshape(token_count, 2)
        flat_weights = selected_scores.new_zeros((token_count, 2))

        top1_indices = torch.nonzero(flat_arity == 1, as_tuple=False).flatten()
        if top1_indices.numel():
            top1_weights = selected_scores.new_tensor([1.0, 0.0]).expand(
                top1_indices.numel(),
                2,
            )
            flat_weights = torch.index_copy(
                flat_weights,
                dim=0,
                index=top1_indices,
                source=top1_weights,
            )

        top2_indices = torch.nonzero(flat_arity == 2, as_tuple=False).flatten()
        if top2_indices.numel():
            top2_scores = torch.index_select(flat_scores, 0, top2_indices)
            top2_weights = torch.softmax(
                top2_scores / self.mix_temperature,
                dim=-1,
            )
            flat_weights = torch.index_copy(
                flat_weights,
                dim=0,
                index=top2_indices,
                source=top2_weights,
            )

        weights = flat_weights.reshape(*token_shape, 2)
        trailing_rank = anchor_output.ndim - route_arity.ndim
        broadcast_weights = weights.reshape(
            *weights.shape,
            *((1,) * trailing_rank),
        )
        selected_dimension = route_arity.ndim
        weighted_delta = torch.sum(
            broadcast_weights * selected_deltas,
            dim=selected_dimension,
        )
        active_mask = (route_arity > 0).reshape(
            *token_shape,
            *((1,) * trailing_rank),
        )
        candidate_output = anchor_output + weighted_delta
        # Exact fallback is a semantic invariant: anchor rows select the
        # original activation object values instead of depending on A + 0.
        output = torch.where(active_mask, candidate_output, anchor_output)
        return AnchorDeltaCombinationResult(output=output, weights=weights)


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


def _validate_anchor_delta_inputs(
    anchor_output: Tensor,
    selected_deltas: Tensor,
    *,
    selected_scores: Tensor,
    route_arity: Tensor,
) -> None:
    """Validate COM-004 pair-shaped delta reconstruction inputs."""

    if not isinstance(anchor_output, Tensor) or not anchor_output.is_floating_point():
        raise TypeError("anchor_output must be a floating-point torch.Tensor.")
    if (
        not isinstance(selected_deltas, Tensor)
        or not selected_deltas.is_floating_point()
    ):
        raise TypeError("selected_deltas must be a floating-point torch.Tensor.")
    if (
        not isinstance(selected_scores, Tensor)
        or not selected_scores.is_floating_point()
    ):
        raise TypeError("selected_scores must be a floating-point torch.Tensor.")
    if not isinstance(route_arity, Tensor) or route_arity.dtype != torch.long:
        raise TypeError("route_arity must be a torch.long tensor.")
    if selected_scores.shape != route_arity.shape + (2,):
        raise ValueError("selected_scores must have route shape plus two slots.")
    if anchor_output.ndim <= route_arity.ndim:
        raise ValueError("anchor_output must include a trailing output dimension.")
    expected_delta_shape = (
        route_arity.shape
        + (2,)
        + anchor_output.shape[route_arity.ndim :]
    )
    if selected_deltas.shape != expected_delta_shape:
        raise ValueError(
            "selected_deltas must have route shape, two slots, and the anchor "
            f"output tail: {tuple(selected_deltas.shape)} != "
            f"{tuple(expected_delta_shape)}."
        )
    tensors = (anchor_output, selected_deltas, selected_scores)
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("anchor, deltas, and scores must share one dtype.")
    if len({tensor.device for tensor in tensors}) != 1 or (
        route_arity.device != anchor_output.device
    ):
        raise ValueError("anchor, deltas, scores, and route_arity share one device.")
    if not torch.isfinite(selected_scores).all().item():
        raise ValueError("selected_scores must contain only finite values.")
    if route_arity.numel():
        if torch.any((route_arity < 0) | (route_arity > 2)).item():
            raise ValueError("route_arity values must be zero, one, or two.")
    slot_indices = torch.arange(
        2,
        dtype=torch.long,
        device=route_arity.device,
    )
    inactive_slots = slot_indices >= route_arity.unsqueeze(-1)
    if inactive_slots.any().item():
        trailing_rank = anchor_output.ndim - route_arity.ndim
        expanded_inactive = inactive_slots.reshape(
            *inactive_slots.shape,
            *((1,) * trailing_rank),
        ).expand_as(selected_deltas)
        if torch.any(selected_deltas[expanded_inactive] != 0).item():
            raise ValueError("inactive selected_deltas slots must be exactly zero.")


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
