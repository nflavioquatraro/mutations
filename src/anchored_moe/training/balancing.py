"""Training-time auxiliary load balancing for package-local routers.

BAL-003 measures whether a local router's preferred expert assignments and
mean routing probabilities are distributed across the available experts. It
does not change token dispatch, capacity, inference scores, or expert
availability.

References:
    Fedus, Zoph, and Shazeer, "Switch Transformers: Scaling to Trillion
    Parameter Models with Simple and Efficient Sparsity."
    https://arxiv.org/abs/2101.03961

Differences from the paper:
    The objective is applied independently to each converted layer of a
    package-local, two-expert, dense-mixture training path. Both experts still
    execute for every token; hard assignments are used only to measure load.
    Optional token weights exclude padding and the trainer averages the loss
    across converted layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor
from torch.nn import functional as F


class AuxiliaryLoadBalancingError(ValueError):
    """Raised when BAL-003 probabilities or token weights are invalid."""


@dataclass(frozen=True)
class LocalMiniMoEObjectiveReport:
    """Detached loss components from the latest local mini-MoE forward.

    Attributes:
        specialization_loss:
            Unweighted causal-language-model loss returned by the anchor model.
        load_balancing_loss:
            Unweighted BAL-003 loss averaged across converted layers, or zero
            when balancing is disabled.
        total_loss:
            Differentiable training loss after applying the configured
            load-balancing coefficient.
    """

    specialization_loss: float
    load_balancing_loss: float
    total_loss: float

    def __post_init__(self) -> None:
        """Reject non-finite diagnostics before they reach training reports."""

        for field_name in (
            "specialization_loss",
            "load_balancing_loss",
            "total_loss",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not isfinite(value):
                raise AuxiliaryLoadBalancingError(
                    f"{field_name} must be a finite float."
                )
        if self.load_balancing_loss < 0:
            raise AuxiliaryLoadBalancingError(
                "load_balancing_loss must be non-negative."
            )


class AuxiliaryLoadBalancingLoss:
    r"""Compute the Switch-style router load-balancing objective.

    For token routing probabilities \(p_{t,i}\), optional non-negative token
    weights \(m_t\), and \(N\) experts, define

    \[
    P_i = \frac{\sum_t m_t p_{t,i}}{\sum_t m_t},
    \qquad
    f_i = \frac{\sum_t m_t
    \mathbb{1}[\operatorname{argmax}_j p_{t,j}=i]}{\sum_t m_t}.
    \]

    BAL-003 returns

    \[
    L_{\mathrm{balance}} = N\sum_i f_i P_i.
    \]

    The hard assignment fractions \(f_i\) are detached; gradients flow through
    the mean probabilities \(P_i\). Stable ``torch.argmax`` tie-breaking makes
    the first local expert win an exact probability tie. A perfectly balanced
    distribution has loss one, while complete two-expert collapse has loss
    two. An all-zero token mask returns a differentiable zero.

    References:
        Fedus, Zoph, and Shazeer, "Switch Transformers: Scaling to Trillion
        Parameter Models with Simple and Efficient Sparsity."
        https://arxiv.org/abs/2101.03961

    Differences from the paper:
        The framework uses optional weighted token masks, computes the
        objective per converted layer, and applies it to mixture weights from
        a dense two-expert local training path rather than a sparse global
        top-1 dispatcher.
    """

    def __call__(
        self,
        routing_probabilities: Tensor,
        *,
        token_weights: Tensor | None = None,
    ) -> Tensor:
        """Return one scalar differentiable balancing loss.

        Args:
            routing_probabilities:
                Floating tensor with shape ``token_shape + (N,)``. Values must
                be finite, non-negative, and sum to one over experts.
            token_weights:
                Optional finite non-negative tensor with ``token_shape``.
                Boolean attention masks are accepted.

        Raises:
            AuxiliaryLoadBalancingError:
                If shapes, values, devices, or probability normalization are
                incompatible with the objective.
        """

        if not isinstance(routing_probabilities, Tensor):
            raise AuxiliaryLoadBalancingError(
                "routing_probabilities must be a torch.Tensor."
            )
        if (
            not routing_probabilities.is_floating_point()
            or routing_probabilities.ndim < 2
            or routing_probabilities.shape[-1] < 2
        ):
            raise AuxiliaryLoadBalancingError(
                "routing_probabilities must be floating point with at least "
                "one token dimension and two experts."
            )
        if not torch.isfinite(routing_probabilities).all().item():
            raise AuxiliaryLoadBalancingError(
                "routing_probabilities must contain only finite values."
            )
        if torch.any(routing_probabilities < 0).item():
            raise AuxiliaryLoadBalancingError(
                "routing_probabilities must be non-negative."
            )

        probability_sums = routing_probabilities.sum(dim=-1)
        tolerance = max(
            1e-6,
            8 * torch.finfo(routing_probabilities.dtype).eps,
        )
        if not torch.allclose(
            probability_sums,
            torch.ones_like(probability_sums),
            rtol=0.0,
            atol=tolerance,
        ):
            raise AuxiliaryLoadBalancingError(
                "routing_probabilities must sum to one over experts."
            )

        token_shape = routing_probabilities.shape[:-1]
        if token_weights is None:
            weights = torch.ones(
                token_shape,
                dtype=routing_probabilities.dtype,
                device=routing_probabilities.device,
            )
        else:
            if not isinstance(token_weights, Tensor):
                raise AuxiliaryLoadBalancingError(
                    "token_weights must be a torch.Tensor or None."
                )
            if token_weights.shape != token_shape:
                raise AuxiliaryLoadBalancingError(
                    "token_weights shape must equal the probability token "
                    f"shape: {tuple(token_weights.shape)} != {token_shape}."
                )
            if token_weights.device != routing_probabilities.device:
                raise AuxiliaryLoadBalancingError(
                    "token_weights and routing_probabilities must share a device."
                )
            if token_weights.is_complex():
                raise AuxiliaryLoadBalancingError(
                    "token_weights must not be complex."
                )
            if not torch.isfinite(token_weights).all().item():
                raise AuxiliaryLoadBalancingError(
                    "token_weights must contain only finite values."
                )
            if torch.any(token_weights < 0).item():
                raise AuxiliaryLoadBalancingError(
                    "token_weights must be non-negative."
                )
            weights = token_weights.to(dtype=routing_probabilities.dtype)

        flat_probabilities = routing_probabilities.reshape(
            -1,
            routing_probabilities.shape[-1],
        )
        flat_weights = weights.reshape(-1)
        weight_sum = flat_weights.sum()
        if weight_sum.item() == 0:
            return routing_probabilities.sum() * 0.0

        normalized_weights = flat_weights / weight_sum
        mean_probabilities = (
            flat_probabilities * normalized_weights.unsqueeze(-1)
        ).sum(dim=0)
        hard_assignments = torch.argmax(flat_probabilities.detach(), dim=-1)
        assignment_fractions = (
            F.one_hot(
                hard_assignments,
                num_classes=flat_probabilities.shape[-1],
            ).to(dtype=flat_probabilities.dtype)
            * normalized_weights.unsqueeze(-1)
        ).sum(dim=0).detach()
        loss = flat_probabilities.shape[-1] * torch.sum(
            assignment_fractions * mean_probabilities
        )
        if not torch.isfinite(loss).item():
            raise AuxiliaryLoadBalancingError(
                "Auxiliary load balancing produced a non-finite loss."
            )
        return loss
