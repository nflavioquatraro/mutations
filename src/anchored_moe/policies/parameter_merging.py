"""Reference parameter-space merging for compatible PEFT/LoRA expert pairs.

COM-005 constructs an input-dependent MLP by merging effective LoRA projection
updates before one logical anchor-MLP execution. It deliberately computes
``alpha_1 * (s_1 B_1 A_1) + alpha_2 * (s_2 B_2 A_2)``. Interpolating ``A`` and
``B`` independently would introduce unintended cross terms.

References:
    Muqeeth et al., "Soft Merging of Experts with Adaptive Routing."
    https://arxiv.org/abs/2306.03745

    Buehler and Buehler, "X-LoRA: Mixture of Low-Rank Adapter Experts."
    https://arxiv.org/abs/2402.07148

Differences from the referenced methods:
    This correctness-first backend merges exactly one or two already selected
    PEFT/LoRA MLP experts with a frozen COM-003 score head. It materializes
    effective dense projection overrides per token through
    ``torch.func.functional_call``; it does not implement SMEAR training,
    X-LoRA's scaling network, arbitrary expert counts, or a fused kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.func import functional_call

from anchored_moe.experts.low_rank import PEFTLoRAExpert, LoRAProjectionDelta
from anchored_moe.policies.learned_combination import LearnedMixtureWeightHead


class ParameterSpaceMergeError(ValueError):
    """Raised when LoRA experts cannot form one safe merged parameter space."""


@dataclass(frozen=True)
class ParameterSpaceMergeResult:
    """Output and exact coefficients from one parameter-space execution.

    Attributes:
        output:
            Merged MLP output with shape ``(token_count, *output_shape)``.
        weights:
            Applied coefficients with shape ``(token_count, expert_count)``.
        projection_names:
            Canonical full names of all projection weights overridden during
            the functional anchor execution.
    """

    output: Tensor
    weights: Tensor
    projection_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate output, coefficient, and projection metadata."""

        if not isinstance(self.output, Tensor) or not self.output.is_floating_point():
            raise TypeError("output must be a floating-point torch.Tensor.")
        if not isinstance(self.weights, Tensor) or not self.weights.is_floating_point():
            raise TypeError("weights must be a floating-point torch.Tensor.")
        if self.weights.ndim != 2 or self.weights.shape[1] not in (1, 2):
            raise ValueError("weights must have shape (token_count, 1 or 2).")
        if self.output.shape[0] != self.weights.shape[0]:
            raise ValueError("output and weights must share token_count.")
        if (
            self.output.dtype != self.weights.dtype
            or self.output.device != self.weights.device
        ):
            raise ValueError("output and weights must share one dtype and device.")
        if (
            not torch.isfinite(self.weights).all().item()
            or torch.any(self.weights < 0).item()
        ):
            raise ValueError("weights must be finite and non-negative.")
        tolerance = max(1e-6, 8 * torch.finfo(self.weights.dtype).eps)
        if not torch.allclose(
            self.weights.sum(dim=-1),
            torch.ones(
                self.weights.shape[0],
                dtype=self.weights.dtype,
                device=self.weights.device,
            ),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("weights must sum to one for every token.")
        if (
            not isinstance(self.projection_names, tuple)
            or not self.projection_names
            or len(set(self.projection_names)) != len(self.projection_names)
        ):
            raise ValueError("projection_names must be a non-empty unique tuple.")
        if not all(
            isinstance(name, str)
            and name
            and name == name.strip()
            for name in self.projection_names
        ):
            raise ValueError("projection_names must contain normalized strings.")


class LoRAParameterSpaceMerge:
    r"""Merge effective LoRA projection deltas before MLP execution.

    For compatible experts \(i,j\) and one targeted anchor projection \(W\),
    this policy constructs, independently for every input token \(t\),

    \[
    W_t = W + \alpha_{t,i}s_iB_iA_i
              + \alpha_{t,j}s_jB_jA_j.
    \]

    The selected coefficients come from one frozen
    :class:`LearnedMixtureWeightHead`. Capacity-degraded top-1 calls bypass
    that head and supply the exact unit coefficient through :meth:`merge`.

    The implementation is intentionally a reference backend: each token uses
    a separate functional call so its projection parameters may differ. It
    validates private expert bases against the live anchor and never mutates
    either object.
    """

    policy_name: ClassVar[str] = "parameter_space_lora_merge"

    def __init__(self, head: LearnedMixtureWeightHead) -> None:
        """Bind one independently fitted, immutable COM-003 mixture head."""

        if not isinstance(head, LearnedMixtureWeightHead):
            raise TypeError("head must be a LearnedMixtureWeightHead.")
        if tuple(head.parameters()):
            raise ValueError("The COM-003 deployment head must be frozen.")
        self.head = head

    def pair_weights(self, selected_scores: Tensor) -> Tensor:
        """Return learned weights for ordered capacity-approved score pairs."""

        return self.head(selected_scores)

    def validate_compatibility(
        self,
        anchor_mlp: nn.Module,
        experts: Sequence[PEFTLoRAExpert],
    ) -> tuple[str, ...]:
        """Validate selected expert topology without executing the anchor.

        This preflight lets dispatch and routed-layer configuration fail before
        any candidate call or runtime mutation. It intentionally recomputes
        graph-connected deltas during the later merge so the execution graph
        never depends on cached parameter-derived tensors.
        """

        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(experts, Sequence) or isinstance(experts, (str, bytes)):
            raise TypeError(
                "experts must be a sequence of PEFTLoRAExpert objects."
            )
        normalized_experts = tuple(experts)
        if len(normalized_experts) not in (1, 2):
            raise ValueError("COM-005 supports exactly one or two selected experts.")
        if not all(
            isinstance(expert, PEFTLoRAExpert)
            for expert in normalized_experts
        ):
            raise TypeError("experts must contain only PEFTLoRAExpert objects.")
        projection_banks = tuple(
            expert.projection_deltas() for expert in normalized_experts
        )
        projection_names, _, _ = _validate_topology(
            anchor_mlp,
            normalized_experts,
            projection_banks,
            hidden_states=None,
        )
        return projection_names

    def merge(
        self,
        hidden_states: Tensor,
        *,
        anchor_mlp: nn.Module,
        experts: Sequence[PEFTLoRAExpert],
        weights: Tensor,
        module_args: Sequence[Any] = (),
        module_kwargs: dict[str, Any] | None = None,
    ) -> ParameterSpaceMergeResult:
        """Execute one or two effective LoRA experts in parameter space.

        Args:
            hidden_states:
                Two-dimensional floating-point tensor ``(token_count, H)``.
            anchor_mlp:
                Live immutable MLP whose targeted projection weights define
                the common parameter origin.
            experts:
                One or two PEFT/LoRA experts in exact coefficient order.
            weights:
                Non-negative normalized coefficients with shape
                ``(token_count, len(experts))``.
            module_args:
                Global or broadcast-compatible positional MLP arguments.
            module_kwargs:
                Global or broadcast-compatible keyword MLP arguments.

        Returns:
            A :class:`ParameterSpaceMergeResult` containing outputs, the exact
            applied weights, and the canonical overridden projections.

        Raises:
            ParameterSpaceMergeError:
                If expert topology, private bases, or live anchor projections
                are incompatible.
            TypeError:
                If public inputs do not have their declared types.
            ValueError:
                If tensor shapes, dtypes, devices, or weights are invalid.
        """

        normalized_experts = _validate_merge_inputs(
            hidden_states,
            anchor_mlp=anchor_mlp,
            experts=experts,
            weights=weights,
            module_args=module_args,
            module_kwargs=module_kwargs,
        )
        projection_banks = tuple(
            expert.projection_deltas() for expert in normalized_experts
        )
        projection_names, anchor_weights, delta_banks = _validate_topology(
            anchor_mlp,
            normalized_experts,
            projection_banks,
            hidden_states=hidden_states,
        )
        forwarded_kwargs = {} if module_kwargs is None else dict(module_kwargs)
        forwarded_args = tuple(module_args)

        zero_updates = all(
            torch.count_nonzero(delta).item() == 0
            for bank in delta_banks
            for delta in bank
        )

        token_outputs: list[Tensor] = []
        output_tail: tuple[int, ...] | None = None
        for token_index in range(hidden_states.shape[0]):
            overrides: dict[str, Tensor] = {}
            for projection_index, projection_name in enumerate(projection_names):
                merged_weight = anchor_weights[projection_index]
                for expert_index in range(len(normalized_experts)):
                    merged_weight = merged_weight + (
                        weights[token_index, expert_index]
                        * delta_banks[expert_index][projection_index]
                    )
                overrides[f"{projection_name}.weight"] = merged_weight

            # A separate functional call is the correctness reference for
            # token-dependent parameters. It leaves the registered anchor
            # tensors untouched and keeps gradients connected to scores and
            # LoRA factors through each override.
            token_output = functional_call(
                anchor_mlp,
                overrides,
                (
                    hidden_states[token_index : token_index + 1],
                    *forwarded_args,
                ),
                forwarded_kwargs,
                strict=False,
            )
            if not isinstance(token_output, Tensor) or not token_output.is_floating_point():
                raise ParameterSpaceMergeError(
                    "The anchor MLP must return one floating-point tensor."
                )
            if token_output.shape[0] != 1:
                raise ParameterSpaceMergeError(
                    "The anchor MLP must preserve the one-token leading dimension."
                )
            if (
                token_output.dtype != hidden_states.dtype
                or token_output.device != hidden_states.device
            ):
                raise ParameterSpaceMergeError(
                    "The merged MLP output must preserve input dtype and device."
                )
            current_tail = tuple(token_output.shape[1:])
            if output_tail is None:
                output_tail = current_tail
            elif current_tail != output_tail:
                raise ParameterSpaceMergeError(
                    "Merged MLP outputs must share one trailing shape."
                )
            token_outputs.append(token_output)

        if not token_outputs:
            raise ValueError("Cannot merge an input with zero tokens.")
        merged_output = torch.cat(token_outputs, dim=0)
        if zero_updates:
            # Equivalent per-token functional calls can choose a different
            # matmul kernel than one batched anchor call. Use exact batched
            # anchor values in the forward pass while retaining the functional
            # graph for LoRA and hidden-state gradients.
            anchor_output = anchor_mlp(
                hidden_states,
                *forwarded_args,
                **forwarded_kwargs,
            )
            if (
                not isinstance(anchor_output, Tensor)
                or not anchor_output.is_floating_point()
                or anchor_output.shape != merged_output.shape
                or anchor_output.dtype != merged_output.dtype
                or anchor_output.device != merged_output.device
            ):
                raise ParameterSpaceMergeError(
                    "The anchor MLP must match the merged output contract."
                )
            merged_output = anchor_output.detach() + (
                merged_output - merged_output.detach()
            )

        return ParameterSpaceMergeResult(
            output=merged_output,
            weights=weights,
            projection_names=projection_names,
        )


def _validate_merge_inputs(
    hidden_states: Tensor,
    *,
    anchor_mlp: nn.Module,
    experts: Sequence[PEFTLoRAExpert],
    weights: Tensor,
    module_args: Sequence[Any],
    module_kwargs: dict[str, Any] | None,
) -> tuple[PEFTLoRAExpert, ...]:
    """Validate public reference-executor inputs before reading expert state."""

    if not isinstance(hidden_states, Tensor) or not hidden_states.is_floating_point():
        raise TypeError("hidden_states must be a floating-point torch.Tensor.")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape (token_count, hidden_size).")
    if hidden_states.shape[0] == 0:
        raise ValueError("hidden_states must contain at least one token.")
    if not isinstance(anchor_mlp, nn.Module):
        raise TypeError("anchor_mlp must be a torch.nn.Module.")
    if not isinstance(experts, Sequence) or isinstance(experts, (str, bytes)):
        raise TypeError("experts must be a sequence of PEFTLoRAExpert objects.")
    normalized_experts = tuple(experts)
    if len(normalized_experts) not in (1, 2):
        raise ValueError("COM-005 supports exactly one or two selected experts.")
    if not all(isinstance(expert, PEFTLoRAExpert) for expert in normalized_experts):
        raise TypeError("experts must contain only PEFTLoRAExpert objects.")
    if not isinstance(weights, Tensor) or not weights.is_floating_point():
        raise TypeError("weights must be a floating-point torch.Tensor.")
    if weights.shape != (hidden_states.shape[0], len(normalized_experts)):
        raise ValueError(
            "weights must have shape (token_count, selected_expert_count)."
        )
    if weights.dtype != hidden_states.dtype or weights.device != hidden_states.device:
        raise ValueError("weights and hidden_states must share one dtype and device.")
    if not torch.isfinite(weights).all().item() or torch.any(weights < 0).item():
        raise ValueError("weights must be finite and non-negative.")
    tolerance = max(1e-6, 8 * torch.finfo(weights.dtype).eps)
    if not torch.allclose(
        weights.sum(dim=-1),
        torch.ones(weights.shape[0], dtype=weights.dtype, device=weights.device),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("weights must sum to one for every token.")
    if not isinstance(module_args, Sequence) or isinstance(module_args, (str, bytes)):
        raise TypeError("module_args must be a sequence.")
    if module_kwargs is not None and not isinstance(module_kwargs, dict):
        raise TypeError("module_kwargs must be a dictionary or None.")
    return normalized_experts


def _validate_topology(
    anchor_mlp: nn.Module,
    experts: tuple[PEFTLoRAExpert, ...],
    projection_banks: tuple[tuple[LoRAProjectionDelta, ...], ...],
    *,
    hidden_states: Tensor | None,
) -> tuple[tuple[str, ...], tuple[Tensor, ...], tuple[tuple[Tensor, ...], ...]]:
    """Resolve matching live projections and reject incompatible private bases."""

    anchor_type = type(anchor_mlp)
    for expert in experts:
        if type(expert.expert_mlp) is not anchor_type:
            raise ParameterSpaceMergeError(
                "Every expert must wrap the same MLP class as the live anchor."
            )
    projection_names = tuple(
        projection.module_name for projection in projection_banks[0]
    )
    for bank in projection_banks[1:]:
        if tuple(projection.module_name for projection in bank) != projection_names:
            raise ParameterSpaceMergeError(
                "Selected LoRA experts must target the same resolved projections."
            )

    anchor_weights: list[Tensor] = []
    delta_banks: list[list[Tensor]] = [[] for _ in experts]
    for projection_index, projection_name in enumerate(projection_names):
        try:
            live_projection = anchor_mlp.get_submodule(projection_name)
        except AttributeError as error:
            raise ParameterSpaceMergeError(
                f"Live anchor lacks LoRA projection {projection_name!r}."
            ) from error
        if not isinstance(live_projection, nn.Linear):
            raise ParameterSpaceMergeError(
                f"Live projection {projection_name!r} must be torch.nn.Linear."
            )
        live_weight = live_projection.weight
        if hidden_states is not None and (
            live_weight.dtype != hidden_states.dtype
            or live_weight.device != hidden_states.device
        ):
            raise ParameterSpaceMergeError(
                "Live anchor projections and hidden states must share dtype/device."
            )
        for expert_index, bank in enumerate(projection_banks):
            projection = bank[projection_index]
            if not torch.equal(projection.base_weight, live_weight):
                raise ParameterSpaceMergeError(
                    f"Expert {expert_index} private base differs from live "
                    f"projection {projection_name!r}."
                )
            live_bias = live_projection.bias
            if (projection.base_bias is None) != (live_bias is None):
                raise ParameterSpaceMergeError(
                    f"Expert {expert_index} bias topology differs at "
                    f"{projection_name!r}."
                )
            if (
                live_bias is not None
                and projection.base_bias is not None
                and not torch.equal(projection.base_bias, live_bias)
            ):
                raise ParameterSpaceMergeError(
                    f"Expert {expert_index} private bias differs from live "
                    f"projection {projection_name!r}."
                )
            delta_banks[expert_index].append(projection.delta_weight)
        anchor_weights.append(live_weight)

    return (
        projection_names,
        tuple(anchor_weights),
        tuple(tuple(bank) for bank in delta_banks),
    )
