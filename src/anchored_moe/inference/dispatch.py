"""Sparse token dispatch for anchored hard top-1 routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from anchored_moe.policies.capacity import (
    CapacityDecision,
    OverflowToAnchorCapacity,
)
from anchored_moe.policies.selection import (
    AnchoredTop1Selection,
    RouteDecision,
)


@dataclass(frozen=True)
class Top1DispatchResult:
    """Output and auditable decisions from one anchored top-1 dispatch.

    Attributes:
        output:
            Reconstructed MLP output with the same token-leading shape as the
            input hidden states.
        selected_routes:
            Route decision produced before capacity fallback.
        capacity:
            Effective routes and overflow mask used for execution.
    """

    output: Tensor
    selected_routes: RouteDecision
    capacity: CapacityDecision

    def __post_init__(self) -> None:
        """Validate result types and token-leading shape agreement."""

        if not isinstance(self.output, Tensor):
            raise TypeError("output must be a torch.Tensor.")
        if not isinstance(self.selected_routes, RouteDecision):
            raise TypeError("selected_routes must be a RouteDecision.")
        if not isinstance(self.capacity, CapacityDecision):
            raise TypeError("capacity must be a CapacityDecision.")
        if self.capacity.routes.expert_ids != self.selected_routes.expert_ids:
            raise ValueError("Selected and effective routes must share expert_ids.")
        if (
            self.capacity.routes.selected_indices.shape
            != self.selected_routes.selected_indices.shape
        ):
            raise ValueError("Selected and effective routes must share token shape.")
        token_shape = self.selected_routes.selected_indices.shape
        if self.output.shape[: len(token_shape)] != token_shape:
            raise ValueError(
                "output must preserve the selected route token-leading shape."
            )


class AnchoredTop1Dispatcher:
    """Execute one anchor or specialist MLP per token and reconstruct outputs.

    Given hidden states ``h[..., H]`` and calibrated specialist scores
    ``G[..., K]``, dispatch performs:

    1. fixed-zero anchored top-1 selection;
    2. overflow-to-anchor capacity fallback;
    3. stable candidate-order grouping;
    4. one sparse MLP call for each non-empty route;
    5. differentiable reconstruction in original token order.

    The anchor is candidate zero and ``experts[i]`` is candidate ``i + 1``.
    No token is dropped. Additional positional and keyword arguments are passed
    unchanged to each invoked module; callers must therefore use only global
    or broadcast-compatible auxiliary arguments, not token-aligned tensors
    that would also require sparse indexing.

    This dispatcher is infrastructure and is not derived from a paper.
    Top-2 combination, package installation, and capacity estimation are
    deliberately excluded.
    """

    def __init__(
        self,
        *,
        selection_policy: AnchoredTop1Selection | None = None,
        capacity_policy: OverflowToAnchorCapacity | None = None,
    ) -> None:
        """Create a dispatcher from independently configurable policies."""

        self.selection_policy = selection_policy or AnchoredTop1Selection()
        self.capacity_policy = capacity_policy or OverflowToAnchorCapacity()
        if not isinstance(self.selection_policy, AnchoredTop1Selection):
            raise TypeError("selection_policy must be AnchoredTop1Selection.")
        if not isinstance(self.capacity_policy, OverflowToAnchorCapacity):
            raise TypeError(
                "capacity_policy must be OverflowToAnchorCapacity."
            )

    def dispatch(
        self,
        hidden_states: Tensor,
        *,
        expert_scores: Tensor,
        expert_ids: Sequence[str],
        anchor_mlp: nn.Module,
        experts: Sequence[nn.Module],
        expert_availability: Tensor | None = None,
        module_args: Sequence[Any] = (),
        module_kwargs: dict[str, Any] | None = None,
    ) -> Top1DispatchResult:
        """Select, execute, and reconstruct one routed MLP output.

        Args:
            hidden_states:
                Floating-point pre-MLP states with shape ``[..., hidden_size]``.
            expert_scores:
                Calibrated scores with shape ``hidden_states.shape[:-1] + (K,)``.
            expert_ids:
                Stable identifier order for scores and expert modules.
            anchor_mlp:
                Immutable dense fallback transformation.
            experts:
                Expert modules in exact ``expert_ids`` order.
            expert_availability:
                Optional boolean capacity mask with score shape.
            module_args:
                Additional arguments forwarded unchanged to invoked MLPs.
            module_kwargs:
                Additional keyword arguments forwarded unchanged to invoked
                MLPs. ``None`` is treated as an empty mapping.

        Returns:
            A :class:`Top1DispatchResult` containing the reconstructed output
            and both pre- and post-capacity route decisions.

        Raises:
            TypeError:
                If tensors, modules, or forwarding containers are invalid.
            ValueError:
                If tensor shapes, module counts, outputs, or devices conflict.
        """

        _validate_dispatch_inputs(
            hidden_states=hidden_states,
            expert_scores=expert_scores,
            expert_ids=expert_ids,
            anchor_mlp=anchor_mlp,
            experts=experts,
            module_args=module_args,
            module_kwargs=module_kwargs,
        )
        normalized_ids = tuple(expert_ids)
        normalized_experts = tuple(experts)
        forwarded_kwargs = {} if module_kwargs is None else dict(module_kwargs)

        selected_routes = self.selection_policy(
            expert_scores,
            expert_ids=normalized_ids,
        )
        capacity = self.capacity_policy(
            selected_routes,
            expert_availability=expert_availability,
        )
        effective_indices = capacity.routes.selected_indices

        hidden_size = hidden_states.shape[-1]
        token_count = effective_indices.numel()
        flat_hidden = hidden_states.reshape(token_count, hidden_size)
        flat_routes = effective_indices.reshape(token_count)
        candidates = (anchor_mlp, *normalized_experts)

        routed_outputs: list[tuple[Tensor, Tensor]] = []
        output_tail: tuple[int, ...] | None = None
        output_dtype: torch.dtype | None = None
        output_device: torch.device | None = None
        for candidate_index, module in enumerate(candidates):
            token_indices = torch.nonzero(
                flat_routes == candidate_index,
                as_tuple=False,
            ).flatten()
            if token_indices.numel() == 0:
                continue

            candidate_hidden = torch.index_select(
                flat_hidden,
                dim=0,
                index=token_indices,
            )
            candidate_output = module(
                candidate_hidden,
                *module_args,
                **forwarded_kwargs,
            )
            if not isinstance(candidate_output, Tensor):
                raise TypeError(
                    "Routed MLP candidates must return a torch.Tensor."
                )
            if candidate_output.ndim < 1:
                raise ValueError(
                    "Routed MLP candidate output must have a token dimension."
                )
            if candidate_output.shape[0] != token_indices.numel():
                raise ValueError(
                    "Routed MLP candidate changed the selected token count."
                )
            candidate_tail = tuple(candidate_output.shape[1:])
            if output_tail is None:
                output_tail = candidate_tail
                output_dtype = candidate_output.dtype
                output_device = candidate_output.device
            elif (
                candidate_tail != output_tail
                or candidate_output.dtype != output_dtype
                or candidate_output.device != output_device
            ):
                raise ValueError(
                    "Anchor and expert outputs must share trailing shape, "
                    "dtype, and device."
                )
            routed_outputs.append((token_indices, candidate_output))

        if output_tail is None or output_dtype is None or output_device is None:
            # A zero-token input cannot reveal the candidate output shape
            # without executing an artificial token, which could change module
            # semantics. Transformer batches are expected to contain tokens.
            raise ValueError("Cannot dispatch an input with zero tokens.")

        flat_output = torch.empty(
            (token_count, *output_tail),
            dtype=output_dtype,
            device=output_device,
        )
        # index_copy is differentiable with respect to each sparse candidate
        # output and restores original token order without executing inactive
        # experts.
        for token_indices, candidate_output in routed_outputs:
            flat_output = torch.index_copy(
                flat_output,
                dim=0,
                index=token_indices,
                source=candidate_output,
            )
        output = flat_output.reshape(*effective_indices.shape, *output_tail)
        return Top1DispatchResult(
            output=output,
            selected_routes=selected_routes,
            capacity=capacity,
        )


def _validate_dispatch_inputs(
    *,
    hidden_states: Tensor,
    expert_scores: Tensor,
    expert_ids: Sequence[str],
    anchor_mlp: nn.Module,
    experts: Sequence[nn.Module],
    module_args: Sequence[Any],
    module_kwargs: dict[str, Any] | None,
) -> None:
    """Validate dispatch structure before any candidate module executes."""

    if not isinstance(hidden_states, Tensor):
        raise TypeError("hidden_states must be a torch.Tensor.")
    if not hidden_states.is_floating_point():
        raise TypeError("hidden_states must use a floating-point dtype.")
    if hidden_states.ndim < 1:
        raise ValueError("hidden_states must have a final feature dimension.")
    if hidden_states.numel() == 0 or hidden_states.shape[-1] == 0:
        raise ValueError("hidden_states must contain at least one token and feature.")
    if not isinstance(expert_scores, Tensor):
        raise TypeError("expert_scores must be a torch.Tensor.")
    expected_score_shape = hidden_states.shape[:-1] + (len(expert_ids),)
    if expert_scores.shape != expected_score_shape:
        raise ValueError(
            "expert_scores must have the hidden-state token shape plus one "
            f"expert dimension: {tuple(expert_scores.shape)} != "
            f"{tuple(expected_score_shape)}."
        )
    if expert_scores.device != hidden_states.device:
        raise ValueError("expert_scores and hidden_states must share one device.")
    if not isinstance(anchor_mlp, nn.Module):
        raise TypeError("anchor_mlp must be a torch.nn.Module.")
    if isinstance(experts, (str, bytes)) or not isinstance(experts, Sequence):
        raise TypeError("experts must be a sequence of torch.nn.Module objects.")
    if len(experts) != len(expert_ids):
        raise ValueError("experts and expert_ids must have the same length.")
    if not all(isinstance(expert, nn.Module) for expert in experts):
        raise TypeError("experts must contain only torch.nn.Module objects.")
    if isinstance(module_args, (str, bytes)) or not isinstance(
        module_args,
        Sequence,
    ):
        raise TypeError("module_args must be a sequence.")
    if module_kwargs is not None and not isinstance(module_kwargs, dict):
        raise TypeError("module_kwargs must be a dict or None.")
