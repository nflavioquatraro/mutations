"""Sparse token dispatch for anchored top-1 and pair-shaped top-2 routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from anchored_moe.experts.base import AnchorRelativeExpert
from anchored_moe.experts.low_rank import PEFTLoRAExpert
from anchored_moe.policies.capacity import (
    AdaptiveToAnchorCapacity,
    CapacityDecision,
    OverflowToAnchorCapacity,
    Top2CapacityDecision,
    Top2ToTop1Capacity,
)
from anchored_moe.policies.combination import (
    AlwaysOnAnchorDeltaCombination,
    CombinationPolicy,
    ScoreSoftmaxCombination,
)
from anchored_moe.policies.selection import (
    AdaptiveRouteDecision,
    AdaptiveTop1Top2Selection,
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    RouteDecision,
    Top2SelectionPolicy,
    Top2RouteDecision,
)
from anchored_moe.policies.parameter_merging import LoRAParameterSpaceMerge


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


@dataclass(frozen=True)
class Top2DispatchResult:
    """Output and auditable metadata from capacity-aware top-2 dispatch.

    Attributes:
        output:
            Reconstructed MLP output with the same token-leading shape as the
            input hidden states.
        selected_routes:
            Ranked pair and anchor-veto decision produced before capacity.
        capacity:
            BAL-002 effective anchor/top-1/top-2 routes and overflow metadata.
        mixture_weights:
            Tensor with the selected-pair shape. Effective top-2 tokens contain
            configured combination weights, degraded top-1 tokens contain
            ``[1, 0]``, and anchor tokens contain ``[0, 0]``.
    """

    output: Tensor
    selected_routes: Top2RouteDecision
    capacity: Top2CapacityDecision
    mixture_weights: Tensor

    def __post_init__(self) -> None:
        """Validate output, route, and specialist-weight invariants."""

        if not isinstance(self.output, Tensor):
            raise TypeError("output must be a torch.Tensor.")
        if not self.output.is_floating_point():
            raise TypeError("output must use a floating-point dtype.")
        if not isinstance(self.selected_routes, Top2RouteDecision):
            raise TypeError("selected_routes must be a Top2RouteDecision.")
        _validate_top2_capacity_result(self.capacity, self.selected_routes)
        if not isinstance(self.mixture_weights, Tensor):
            raise TypeError("mixture_weights must be a torch.Tensor.")
        if not self.mixture_weights.is_floating_point():
            raise TypeError("mixture_weights must use a floating-point dtype.")
        if self.mixture_weights.shape != self.selected_routes.selected_scores.shape:
            raise ValueError(
                "mixture_weights must have the selected top-2 pair shape."
            )
        if self.mixture_weights.dtype != self.selected_routes.selected_scores.dtype:
            raise ValueError(
                "mixture_weights and selected scores must share one dtype."
            )
        if self.mixture_weights.device != self.selected_routes.selected_scores.device:
            raise ValueError(
                "mixture_weights and selected scores must share one device."
            )
        if self.output.dtype != self.mixture_weights.dtype:
            raise ValueError("output and mixture_weights must share one dtype.")
        if self.output.device != self.mixture_weights.device:
            raise ValueError("output and mixture_weights must share one device.")
        if not torch.isfinite(self.mixture_weights).all().item():
            raise ValueError("mixture_weights must contain only finite values.")
        if torch.any(self.mixture_weights < 0).item():
            raise ValueError("mixture_weights must be non-negative.")

        token_shape = self.selected_routes.selected_indices.shape[:-1]
        if self.output.shape[: len(token_shape)] != token_shape:
            raise ValueError(
                "output must preserve the selected route token-leading shape."
            )
        flat_weights = self.mixture_weights.reshape(-1, 2)
        flat_anchor_mask = self.capacity.anchor_mask.reshape(-1)
        flat_top1_mask = self.capacity.top1_mask.reshape(-1)
        flat_top2_mask = self.capacity.top2_mask.reshape(-1)
        if torch.any(flat_weights[flat_anchor_mask] != 0).item():
            raise ValueError(
                "effective anchor tokens must have zero specialist weights."
            )
        if flat_top1_mask.any().item():
            expected_top1_weights = torch.tensor(
                [1.0, 0.0],
                dtype=flat_weights.dtype,
                device=flat_weights.device,
            ).expand(int(flat_top1_mask.sum().item()), 2)
            if not torch.equal(
                flat_weights[flat_top1_mask],
                expected_top1_weights,
            ):
                raise ValueError(
                    "effective top-1 tokens must have weights [1, 0]."
                )
        if flat_top2_mask.any().item():
            pair_sums = flat_weights[flat_top2_mask].sum(dim=-1)
            tolerance = max(
                1e-6,
                8 * torch.finfo(self.mixture_weights.dtype).eps,
            )
            if not torch.allclose(
                pair_sums,
                torch.ones_like(pair_sums),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "approved specialist mixture weights must sum to one."
                )


@dataclass(frozen=True)
class AnchorDeltaDispatchResult:
    """Output and route metadata from always-on-anchor delta execution.

    The live anchor executes for every token. ``capacity`` records the
    effective zero-, one-, or two-delta route after availability fallback, and
    ``delta_weights`` records the exact specialist coefficients applied to the
    anchor activation.
    """

    output: Tensor
    selected_routes: RouteDecision | AdaptiveRouteDecision
    capacity: Top2CapacityDecision
    delta_weights: Tensor

    def __post_init__(self) -> None:
        """Validate output, route, capacity, and pair-weight alignment."""

        if not isinstance(self.output, Tensor) or not self.output.is_floating_point():
            raise TypeError("output must be a floating-point torch.Tensor.")
        if not isinstance(
            self.selected_routes,
            (RouteDecision, AdaptiveRouteDecision),
        ):
            raise TypeError(
                "selected_routes must be RouteDecision or AdaptiveRouteDecision."
            )
        if not isinstance(self.capacity, Top2CapacityDecision):
            raise TypeError("capacity must be a Top2CapacityDecision.")
        if self.capacity.expert_ids != self.selected_routes.expert_ids:
            raise ValueError("Selected and effective routes must share expert_ids.")
        token_shape = self.capacity.route_arity.shape
        if not isinstance(self.delta_weights, Tensor):
            raise TypeError("delta_weights must be a torch.Tensor.")
        if self.delta_weights.shape != token_shape + (2,):
            raise ValueError("delta_weights must have the capacity pair shape.")
        if self.output.shape[: len(token_shape)] != token_shape:
            raise ValueError("output must preserve the capacity token shape.")
        if self.output.dtype != self.delta_weights.dtype:
            raise ValueError("output and delta_weights must share one dtype.")
        if self.output.device != self.delta_weights.device:
            raise ValueError("output and delta_weights must share one device.")


@dataclass(frozen=True)
class ParameterMergeDispatchResult:
    """Output and route metadata from COM-005 LoRA parameter merging.

    ``merge_weights`` has the capacity pair shape. Anchor rows contain
    ``[0, 0]``, degraded top-1 rows contain ``[1, 0]``, and top-2 rows contain
    the exact frozen COM-003 coefficients applied in selected expert order.
    """

    output: Tensor
    selected_routes: Top2RouteDecision
    capacity: Top2CapacityDecision
    merge_weights: Tensor

    def __post_init__(self) -> None:
        """Validate output, selection, capacity, and coefficient alignment."""

        if not isinstance(self.output, Tensor) or not self.output.is_floating_point():
            raise TypeError("output must be a floating-point torch.Tensor.")
        if not isinstance(self.selected_routes, Top2RouteDecision):
            raise TypeError("selected_routes must be a Top2RouteDecision.")
        if not isinstance(self.capacity, Top2CapacityDecision):
            raise TypeError("capacity must be a Top2CapacityDecision.")
        if self.capacity.expert_ids != self.selected_routes.expert_ids:
            raise ValueError("Selected and effective routes must share expert_ids.")
        token_shape = self.capacity.route_arity.shape
        if not isinstance(self.merge_weights, Tensor):
            raise TypeError("merge_weights must be a torch.Tensor.")
        if self.merge_weights.shape != token_shape + (2,):
            raise ValueError("merge_weights must have the capacity pair shape.")
        if self.output.shape[: len(token_shape)] != token_shape:
            raise ValueError("output must preserve the capacity token shape.")
        if (
            self.output.dtype != self.merge_weights.dtype
            or self.output.device != self.merge_weights.device
        ):
            raise ValueError(
                "output and merge_weights must share one dtype and device."
            )
        if (
            not self.merge_weights.is_floating_point()
            or not torch.isfinite(self.merge_weights).all().item()
            or torch.any(self.merge_weights < 0).item()
        ):
            raise ValueError(
                "merge_weights must be finite non-negative floating-point values."
            )
        flat_arity = self.capacity.route_arity.reshape(-1)
        flat_weights = self.merge_weights.reshape(-1, 2)
        if not torch.equal(
            flat_weights[flat_arity == 0],
            torch.zeros_like(flat_weights[flat_arity == 0]),
        ):
            raise ValueError("Anchor rows must have merge weights [0, 0].")
        if (flat_arity == 1).any().item():
            expected_top1 = flat_weights.new_tensor([1.0, 0.0]).expand(
                int((flat_arity == 1).sum().item()),
                2,
            )
            if not torch.equal(flat_weights[flat_arity == 1], expected_top1):
                raise ValueError("Top-1 rows must have merge weights [1, 0].")
        if (flat_arity == 2).any().item():
            sums = flat_weights[flat_arity == 2].sum(dim=-1)
            tolerance = max(
                1e-6,
                8 * torch.finfo(self.merge_weights.dtype).eps,
            )
            if not torch.allclose(
                sums,
                torch.ones_like(sums),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("Top-2 merge weights must sum to one.")


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


class AnchoredTop2Dispatcher:
    """Execute capacity-aware pair routes and reconstruct sparse activations.

    Given hidden states ``h[..., H]`` and calibrated specialist scores
    ``G[..., K]``, INF-002 performs:

    1. one configured :class:`Top2SelectionPolicy`;
    2. BAL-002 top-2 to top-1 or anchor degradation;
    3. one sparse anchor call for all effective anchor tokens;
    4. one sparse call per capacity-approved specialist;
    5. direct top-1 output or configured top-2 activation combination;
    6. differentiable reconstruction in original token order.

    The anchor is candidate zero and ``experts[i]`` is candidate ``i + 1``.
    No token is dropped. Inactive or unavailable specialists are not executed,
    and an active specialist is called once for all of its tokens regardless
    of whether it was originally ranked first or second.

    Additional positional and keyword arguments are forwarded unchanged to
    each invoked module. Callers must therefore use only global or
    broadcast-compatible auxiliary arguments, not token-aligned tensors that
    would also require sparse indexing.

    This dispatcher is framework infrastructure and is not derived from a
    paper. Pair compatibility is enforced only when the configured selection
    policy declares it. Availability is supplied explicitly; capacity
    estimation, telemetry, package attachment, and distributed execution are
    deliberately excluded.
    """

    def __init__(
        self,
        *,
        selection_policy: Top2SelectionPolicy | None = None,
        combination_policy: CombinationPolicy | None = None,
        capacity_policy: Top2ToTop1Capacity | None = None,
    ) -> None:
        """Create a dispatcher from independently configurable policies."""

        self.selection_policy = (
            AnchorVetoTop2Selection()
            if selection_policy is None
            else selection_policy
        )
        self.combination_policy = (
            ScoreSoftmaxCombination()
            if combination_policy is None
            else combination_policy
        )
        self.capacity_policy = (
            Top2ToTop1Capacity()
            if capacity_policy is None
            else capacity_policy
        )
        if not isinstance(self.selection_policy, Top2SelectionPolicy):
            raise TypeError(
                "selection_policy must implement Top2SelectionPolicy."
            )
        if not isinstance(self.combination_policy, CombinationPolicy):
            raise TypeError(
                "combination_policy must implement CombinationPolicy."
            )
        if not isinstance(self.capacity_policy, Top2ToTop1Capacity):
            raise TypeError("capacity_policy must be Top2ToTop1Capacity.")

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
    ) -> Top2DispatchResult:
        """Select, capacity-check, execute, and reconstruct sparse outputs.

        Args:
            hidden_states:
                Floating-point pre-MLP states with shape ``[..., hidden_size]``.
            expert_scores:
                Calibrated scores with shape
                ``hidden_states.shape[:-1] + (K,)`` and the hidden-state dtype.
            expert_ids:
                Stable identifier order for scores and expert modules.
            anchor_mlp:
                Immutable dense fallback transformation.
            experts:
                Expert modules in exact ``expert_ids`` order.
            expert_availability:
                Optional boolean tensor with shape
                ``hidden_states.shape[:-1] + (K,)``. Omitting it declares
                every specialist available.
            module_args:
                Additional arguments forwarded unchanged to invoked MLPs.
            module_kwargs:
                Additional keyword arguments forwarded unchanged to invoked
                MLPs. ``None`` is treated as an empty mapping.

        Returns:
            A :class:`Top2DispatchResult` containing reconstructed output,
            ranked routes, and full-shape specialist mixture weights.

        Raises:
            TypeError:
                If tensors, modules, policies, or forwarding containers are
                invalid.
            ValueError:
                If tensor shapes, dtypes, module counts, outputs, or devices
                conflict, or a selection implementation violates the
                pair-shaped policy contract.
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
        if not expert_scores.is_floating_point():
            raise TypeError("expert_scores must use a floating-point dtype.")
        if expert_scores.dtype != hidden_states.dtype:
            raise ValueError(
                "expert_scores and hidden_states must share one dtype for "
                "activation combination."
            )

        normalized_ids = tuple(expert_ids)
        normalized_experts = tuple(experts)
        forwarded_kwargs = {} if module_kwargs is None else dict(module_kwargs)
        selected_routes = self.selection_policy(
            expert_scores,
            expert_ids=normalized_ids,
        )
        _validate_top2_selection_result(
            selected_routes,
            expert_scores=expert_scores,
            expert_ids=normalized_ids,
        )
        capacity = self.capacity_policy(
            selected_routes,
            expert_availability=expert_availability,
        )
        # A composed policy is validated before any MLP executes. In
        # particular, capacity cannot introduce an unselected specialist.
        _validate_top2_capacity_result(capacity, selected_routes)

        token_shape = selected_routes.selected_indices.shape[:-1]
        token_count = capacity.route_arity.numel()
        hidden_size = hidden_states.shape[-1]
        flat_hidden = hidden_states.reshape(token_count, hidden_size)
        flat_effective = capacity.effective_indices.reshape(token_count, 2)
        flat_arity = capacity.route_arity.reshape(token_count)
        anchor_indices = torch.nonzero(
            flat_arity == 0,
            as_tuple=False,
        ).flatten()
        top1_indices = torch.nonzero(
            flat_arity == 1,
            as_tuple=False,
        ).flatten()
        top2_indices = torch.nonzero(
            flat_arity == 2,
            as_tuple=False,
        ).flatten()

        anchor_output: tuple[Tensor, Tensor] | None = None
        specialist_outputs: list[tuple[Tensor, Tensor]] = []
        output_tail: tuple[int, ...] | None = None
        output_dtype: torch.dtype | None = None
        output_device: torch.device | None = None

        if anchor_indices.numel():
            candidate_hidden = torch.index_select(
                flat_hidden,
                dim=0,
                index=anchor_indices,
            )
            candidate_output = anchor_mlp(
                candidate_hidden,
                *module_args,
                **forwarded_kwargs,
            )
            output_tail, output_dtype, output_device = _validate_candidate_output(
                candidate_output,
                expected_tokens=anchor_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            anchor_output = (anchor_indices, candidate_output)

        for candidate_index, expert in enumerate(normalized_experts, start=1):
            active_occurrences = torch.nonzero(
                flat_effective == candidate_index,
                as_tuple=False,
            )
            if active_occurrences.numel() == 0:
                continue

            token_indices = active_occurrences[:, 0]
            pair_slots = active_occurrences[:, 1]
            candidate_hidden = torch.index_select(
                flat_hidden,
                dim=0,
                index=token_indices,
            )
            candidate_output = expert(
                candidate_hidden,
                *module_args,
                **forwarded_kwargs,
            )
            output_tail, output_dtype, output_device = _validate_candidate_output(
                candidate_output,
                expected_tokens=token_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            # Packing always promotes a lone survivor to slot zero, while
            # retained pairs preserve their original first/second ordering.
            flat_pair_positions = token_indices * 2 + pair_slots
            specialist_outputs.append((flat_pair_positions, candidate_output))

        if output_tail is None or output_dtype is None or output_device is None:
            raise ValueError("Cannot dispatch an input with zero tokens.")
        if output_dtype != expert_scores.dtype:
            raise ValueError(
                "Anchor and expert outputs must preserve the router-score dtype "
                "for activation combination."
            )
        if output_device != expert_scores.device:
            raise ValueError(
                "Anchor and expert outputs must preserve the router-score device "
                "for activation combination."
            )

        flat_output = torch.empty(
            (token_count, *output_tail),
            dtype=output_dtype,
            device=output_device,
        )
        if anchor_output is not None:
            flat_output = torch.index_copy(
                flat_output,
                dim=0,
                index=anchor_output[0],
                source=anchor_output[1],
            )

        flat_mixture_weights = expert_scores.new_zeros((token_count, 2))
        active_slot_count = int(flat_arity.sum().item())
        actual_active_slots = sum(
            int(indices.numel()) for indices, _ in specialist_outputs
        )
        if actual_active_slots != active_slot_count:
            raise RuntimeError(
                "Capacity-approved specialist routes were not completely executed."
            )

        if active_slot_count:
            flat_specialist_output = torch.empty(
                (token_count * 2, *output_tail),
                dtype=output_dtype,
                device=output_device,
            )
            for pair_positions, candidate_output in specialist_outputs:
                flat_specialist_output = torch.index_copy(
                    flat_specialist_output,
                    dim=0,
                    index=pair_positions,
                    source=candidate_output,
                )

        if top1_indices.numel():
            top1_outputs = torch.index_select(
                flat_specialist_output,
                dim=0,
                index=top1_indices * 2,
            )
            flat_output = torch.index_copy(
                flat_output,
                dim=0,
                index=top1_indices,
                source=top1_outputs,
            )
            top1_weights = expert_scores.new_tensor([1.0, 0.0]).expand(
                top1_indices.numel(),
                2,
            )
            flat_mixture_weights = torch.index_copy(
                flat_mixture_weights,
                dim=0,
                index=top1_indices,
                source=top1_weights,
            )

        if top2_indices.numel():
            top2_positions = (
                top2_indices.unsqueeze(-1) * 2
                + torch.arange(2, device=top2_indices.device)
            ).reshape(-1)
            selected_outputs = torch.index_select(
                flat_specialist_output,
                dim=0,
                index=top2_positions,
            ).reshape(
                top2_indices.numel(),
                2,
                *output_tail,
            )
            approved_scores = capacity.effective_scores.reshape(
                token_count,
                2,
            )[flat_arity == 2]
            combination = self.combination_policy(
                selected_outputs,
                selected_scores=approved_scores,
            )
            flat_output = torch.index_copy(
                flat_output,
                dim=0,
                index=top2_indices,
                source=combination.output,
            )
            flat_mixture_weights = torch.index_copy(
                flat_mixture_weights,
                dim=0,
                index=top2_indices,
                source=combination.weights,
            )

        output = flat_output.reshape(*token_shape, *output_tail)
        mixture_weights = flat_mixture_weights.reshape(*token_shape, 2)
        return Top2DispatchResult(
            output=output,
            selected_routes=selected_routes,
            capacity=capacity,
            mixture_weights=mixture_weights,
        )

    def dispatch_preselected(
        self,
        hidden_states: Tensor,
        *,
        selected_routes: Top2RouteDecision,
        anchor_mlp: nn.Module,
        experts: Sequence[nn.Module],
        expert_availability: Tensor | None = None,
        module_args: Sequence[Any] = (),
        module_kwargs: dict[str, Any] | None = None,
    ) -> Top2DispatchResult:
        """Execute an already selected pair route through the INF-002 runtime.

        This entry point supports staged selectors such as POL-004, whose
        package and local scores cannot be represented by the flat
        ``expert_scores`` input of :meth:`dispatch`. Selection remains
        external, while capacity degradation, sparse module execution,
        activation combination, and reconstruction are exactly the ordinary
        INF-002 path.

        ``selected_routes.selected_scores`` provides the local mixture scores.
        A shape-only zero score bank is allocated internally so the established
        dispatcher validation and output dtype/device contracts remain shared.
        """

        if not isinstance(selected_routes, Top2RouteDecision):
            raise TypeError("selected_routes must be a Top2RouteDecision.")
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.ndim < 1:
            raise ValueError("hidden_states must have a final hidden dimension.")
        placeholder_scores = hidden_states.new_zeros(
            hidden_states.shape[:-1] + (len(selected_routes.expert_ids),)
        )
        dispatcher = AnchoredTop2Dispatcher(
            selection_policy=_PreselectedTop2Selection(selected_routes),
            combination_policy=self.combination_policy,
            capacity_policy=self.capacity_policy,
        )
        return dispatcher.dispatch(
            hidden_states,
            expert_scores=placeholder_scores,
            expert_ids=selected_routes.expert_ids,
            anchor_mlp=anchor_mlp,
            experts=experts,
            expert_availability=expert_availability,
            module_args=module_args,
            module_kwargs=module_kwargs,
        )


class AnchoredLoRAParameterMergeDispatcher:
    r"""Select two LoRA experts and execute their merged effective parameters.

    The dispatcher composes an existing pair selector and BAL-002 capacity
    policy with COM-005. It supports exactly two installed PEFT/LoRA experts:

    - anchor routes execute the live anchor directly;
    - degraded top-1 routes use one unit-weight effective LoRA delta;
    - top-2 routes use frozen COM-003 coefficients in selected order.

    The selected expert modules are never called. Instead,
    :class:`LoRAParameterSpaceMerge` applies their effective projection deltas
    to the live anchor through a correctness-first per-token functional
    execution. No token is dropped and no registered parameter is mutated.
    """

    def __init__(
        self,
        *,
        merge_policy: LoRAParameterSpaceMerge,
        selection_policy: Top2SelectionPolicy | None = None,
        capacity_policy: Top2ToTop1Capacity | None = None,
    ) -> None:
        """Create the pair-only COM-005 inference runtime."""

        if not isinstance(merge_policy, LoRAParameterSpaceMerge):
            raise TypeError("merge_policy must be LoRAParameterSpaceMerge.")
        self.merge_policy = merge_policy
        self.selection_policy = selection_policy or AnchorVetoTop2Selection()
        self.capacity_policy = capacity_policy or Top2ToTop1Capacity()
        if not isinstance(self.selection_policy, Top2SelectionPolicy):
            raise TypeError(
                "selection_policy must implement Top2SelectionPolicy."
            )
        if not isinstance(self.capacity_policy, Top2ToTop1Capacity):
            raise TypeError("capacity_policy must be Top2ToTop1Capacity.")

    def dispatch(
        self,
        hidden_states: Tensor,
        *,
        expert_scores: Tensor,
        expert_ids: Sequence[str],
        anchor_mlp: nn.Module,
        experts: Sequence[PEFTLoRAExpert],
        expert_availability: Tensor | None = None,
        module_args: Sequence[Any] = (),
        module_kwargs: dict[str, Any] | None = None,
    ) -> ParameterMergeDispatchResult:
        """Select, capacity-check, merge effective deltas, and reconstruct."""

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
        if len(normalized_experts) != 2:
            raise ValueError(
                "COM-005 routed execution requires exactly two experts."
            )
        if not all(
            isinstance(expert, PEFTLoRAExpert)
            for expert in normalized_experts
        ):
            raise TypeError(
                "COM-005 routed execution requires PEFTLoRAExpert objects."
            )
        if expert_scores.dtype != hidden_states.dtype:
            raise ValueError(
                "expert_scores and hidden_states must share one dtype."
            )
        self.merge_policy.validate_compatibility(
            anchor_mlp,
            normalized_experts,
        )

        selected_routes = self.selection_policy(
            expert_scores,
            expert_ids=normalized_ids,
        )
        _validate_top2_selection_result(
            selected_routes,
            expert_scores=expert_scores,
            expert_ids=normalized_ids,
        )
        capacity = self.capacity_policy(
            selected_routes,
            expert_availability=expert_availability,
        )
        _validate_top2_capacity_result(capacity, selected_routes)

        token_shape = selected_routes.selected_indices.shape[:-1]
        token_count = capacity.route_arity.numel()
        if token_count == 0:
            raise ValueError("Cannot dispatch an input with zero tokens.")
        flat_hidden = hidden_states.reshape(token_count, hidden_states.shape[-1])
        flat_effective = capacity.effective_indices.reshape(token_count, 2)
        flat_scores = capacity.effective_scores.reshape(token_count, 2)
        flat_arity = capacity.route_arity.reshape(token_count)
        forwarded_kwargs = {} if module_kwargs is None else dict(module_kwargs)

        executed: list[tuple[Tensor, Tensor]] = []
        flat_merge_weights = expert_scores.new_zeros((token_count, 2))
        output_tail: tuple[int, ...] | None = None
        output_dtype: torch.dtype | None = None
        output_device: torch.device | None = None

        top2_groups: list[tuple[Tensor, tuple[int, int], Tensor]] = []
        for first_index, second_index in ((1, 2), (2, 1)):
            top2_indices = torch.nonzero(
                (flat_arity == 2)
                & (flat_effective[:, 0] == first_index)
                & (flat_effective[:, 1] == second_index),
                as_tuple=False,
            ).flatten()
            if top2_indices.numel():
                approved_scores = torch.index_select(
                    flat_scores,
                    0,
                    top2_indices,
                )
                # Validate frozen-head dtype/device and numerical behavior
                # before the first anchor or merged candidate executes.
                pair_weights = self.merge_policy.pair_weights(approved_scores)
                top2_groups.append(
                    (
                        top2_indices,
                        (first_index, second_index),
                        pair_weights,
                    )
                )

        anchor_indices = torch.nonzero(flat_arity == 0, as_tuple=False).flatten()
        if anchor_indices.numel():
            anchor_output = anchor_mlp(
                torch.index_select(flat_hidden, 0, anchor_indices),
                *module_args,
                **forwarded_kwargs,
            )
            output_tail, output_dtype, output_device = _validate_candidate_output(
                anchor_output,
                expected_tokens=anchor_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            executed.append((anchor_indices, anchor_output))

        for candidate_index, expert in enumerate(normalized_experts, start=1):
            top1_indices = torch.nonzero(
                (flat_arity == 1) & (flat_effective[:, 0] == candidate_index),
                as_tuple=False,
            ).flatten()
            if not top1_indices.numel():
                continue
            top1_weights = expert_scores.new_ones((top1_indices.numel(), 1))
            merged = self.merge_policy.merge(
                torch.index_select(flat_hidden, 0, top1_indices),
                anchor_mlp=anchor_mlp,
                experts=(expert,),
                weights=top1_weights,
                module_args=module_args,
                module_kwargs=forwarded_kwargs,
            )
            output_tail, output_dtype, output_device = _validate_candidate_output(
                merged.output,
                expected_tokens=top1_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            executed.append((top1_indices, merged.output))
            full_top1_weights = expert_scores.new_tensor([1.0, 0.0]).expand(
                top1_indices.numel(),
                2,
            )
            flat_merge_weights = torch.index_copy(
                flat_merge_weights,
                0,
                top1_indices,
                full_top1_weights,
            )

        for top2_indices, selected_pair, pair_weights in top2_groups:
            first_index, second_index = selected_pair
            merged = self.merge_policy.merge(
                torch.index_select(flat_hidden, 0, top2_indices),
                anchor_mlp=anchor_mlp,
                experts=(
                    normalized_experts[first_index - 1],
                    normalized_experts[second_index - 1],
                ),
                weights=pair_weights,
                module_args=module_args,
                module_kwargs=forwarded_kwargs,
            )
            output_tail, output_dtype, output_device = _validate_candidate_output(
                merged.output,
                expected_tokens=top2_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            executed.append((top2_indices, merged.output))
            flat_merge_weights = torch.index_copy(
                flat_merge_weights,
                0,
                top2_indices,
                pair_weights,
            )

        executed_token_count = sum(int(indices.numel()) for indices, _ in executed)
        if executed_token_count != token_count:
            raise RuntimeError(
                "Capacity-approved parameter-merge routes were not completely "
                "executed."
            )
        if output_tail is None or output_dtype is None or output_device is None:
            raise RuntimeError("COM-005 execution produced no candidate output.")
        if output_dtype != expert_scores.dtype or output_device != expert_scores.device:
            raise ValueError(
                "Merged outputs and router scores must share dtype and device."
            )

        flat_output = torch.empty(
            (token_count, *output_tail),
            dtype=output_dtype,
            device=output_device,
        )
        for token_indices, candidate_output in executed:
            flat_output = torch.index_copy(
                flat_output,
                0,
                token_indices,
                candidate_output,
            )
        return ParameterMergeDispatchResult(
            output=flat_output.reshape(*token_shape, *output_tail),
            selected_routes=selected_routes,
            capacity=capacity,
            merge_weights=flat_merge_weights.reshape(*token_shape, 2),
        )


class AlwaysOnAnchorDeltaDispatcher:
    r"""Execute the anchor for every token and add sparse routed deltas.

    COM-004 composes either anchored top-1 or adaptive top-1/top-2 selection
    with availability fallback and
    :class:`AlwaysOnAnchorDeltaCombination`. The anchor executes once over the
    complete token batch. Each capacity-approved anchor-relative expert then
    executes only its selected rows and receives the matching already-computed
    anchor activation:

    ``y = A(h) + sum_k alpha_k Delta_k(h)``.

    No token is dropped. Anchor routes execute no expert and return the exact
    anchor activation. Additional MLP arguments must be global or
    broadcast-compatible because token-aligned auxiliary arguments are not
    sparsely indexed.
    """

    def __init__(
        self,
        *,
        selection_policy: (
            AnchoredTop1Selection | AdaptiveTop1Top2Selection | None
        ) = None,
        combination_policy: AlwaysOnAnchorDeltaCombination | None = None,
        top1_capacity_policy: OverflowToAnchorCapacity | None = None,
        adaptive_capacity_policy: AdaptiveToAnchorCapacity | None = None,
    ) -> None:
        """Create an always-on delta runtime from composable policies."""

        self.selection_policy = selection_policy or AdaptiveTop1Top2Selection()
        self.combination_policy = (
            combination_policy or AlwaysOnAnchorDeltaCombination()
        )
        self.top1_capacity_policy = (
            top1_capacity_policy or OverflowToAnchorCapacity()
        )
        self.adaptive_capacity_policy = (
            adaptive_capacity_policy or AdaptiveToAnchorCapacity()
        )
        if not isinstance(
            self.selection_policy,
            (AnchoredTop1Selection, AdaptiveTop1Top2Selection),
        ):
            raise TypeError(
                "selection_policy must be AnchoredTop1Selection or "
                "AdaptiveTop1Top2Selection."
            )
        if not isinstance(
            self.combination_policy,
            AlwaysOnAnchorDeltaCombination,
        ):
            raise TypeError(
                "combination_policy must be AlwaysOnAnchorDeltaCombination."
            )
        if not isinstance(self.top1_capacity_policy, OverflowToAnchorCapacity):
            raise TypeError(
                "top1_capacity_policy must be OverflowToAnchorCapacity."
            )
        if not isinstance(
            self.adaptive_capacity_policy,
            AdaptiveToAnchorCapacity,
        ):
            raise TypeError(
                "adaptive_capacity_policy must be AdaptiveToAnchorCapacity."
            )

    def dispatch(
        self,
        hidden_states: Tensor,
        *,
        expert_scores: Tensor,
        expert_ids: Sequence[str],
        anchor_mlp: nn.Module,
        experts: Sequence[AnchorRelativeExpert],
        expert_availability: Tensor | None = None,
        module_args: Sequence[Any] = (),
        module_kwargs: dict[str, Any] | None = None,
    ) -> AnchorDeltaDispatchResult:
        """Select deltas, execute sparse experts, and reconstruct outputs."""

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
        if not all(
            isinstance(expert, AnchorRelativeExpert)
            for expert in normalized_experts
        ):
            raise TypeError(
                "experts must implement the AnchorRelativeExpert interface."
            )
        if expert_scores.dtype != hidden_states.dtype:
            raise ValueError(
                "expert_scores and hidden_states must share one dtype for "
                "delta combination."
            )
        forwarded_kwargs = {} if module_kwargs is None else dict(module_kwargs)
        selected_routes = self.selection_policy(
            expert_scores,
            expert_ids=normalized_ids,
        )
        capacity = self._apply_capacity(
            selected_routes,
            expert_availability=expert_availability,
        )

        token_shape = capacity.route_arity.shape
        token_count = capacity.route_arity.numel()
        if token_count == 0:
            raise ValueError("Cannot dispatch an input with zero tokens.")
        hidden_size = hidden_states.shape[-1]
        flat_hidden = hidden_states.reshape(token_count, hidden_size)
        flat_effective = capacity.effective_indices.reshape(token_count, 2)

        # The anchor is deliberately dense in COM-004: it executes once for
        # every token, including tokens that also receive specialist deltas.
        flat_anchor_output = anchor_mlp(
            flat_hidden,
            *module_args,
            **forwarded_kwargs,
        )
        output_tail, output_dtype, output_device = _validate_candidate_output(
            flat_anchor_output,
            expected_tokens=token_count,
            expected_tail=None,
            expected_dtype=None,
            expected_device=None,
        )
        if output_dtype != expert_scores.dtype:
            raise ValueError(
                "Anchor output and router scores must share one dtype."
            )
        if output_device != expert_scores.device:
            raise ValueError(
                "Anchor output and router scores must share one device."
            )

        flat_selected_deltas = flat_anchor_output.new_zeros(
            (token_count * 2, *output_tail)
        )
        executed_slots = 0
        for candidate_index, expert in enumerate(normalized_experts, start=1):
            active_occurrences = torch.nonzero(
                flat_effective == candidate_index,
                as_tuple=False,
            )
            if active_occurrences.numel() == 0:
                continue
            token_indices = active_occurrences[:, 0]
            pair_slots = active_occurrences[:, 1]
            candidate_hidden = torch.index_select(
                flat_hidden,
                dim=0,
                index=token_indices,
            )
            candidate_anchor = torch.index_select(
                flat_anchor_output,
                dim=0,
                index=token_indices,
            )
            candidate_delta = expert.forward_delta(
                candidate_hidden,
                *module_args,
                anchor_output=candidate_anchor,
                **forwarded_kwargs,
            )
            _validate_candidate_output(
                candidate_delta,
                expected_tokens=token_indices.numel(),
                expected_tail=output_tail,
                expected_dtype=output_dtype,
                expected_device=output_device,
            )
            pair_positions = token_indices * 2 + pair_slots
            flat_selected_deltas = torch.index_copy(
                flat_selected_deltas,
                dim=0,
                index=pair_positions,
                source=candidate_delta,
            )
            executed_slots += int(pair_positions.numel())

        expected_slots = int(capacity.route_arity.sum().item())
        if executed_slots != expected_slots:
            raise RuntimeError(
                "Capacity-approved delta routes were not completely executed."
            )
        anchor_output = flat_anchor_output.reshape(*token_shape, *output_tail)
        selected_deltas = flat_selected_deltas.reshape(
            *token_shape,
            2,
            *output_tail,
        )
        combination = self.combination_policy(
            anchor_output,
            selected_deltas,
            selected_scores=capacity.effective_scores,
            route_arity=capacity.route_arity,
        )
        return AnchorDeltaDispatchResult(
            output=combination.output,
            selected_routes=selected_routes,
            capacity=capacity,
            delta_weights=combination.weights,
        )

    def _apply_capacity(
        self,
        routes: RouteDecision | AdaptiveRouteDecision,
        *,
        expert_availability: Tensor | None,
    ) -> Top2CapacityDecision:
        """Normalize top-1 and adaptive routes to one pair-shaped contract."""

        if isinstance(routes, AdaptiveRouteDecision):
            return self.adaptive_capacity_policy(
                routes,
                expert_availability=expert_availability,
            )
        if not isinstance(routes, RouteDecision):
            raise TypeError(
                "Delta selection must return RouteDecision or "
                "AdaptiveRouteDecision."
            )
        top1_capacity = self.top1_capacity_policy(
            routes,
            expert_availability=expert_availability,
        )
        effective = top1_capacity.routes
        effective_indices = torch.stack(
            (effective.selected_indices, torch.zeros_like(effective.selected_indices)),
            dim=-1,
        )
        effective_scores = torch.stack(
            (effective.selected_scores, torch.zeros_like(effective.selected_scores)),
            dim=-1,
        )
        overflow_mask = torch.stack(
            (
                top1_capacity.overflow_mask,
                torch.zeros_like(top1_capacity.overflow_mask),
            ),
            dim=-1,
        )
        return Top2CapacityDecision(
            effective_indices=effective_indices,
            effective_scores=effective_scores,
            route_arity=(effective.selected_indices != 0).to(dtype=torch.long),
            expert_ids=effective.expert_ids,
            overflow_mask=overflow_mask,
        )


class _PreselectedTop2Selection(Top2SelectionPolicy):
    """Return one immutable staged decision for dispatcher composition."""

    policy_name = "preselected_top2"

    def __init__(self, selected_routes: Top2RouteDecision) -> None:
        self.selected_routes = selected_routes

    def select(
        self,
        expert_scores: Tensor,
        *,
        expert_ids: Sequence[str],
    ) -> Top2RouteDecision:
        """Return the staged decision; the dispatcher validates its contract."""

        return self.selected_routes


def _validate_top2_selection_result(
    selected_routes: Top2RouteDecision,
    *,
    expert_scores: Tensor,
    expert_ids: tuple[str, ...],
) -> None:
    """Validate a composed top-2 policy before any candidate MLP executes."""

    if not isinstance(selected_routes, Top2RouteDecision):
        raise TypeError(
            "Top2SelectionPolicy must return a Top2RouteDecision."
        )
    expected_shape = expert_scores.shape[:-1] + (2,)
    if selected_routes.selected_indices.shape != expected_shape:
        raise ValueError(
            "Top-2 selection result must have the score token shape plus a "
            f"pair dimension: {tuple(selected_routes.selected_indices.shape)} "
            f"!= {tuple(expected_shape)}."
        )
    if selected_routes.expert_ids != expert_ids:
        raise ValueError(
            "Top-2 selection result expert_ids must preserve dispatcher order."
        )
    if selected_routes.selected_scores.dtype != expert_scores.dtype:
        raise ValueError(
            "Top-2 selected scores must preserve the router-score dtype."
        )
    if selected_routes.selected_scores.device != expert_scores.device:
        raise ValueError(
            "Top-2 selected scores must preserve the router-score device."
        )


def _validate_top2_capacity_result(
    capacity: Top2CapacityDecision,
    selected_routes: Top2RouteDecision,
) -> None:
    """Require capacity output to be an exact degradation of selected pairs."""

    if not isinstance(capacity, Top2CapacityDecision):
        raise TypeError("capacity policy must return a Top2CapacityDecision.")
    if capacity.expert_ids != selected_routes.expert_ids:
        raise ValueError("Selected and effective routes must share expert_ids.")
    if capacity.effective_indices.shape != selected_routes.selected_indices.shape:
        raise ValueError("Selected and effective routes must share pair shape.")

    approved_pair = selected_routes.expert_pair_mask.unsqueeze(-1)
    invalid_overflow = capacity.overflow_mask & ~approved_pair
    if invalid_overflow.any().item():
        raise ValueError(
            "Anchor-vetoed routes cannot contain specialist overflow."
        )

    active_slots = approved_pair & ~capacity.overflow_mask
    first_active = active_slots[..., 0]
    second_active = active_slots[..., 1]
    any_active = first_active | second_active
    both_active = first_active & second_active

    expected_indices = torch.zeros_like(selected_routes.selected_indices)
    expected_scores = torch.zeros_like(selected_routes.selected_scores)
    expected_indices[..., 0] = torch.where(
        any_active,
        torch.where(
            first_active,
            selected_routes.selected_indices[..., 0],
            selected_routes.selected_indices[..., 1],
        ),
        0,
    )
    expected_scores[..., 0] = torch.where(
        any_active,
        torch.where(
            first_active,
            selected_routes.selected_scores[..., 0],
            selected_routes.selected_scores[..., 1],
        ),
        0.0,
    )
    expected_indices[..., 1] = torch.where(
        both_active,
        selected_routes.selected_indices[..., 1],
        0,
    )
    expected_scores[..., 1] = torch.where(
        both_active,
        selected_routes.selected_scores[..., 1],
        0.0,
    )
    expected_arity = active_slots.sum(dim=-1, dtype=torch.long)
    if (
        not torch.equal(capacity.effective_indices, expected_indices)
        or not torch.equal(capacity.effective_scores, expected_scores)
        or not torch.equal(capacity.route_arity, expected_arity)
    ):
        raise ValueError(
            "Capacity output must exactly retain, degrade, or reject each "
            "selected specialist pair."
        )


def _validate_candidate_output(
    candidate_output: Any,
    *,
    expected_tokens: int,
    expected_tail: tuple[int, ...] | None,
    expected_dtype: torch.dtype | None,
    expected_device: torch.device | None,
) -> tuple[tuple[int, ...], torch.dtype, torch.device]:
    """Validate one sparse candidate output against the shared MLP contract."""

    if not isinstance(candidate_output, Tensor):
        raise TypeError("Routed MLP candidates must return a torch.Tensor.")
    if candidate_output.ndim < 2:
        raise ValueError(
            "Top-2 routed MLP candidate output must have token and feature "
            "dimensions."
        )
    if candidate_output.shape[0] != expected_tokens:
        raise ValueError("Routed MLP candidate changed the selected token count.")

    candidate_tail = tuple(candidate_output.shape[1:])
    if expected_tail is None:
        return candidate_tail, candidate_output.dtype, candidate_output.device
    if (
        candidate_tail != expected_tail
        or candidate_output.dtype != expected_dtype
        or candidate_output.device != expected_device
    ):
        raise ValueError(
            "Anchor and expert outputs must share trailing shape, dtype, "
            "and device."
        )
    return candidate_tail, candidate_output.dtype, candidate_output.device


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
