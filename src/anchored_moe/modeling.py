"""Model-family-neutral routed components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from anchored_moe.experts.base import Expert
from anchored_moe.inference.dispatch import AnchoredTop1Dispatcher
from anchored_moe.inference.telemetry import RouteTelemetry
from anchored_moe.policies.capacity import OverflowToAnchorCapacity
from anchored_moe.routers.base import RouterFeatureExtractor
from anchored_moe.routers.heads import AppendOnlyRouterHeads


class RoutedMLP(nn.Module):
    """Wrap an immutable dense MLP with explicit opt-in routing.

    The wrapper holds the original MLP object rather than copying it. Its
    foundation behavior is therefore exactly one call to the same anchor
    module; it performs no arithmetic, routing, or tensor reshaping around the
    call. :meth:`configure_anchored_top1` can attach independently constructed
    router features, affine heads, and experts for one routed layer. Until that
    explicit call, forced-anchor behavior remains exact.

    Args:
        anchor_mlp: Original dense MLP module owned by the converted layer.

    The routed path reserves the ``expert_availability`` keyword for a boolean
    tensor with shape ``hidden_states.shape[:-1] + (K,)``. It is consumed by
    BAL-001 rather than forwarded to MLP candidates. Other arguments are
    forwarded unchanged and must be global or broadcast-compatible because
    token-aligned auxiliary tensors are not sparsely indexed in INF-001.
    """

    def __init__(self, anchor_mlp: nn.Module) -> None:
        super().__init__()
        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if isinstance(anchor_mlp, RoutedMLP):
            raise ValueError("A RoutedMLP cannot wrap another RoutedMLP.")

        self.anchor_mlp = anchor_mlp
        # Freezing is mechanical and local: conversion changes no tensor value,
        # while every anchor parameter is made ineligible for optimization.
        self.anchor_mlp.requires_grad_(False)
        self.routing_mode = "forced_anchor"
        self.layer_index: int | None = None
        self.expert_ids: tuple[str, ...] = ()
        self.router_feature_extractor: RouterFeatureExtractor | None = None
        self.router_heads: AppendOnlyRouterHeads | None = None
        self.experts = nn.ModuleList()
        self._dispatcher: AnchoredTop1Dispatcher | None = None
        self._route_telemetry: RouteTelemetry | None = None

    @property
    def route_telemetry(self) -> RouteTelemetry | None:
        """Return the optional bounded telemetry recorder for this layer."""

        return self._route_telemetry

    def configure_anchored_top1(
        self,
        *,
        layer_index: int,
        router_feature_extractor: RouterFeatureExtractor,
        router_heads: AppendOnlyRouterHeads,
        experts: Mapping[str, Expert],
        capacity_policy: OverflowToAnchorCapacity | None = None,
        route_telemetry: RouteTelemetry | None = None,
    ) -> None:
        """Attach one deterministic anchored top-1 inference runtime.

        Args:
            layer_index:
                Zero-based decoder layer represented by this wrapper.
            router_feature_extractor:
                Frozen transformation from pre-MLP states to router features.
            router_heads:
                Affine head bank containing exactly one head at this layer for
                each attached expert.
            experts:
                Mapping from stable expert ID to independent expert module.
                Execution order follows router append order, not mapping order.
            capacity_policy:
                Optional BAL-001 policy. Defaults to overflow-to-anchor.
            route_telemetry:
                Optional bounded recorder whose layer must match this wrapper.

        This operation is one-shot. It freezes the feature extractor and router
        heads but preserves each expert's existing gradient flags so the routed
        path remains usable by later forced or routed training work.

        Raises:
            RuntimeError: If a routed runtime was already configured.
            TypeError: If a component does not implement its declared interface.
            ValueError:
                If layer ownership, feature sizes, or expert identifiers differ.
        """

        if self.routing_mode != "forced_anchor":
            raise RuntimeError("RoutedMLP routing is already configured.")
        self._replace_anchored_top1(
            layer_index=layer_index,
            router_feature_extractor=router_feature_extractor,
            router_heads=router_heads,
            experts=experts,
            capacity_policy=capacity_policy,
            route_telemetry=route_telemetry,
        )

    def configure_forced_expert(
        self,
        *,
        layer_index: int,
        expert_id: str,
        expert: Expert,
    ) -> None:
        """Attach one expert as the unconditional training path.

        Forced-path mode implements TRN-001: every token at this converted
        layer executes the supplied expert and no router or selection policy is
        evaluated. The method is one-shot from exact ``forced_anchor`` mode so
        training setup cannot silently replace an inference registry.

        Args:
            layer_index: Zero-based decoder layer represented by this wrapper.
            expert_id: Stable package expert identifier shared across layers.
            expert: Independently owned trainable expert transformation.

        Raises:
            RuntimeError: If the wrapper is not in ``forced_anchor`` mode.
            TypeError: If ``expert`` does not implement :class:`Expert`.
            ValueError: If layer or expert ownership metadata is invalid.
        """

        if self.routing_mode != "forced_anchor":
            raise RuntimeError(
                "Forced-expert training requires forced_anchor mode."
            )
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        if (
            not isinstance(expert_id, str)
            or not expert_id
            or expert_id != expert_id.strip()
        ):
            raise ValueError(
                "expert_id must be a non-empty string without surrounding "
                "whitespace."
            )
        if not isinstance(expert, Expert):
            raise TypeError("expert must implement the Expert interface.")

        self.layer_index = layer_index
        self.expert_ids = (expert_id,)
        self.router_feature_extractor = None
        self.router_heads = None
        self.experts = nn.ModuleList([expert])
        self._dispatcher = None
        self._route_telemetry = None
        self.routing_mode = "forced_expert"

    def replace_anchored_top1(
        self,
        *,
        layer_index: int,
        router_feature_extractor: RouterFeatureExtractor,
        router_heads: AppendOnlyRouterHeads,
        experts: Mapping[str, Expert],
        capacity_policy: OverflowToAnchorCapacity | None = None,
        route_telemetry: RouteTelemetry | None = None,
    ) -> None:
        """Atomically replace this layer's explicit top-1 runtime.

        This method supports explicit registry synchronization after package
        installation, unloading, or reinsertion. All candidate components are
        validated and ordered before any existing runtime reference changes.
        The anchor module and its parameter values are never replaced.

        Args:
            layer_index: Zero-based decoder layer represented by this wrapper.
            router_feature_extractor: Frozen pre-MLP feature transformation.
            router_heads: Complete affine-head bank for the new active registry.
            experts: Complete expert mapping for the new active registry.
            capacity_policy: Optional overflow-to-anchor capacity policy.
            route_telemetry: Optional recorder owned by this layer.
        """

        self._replace_anchored_top1(
            layer_index=layer_index,
            router_feature_extractor=router_feature_extractor,
            router_heads=router_heads,
            experts=experts,
            capacity_policy=capacity_policy,
            route_telemetry=route_telemetry,
        )

    def reset_to_forced_anchor(self) -> None:
        """Remove the routed runtime and restore the exact anchor-only path."""

        self.layer_index = None
        self.expert_ids = ()
        self.router_feature_extractor = None
        self.router_heads = None
        self.experts = nn.ModuleList()
        self._dispatcher = None
        self._route_telemetry = None
        self.routing_mode = "forced_anchor"

    def _replace_anchored_top1(
        self,
        *,
        layer_index: int,
        router_feature_extractor: RouterFeatureExtractor,
        router_heads: AppendOnlyRouterHeads,
        experts: Mapping[str, Expert],
        capacity_policy: OverflowToAnchorCapacity | None,
        route_telemetry: RouteTelemetry | None,
    ) -> None:
        """Validate a complete candidate runtime before swapping references."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        if not isinstance(router_feature_extractor, RouterFeatureExtractor):
            raise TypeError(
                "router_feature_extractor must be a RouterFeatureExtractor."
            )
        if not isinstance(router_heads, AppendOnlyRouterHeads):
            raise TypeError("router_heads must be AppendOnlyRouterHeads.")
        if router_feature_extractor.output_size != router_heads.feature_size:
            raise ValueError(
                "Router feature output size must match affine heads: "
                f"{router_feature_extractor.output_size} != "
                f"{router_heads.feature_size}."
            )
        if not isinstance(experts, Mapping):
            raise TypeError("experts must be a mapping from expert ID to Expert.")
        if not all(
            isinstance(expert_id, str) and isinstance(expert, Expert)
            for expert_id, expert in experts.items()
        ):
            raise TypeError("experts must map string IDs to Expert modules.")

        expert_ids = router_heads.expert_ids_for_layer(layer_index)
        if set(experts) != set(expert_ids) or len(experts) != len(expert_ids):
            raise ValueError(
                "Attached expert IDs must exactly match router heads at layer "
                f"{layer_index}: {tuple(experts)} != {expert_ids}."
            )
        if route_telemetry is not None:
            if not isinstance(route_telemetry, RouteTelemetry):
                raise TypeError("route_telemetry must be RouteTelemetry or None.")
            if route_telemetry.layer_index != layer_index:
                raise ValueError(
                    "route_telemetry layer must match the routed MLP layer."
                )

        dispatcher = AnchoredTop1Dispatcher(capacity_policy=capacity_policy)
        ordered_experts = [experts[expert_id] for expert_id in expert_ids]
        # Shared feature and score spaces are immutable during inference.
        # Expert trainability remains independently controlled by EXP-001.
        router_feature_extractor.requires_grad_(False)
        router_heads.requires_grad_(False)

        self.layer_index = layer_index
        self.expert_ids = expert_ids
        self.router_feature_extractor = router_feature_extractor
        self.router_heads = router_heads
        self.experts = nn.ModuleList(ordered_experts)
        self._dispatcher = dispatcher
        self._route_telemetry = route_telemetry
        self.routing_mode = "anchored_top1"

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute forced anchor or the explicitly configured top-1 runtime."""

        if self.routing_mode == "forced_anchor":
            return self.anchor_mlp(hidden_states, *args, **kwargs)
        if self.routing_mode == "forced_expert":
            if len(self.experts) != 1:
                raise RuntimeError("RoutedMLP forced-expert runtime is incomplete.")
            return self.experts[0](hidden_states, *args, **kwargs)

        if (
            self.layer_index is None
            or self.router_feature_extractor is None
            or self.router_heads is None
            or self._dispatcher is None
        ):
            raise RuntimeError("RoutedMLP top-1 runtime is incomplete.")

        expert_availability = kwargs.pop("expert_availability", None)
        features = self.router_feature_extractor(hidden_states)
        if self.expert_ids:
            expert_scores = self.router_heads(
                features,
                layer_index=self.layer_index,
            )
        else:
            expert_scores = hidden_states.new_empty(
                hidden_states.shape[:-1] + (0,)
            )
        result = self._dispatcher.dispatch(
            hidden_states,
            expert_scores=expert_scores,
            expert_ids=self.expert_ids,
            anchor_mlp=self.anchor_mlp,
            experts=tuple(self.experts),
            expert_availability=expert_availability,
            module_args=args,
            module_kwargs=kwargs,
        )
        if self._route_telemetry is not None:
            self._route_telemetry.record(
                result.selected_routes,
                result.capacity,
            )
        return result.output

    def extra_repr(self) -> str:
        """Describe active routing mode and stable layer ownership."""

        return (
            f"routing_mode={self.routing_mode!r}, layer_index={self.layer_index!r}, "
            f"expert_count={len(self.expert_ids)}"
        )
