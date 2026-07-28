"""Model-family-neutral routed components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn
from transformers import (
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
)

from anchored_moe.backends.base import ConversionReport, ModelBackend
from anchored_moe.configuration import AnchoredMoEConfig
from anchored_moe.experts.base import Expert
from anchored_moe.inference.dispatch import AnchoredTop1Dispatcher
from anchored_moe.inference.telemetry import RouteTelemetry
from anchored_moe.policies.capacity import OverflowToAnchorCapacity
from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.registry.registry import (
    ExpertRegistry,
    RegistryChange,
    RegistryUpgradeChange,
)
from anchored_moe.routers.base import RouterFeatureExtractor, RouterScoreBank
from anchored_moe.utils.hashing import sha256_json

if TYPE_CHECKING:
    from anchored_moe.inference.composition import RegistryCompositionReport


class RoutedMLP(nn.Module):
    """Wrap an immutable dense MLP with explicit opt-in routing.

    The wrapper holds the original MLP object rather than copying it. Its
    foundation behavior is therefore exactly one call to the same anchor
    module; it performs no arithmetic, routing, or tensor reshaping around the
    call. :meth:`configure_anchored_top1` can attach independently constructed
    router features, an ordered router score bank, and experts for one routed
    layer. Until that explicit call, forced-anchor behavior remains exact.

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
        self.router_heads: RouterScoreBank | None = None
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
        router_heads: RouterScoreBank,
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
                Score bank containing exactly one ordered score column at this
                layer for each attached expert.
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
        router_heads: RouterScoreBank,
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
            router_heads: Complete ordered score bank for the active registry.
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
        router_heads: RouterScoreBank,
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
        if not isinstance(router_heads, RouterScoreBank):
            raise TypeError("router_heads must implement RouterScoreBank.")
        if router_feature_extractor.output_size != router_heads.feature_size:
            raise ValueError(
                "Router feature output size must match the score bank: "
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
                "Attached expert IDs must exactly match router scores at layer "
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


class AnchoredModelError(ValueError):
    """Raised when an immutable anchor cannot form a valid anchored model."""


class AnchoredMoEForCausalLM(PreTrainedModel, GenerationMixin):
    """Hugging Face causal-LM facade for one immutable routed anchor.

    CORE-005 deliberately composes existing framework responsibilities instead
    of duplicating model-family logic. The wrapped ``anchor_model`` retains its
    native forward and generation implementations; a configured
    :class:`~anchored_moe.backends.base.ModelBackend` only replaces selected
    MLP objects with :class:`RoutedMLP` wrappers. Package installation,
    unloading, upgrades, and rollbacks mutate the in-memory registry but never
    the model. Call
    :meth:`synchronize_registry` explicitly to materialize the active anchored
    top-1 runtime.

    Args:
        config:
            Resolved anchored configuration. Its backend and converted layers
            determine conversion, while the wrapped anchor retains its native
            Hugging Face configuration.
        anchor_model:
            Supported pretrained causal LM to freeze and convert in place.
        backend:
            Optional model-family adapter. When omitted, ``config.backend`` is
            resolved through the built-in backend registry.
        registry:
            Optional existing in-memory expert registry. It is never
            synchronized implicitly.

    The class is infrastructure rather than a paper-derived algorithm. It does
    not implement CORE-006: ``save_pretrained`` is rejected until an explicit
    format can preserve independent packages without silently bundling or
    duplicating the anchor.
    """

    config_class = AnchoredMoEConfig
    base_model_prefix = "anchor_model"
    main_input_name = "input_ids"

    def __init__(
        self,
        config: AnchoredMoEConfig,
        anchor_model: PreTrainedModel,
        *,
        backend: ModelBackend | None = None,
        registry: ExpertRegistry | None = None,
    ) -> None:
        if not isinstance(config, AnchoredMoEConfig):
            raise TypeError("config must be an AnchoredMoEConfig.")
        if not isinstance(anchor_model, PreTrainedModel):
            raise TypeError("anchor_model must be a transformers.PreTrainedModel.")
        if registry is not None and not isinstance(registry, ExpertRegistry):
            raise TypeError("registry must be an ExpertRegistry or None.")

        resolved_backend = backend or _backend_from_name(config.backend)
        if not isinstance(resolved_backend, ModelBackend):
            raise TypeError("backend must implement ModelBackend.")
        if resolved_backend.model_type != config.backend:
            raise AnchoredModelError(
                "Backend identity does not match AnchoredMoEConfig: "
                f"{resolved_backend.model_type!r} != {config.backend!r}."
            )
        if not resolved_backend.supports_model(anchor_model):
            raise AnchoredModelError(
                f"Backend {config.backend!r} does not support "
                f"{type(anchor_model).__name__}."
            )

        live_config_sha256 = sha256_json(anchor_model.config.to_dict())
        if (
            config.base_config_sha256 is not None
            and config.base_config_sha256 != live_config_sha256
        ):
            raise AnchoredModelError(
                "Configured base_config_sha256 does not match the live anchor."
            )

        super().__init__(config)
        self.anchor_model = anchor_model
        self._backend = resolved_backend
        self._registry = registry if registry is not None else ExpertRegistry()

        # The backend performs the only structural mutation: it freezes the
        # complete incumbent model and moves selected MLPs under exact-fallback
        # wrappers. No tensor value is changed by construction.
        self._conversion_report = self._backend.convert(
            self.anchor_model,
            self.config,
        )

        # Generation is delegated to the native anchor implementation. Sharing
        # its GenerationConfig keeps ordinary user overrides visible on both
        # facades without rebuilding model-family generation behavior.
        self.generation_config = self.anchor_model.generation_config
        self.main_input_name = self.anchor_model.main_input_name

    @classmethod
    def from_anchor_model(
        cls,
        anchor_model: PreTrainedModel,
        *,
        config: AnchoredMoEConfig | None = None,
        backend: ModelBackend | None = None,
        registry: ExpertRegistry | None = None,
        converted_layers: str | tuple[int, ...] | list[int] = "upper_half",
    ) -> AnchoredMoEForCausalLM:
        """Freeze and convert an already loaded Hugging Face causal LM.

        Args:
            anchor_model: Live supported causal LM. Conversion occurs in place.
            config:
                Optional complete anchored configuration. When omitted, a local
                configuration is derived from the anchor's model type and
                configuration digest.
            backend: Optional explicit backend adapter.
            registry: Optional unsynchronized in-memory package registry.
            converted_layers:
                Layer selector used only when ``config`` is omitted.

        Returns:
            A facade whose forward and generation calls execute the converted
            anchor model.
        """

        if not isinstance(anchor_model, PreTrainedModel):
            raise TypeError("anchor_model must be a transformers.PreTrainedModel.")
        if config is None:
            model_type = getattr(anchor_model.config, "model_type", None)
            if not isinstance(model_type, str) or not model_type:
                raise AnchoredModelError(
                    "The anchor configuration must declare a model_type."
                )
            config = AnchoredMoEConfig(
                backend=model_type,
                converted_layers=converted_layers,
                base_config_sha256=sha256_json(anchor_model.config.to_dict()),
            )
        return cls(
            config,
            anchor_model,
            backend=backend,
            registry=registry,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        anchored_moe_config: AnchoredMoEConfig | None = None,
        backend: ModelBackend | None = None,
        registry: ExpertRegistry | None = None,
        converted_layers: str | tuple[int, ...] | list[int] = "upper_half",
        revision: str | None = None,
        **kwargs: Any,
    ) -> AnchoredMoEForCausalLM:
        """Load a base checkpoint through ``AutoModelForCausalLM`` and convert it.

        ``pretrained_model_name_or_path``, ``revision``, ``model_args``, and
        remaining keyword arguments are forwarded to the Hugging Face anchor
        loader. This method loads a base checkpoint, not a serialized anchored
        composition; that round trip remains CORE-006.

        Args:
            pretrained_model_name_or_path: Hub identifier or local base model.
            anchored_moe_config:
                Optional resolved anchored configuration. If omitted, the
                backend is derived from the loaded base model and the supplied
                layer selector.
            backend: Optional explicit backend implementation.
            registry: Optional unsynchronized in-memory registry.
            converted_layers:
                Layer selector used only when no anchored configuration is
                supplied.
            revision: Exact base checkpoint revision where applicable.

        Returns:
            Loaded, frozen, and converted anchored causal LM.
        """

        if anchored_moe_config is not None and not isinstance(
            anchored_moe_config,
            AnchoredMoEConfig,
        ):
            raise TypeError(
                "anchored_moe_config must be an AnchoredMoEConfig or None."
            )
        anchor_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            revision=revision,
            **kwargs,
        )
        if anchored_moe_config is None:
            model_type = getattr(anchor_model.config, "model_type", None)
            if not isinstance(model_type, str) or not model_type:
                raise AnchoredModelError(
                    "The loaded anchor configuration must declare a model_type."
                )
            anchored_moe_config = AnchoredMoEConfig(
                base_model_name_or_path=str(pretrained_model_name_or_path),
                base_model_revision=revision,
                base_config_sha256=sha256_json(anchor_model.config.to_dict()),
                backend=model_type,
                converted_layers=converted_layers,
            )
        elif (
            anchored_moe_config.base_model_name_or_path is not None
            and anchored_moe_config.base_model_name_or_path
            != str(pretrained_model_name_or_path)
        ):
            raise AnchoredModelError(
                "anchored_moe_config.base_model_name_or_path does not match "
                "the checkpoint being loaded."
            )

        return cls(
            anchored_moe_config,
            anchor_model,
            backend=backend,
            registry=registry,
        )

    @property
    def backend(self) -> ModelBackend:
        """Return the model-family adapter used for conversion and composition."""

        return self._backend

    @property
    def registry(self) -> ExpertRegistry:
        """Return the in-memory registry without synchronizing model routes."""

        return self._registry

    @property
    def conversion_report(self) -> ConversionReport:
        """Return the immutable backend conversion summary."""

        return self._conversion_report

    def install(self, package: ExpertPackage) -> RegistryChange:
        """Install a package descriptor without changing the model runtime."""

        return self._registry.install(package)

    def unload(self, package: PackageIdentity) -> RegistryChange:
        """Unload a package descriptor without changing the model runtime."""

        return self._registry.unload(package)

    def upgrade(
        self,
        current: PackageIdentity,
        replacement: ExpertPackage,
    ) -> RegistryUpgradeChange:
        """Atomically upgrade registry state without changing model routes."""

        return self._registry.upgrade(current, replacement)

    def rollback(
        self,
        upgrade: RegistryUpgradeChange,
        previous_package: ExpertPackage,
    ) -> RegistryUpgradeChange:
        """Rollback one exact registry upgrade without changing model routes."""

        return self._registry.rollback(upgrade, previous_package)

    def synchronize_registry(self) -> RegistryCompositionReport:
        """Explicitly apply the active anchored top-1 registry to the model.

        Returns:
            A
            :class:`~anchored_moe.inference.composition.RegistryCompositionReport`
            recording the applied registry hash, layers, and expert order.

        Package tensors are revalidated and every candidate runtime is built
        before routed wrappers change. An empty registry restores exact
        forced-anchor execution.
        """

        # Local import avoids a module cycle: package composition imports
        # RoutedMLP from this module.
        from anchored_moe.inference.composition import (
            synchronize_anchored_top1_registry,
        )

        return synchronize_anchored_top1_registry(
            self.anchor_model,
            backend=self._backend,
            registry=self._registry,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate causal-LM forward execution to the native anchor model."""

        return self.anchor_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to the native model-family implementation."""

        return self.anchor_model.generate(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        """Return the immutable anchor's input embedding module."""

        return self.anchor_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module | None:
        """Return the immutable anchor's output embedding module, if present."""

        return self.anchor_model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Reject replacement of immutable anchor input embeddings."""

        raise AnchoredModelError(
            "Anchor input embeddings are immutable after composition."
        )

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Reject replacement of immutable anchor output embeddings."""

        raise AnchoredModelError(
            "Anchor output embeddings are immutable after composition."
        )

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        """Reject unsafe full-model persistence until CORE-006 is implemented."""

        raise NotImplementedError(
            "AnchoredMoEForCausalLM serialization is not implemented. "
            "Persist expert packages independently through their safe "
            "safetensors APIs until CORE-006 defines the composition format."
        )


def _backend_from_name(name: str) -> ModelBackend:
    """Resolve a built-in backend without coupling RoutedMLP to model families."""

    if name == "qwen3":
        # Imported lazily because the Qwen3 backend imports RoutedMLP.
        from anchored_moe.backends.qwen3 import Qwen3Backend

        return Qwen3Backend()
    raise AnchoredModelError(f"Unsupported model backend: {name!r}.")
