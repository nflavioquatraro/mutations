"""Joint two-expert local mini-MoE training with an immutable anchor.

TRN-003 trains one package-local pair of latent experts and their affine
router heads together. Every specialization token executes both experts at
every converted layer, and COM-001 score-softmax combines their activations.
The anchor, shared backbone, router feature transformation, incumbent experts,
and registry state remain frozen.

The first implementation deliberately fixes the local set size to two. This
keeps the selected expert set constant, so gradients reach both router scores
without a discrete top-k boundary or a load-balancing objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SaveStrategy, TrainOutput

from anchored_moe.backends.base import ModelBackend
from anchored_moe.experts.base import Expert
from anchored_moe.modeling import RoutedMLP
from anchored_moe.policies.combination import ScoreSoftmaxCombination
from anchored_moe.routers.base import RouterFeatureExtractor
from anchored_moe.routers.heads import AppendOnlyRouterHeads
from anchored_moe.training.expert_trainer import ForcedPathTrainingError


@dataclass(frozen=True)
class LocalMiniMoETrainingConfig:
    """Configuration for deterministic two-expert local mixture training.

    Args:
        mix_temperature:
            Positive COM-001 temperature applied to the two local scores.
        router_bias_epsilon:
            Positive deterministic symmetry-breaking magnitude. At every
            layer, the first router head starts at ``+epsilon`` and the second
            at ``-epsilon`` while all affine weights remain zero.
    """

    strategy: ClassVar[str] = "local_mini_moe_top2"
    expert_count: ClassVar[int] = 2
    combination: ClassVar[str] = "score_softmax"

    mix_temperature: float = 1.0
    router_bias_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        """Normalize finite positive hyperparameters."""

        for field_name in ("mix_temperature", "router_bias_epsilon"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or float(value) <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive.")
            object.__setattr__(self, field_name, float(value))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible public configuration."""

        return {
            "combination": self.combination,
            "expert_count": self.expert_count,
            "mix_temperature": self.mix_temperature,
            "router_bias_epsilon": self.router_bias_epsilon,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LocalMiniMoETrainingConfig:
        """Strictly reconstruct a serialized TRN-003 configuration."""

        if not isinstance(value, dict):
            raise TypeError("Local mini-MoE config must be a dictionary.")
        expected = {
            "combination",
            "expert_count",
            "mix_temperature",
            "router_bias_epsilon",
            "strategy",
        }
        if set(value) != expected:
            raise ValueError("Local mini-MoE config fields are incompatible.")
        if value["strategy"] != cls.strategy:
            raise ValueError("Unsupported local mini-MoE strategy.")
        if value["expert_count"] != cls.expert_count:
            raise ValueError("TRN-003 requires exactly two local experts.")
        if value["combination"] != cls.combination:
            raise ValueError("Unsupported local mini-MoE combination policy.")
        return cls(
            mix_temperature=value["mix_temperature"],
            router_bias_epsilon=value["router_bias_epsilon"],
        )


@dataclass(frozen=True)
class LocalMiniMoEExpertBinding:
    """Bind one package-local expert to one converted decoder layer."""

    layer_index: int
    expert_id: str
    expert: Expert

    def __post_init__(self) -> None:
        """Validate stable layer and expert ownership metadata."""

        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        if (
            not isinstance(self.expert_id, str)
            or not self.expert_id
            or self.expert_id != self.expert_id.strip()
        ):
            raise ValueError(
                "expert_id must be non-empty without surrounding whitespace."
            )
        if not isinstance(self.expert, Expert):
            raise TypeError("expert must implement the Expert interface.")

    @property
    def key(self) -> tuple[int, str]:
        """Return the stable layer/expert ownership key."""

        return self.layer_index, self.expert_id


@dataclass(frozen=True)
class LocalMiniMoETrainingTopology:
    """Immutable prepared topology and exact optimizer ownership report."""

    layer_indices: tuple[int, ...]
    expert_ids: tuple[str, str]
    router_head_keys: tuple[tuple[int, str], ...]
    trainable_parameter_names: tuple[str, ...]
    total_parameter_count: int
    trainable_parameter_count: int
    config: LocalMiniMoETrainingConfig

    def __post_init__(self) -> None:
        """Validate deterministic ordering and non-empty optimizer state."""

        if not self.layer_indices or tuple(sorted(self.layer_indices)) != (
            self.layer_indices
        ):
            raise ValueError("layer_indices must be non-empty and sorted.")
        if (
            not isinstance(self.expert_ids, tuple)
            or len(self.expert_ids) != 2
            or len(set(self.expert_ids)) != 2
        ):
            raise ValueError("expert_ids must contain two unique identifiers.")
        if not self.router_head_keys:
            raise ValueError("router_head_keys must not be empty.")
        if not self.trainable_parameter_names:
            raise ValueError("trainable_parameter_names must not be empty.")
        if self.total_parameter_count <= 0:
            raise ValueError("total_parameter_count must be positive.")
        if not 0 < self.trainable_parameter_count <= self.total_parameter_count:
            raise ValueError("trainable_parameter_count is invalid.")
        if not isinstance(self.config, LocalMiniMoETrainingConfig):
            raise TypeError("config must be LocalMiniMoETrainingConfig.")


@dataclass(frozen=True)
class LocalMiniMoETrainingReport:
    """Immutable outcome of one completed local mini-MoE Trainer run."""

    topology: LocalMiniMoETrainingTopology
    global_step: int
    training_loss: float

    def __post_init__(self) -> None:
        """Validate positive progress and finite aggregate loss."""

        if not isinstance(self.topology, LocalMiniMoETrainingTopology):
            raise TypeError("topology must be LocalMiniMoETrainingTopology.")
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step <= 0
        ):
            raise ValueError("global_step must be a positive integer.")
        if (
            isinstance(self.training_loss, bool)
            or not isinstance(self.training_loss, (int, float))
            or not math.isfinite(self.training_loss)
        ):
            raise ValueError("training_loss must be finite.")
        object.__setattr__(self, "training_loss", float(self.training_loss))


def prepare_local_mini_moe_training(
    model: nn.Module,
    *,
    backend: ModelBackend,
    expert_bindings: Sequence[LocalMiniMoEExpertBinding],
    router_feature_extractor: RouterFeatureExtractor,
    router_heads: AppendOnlyRouterHeads,
    config: LocalMiniMoETrainingConfig | None = None,
) -> LocalMiniMoETrainingTopology:
    """Freeze an anchor and attach a jointly trainable two-expert local MoE.

    The complete layer-by-expert rectangle, zero-initialized affine heads,
    parameter identity, and storage independence are validated before any
    route is attached. The router then receives deterministic opposite biases
    to break the symmetry of anchor-copied experts.

    This preparation function leaves the local training route attached. The
    :class:`LocalMiniMoETrainer` provides the failure-safe one-run lifecycle.
    """

    resolved_config = config or LocalMiniMoETrainingConfig()
    if not isinstance(resolved_config, LocalMiniMoETrainingConfig):
        raise TypeError("config must be LocalMiniMoETrainingConfig or None.")
    bindings, wrappers, expert_ids = _validate_local_topology(
        model,
        backend=backend,
        expert_bindings=expert_bindings,
        router_feature_extractor=router_feature_extractor,
        router_heads=router_heads,
    )
    original_router_state = {
        name: tensor.detach().clone()
        for name, tensor in router_heads.state_dict().items()
    }
    try:
        model.requires_grad_(False)
        router_feature_extractor.requires_grad_(False)
        router_heads.requires_grad_(True)
        for binding in bindings:
            binding.expert.set_trainable(True)

        with torch.no_grad():
            for layer_index in tuple(sorted(wrappers)):
                layer_heads = [
                    head
                    for head in router_heads.heads
                    if head.layer_index == layer_index
                ]
                layer_heads[0].linear.bias.fill_(
                    resolved_config.router_bias_epsilon
                )
                layer_heads[1].linear.bias.fill_(
                    -resolved_config.router_bias_epsilon
                )

        combination = ScoreSoftmaxCombination(
            mix_temperature=resolved_config.mix_temperature
        )
        by_key = {binding.key: binding.expert for binding in bindings}
        for layer_index, wrapper in wrappers.items():
            wrapper.configure_local_mini_moe(
                layer_index=layer_index,
                router_feature_extractor=router_feature_extractor,
                router_heads=router_heads,
                experts={
                    expert_id: by_key[(layer_index, expert_id)]
                    for expert_id in expert_ids
                },
                combination_policy=combination,
            )

        expected_ids = _expected_trainable_parameter_ids(
            bindings,
            router_heads=router_heads,
        )
        trainable = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if {id(parameter) for _, parameter in trainable} != expected_ids:
            raise ForcedPathTrainingError(
                "Prepared local mini-MoE trainable parameters do not exactly "
                "match package experts and affine router heads."
            )
        return LocalMiniMoETrainingTopology(
            layer_indices=tuple(sorted(wrappers)),
            expert_ids=expert_ids,
            router_head_keys=router_heads.head_keys,
            trainable_parameter_names=tuple(name for name, _ in trainable),
            total_parameter_count=sum(
                parameter.numel() for parameter in model.parameters()
            ),
            trainable_parameter_count=sum(
                parameter.numel() for _, parameter in trainable
            ),
            config=resolved_config,
        )
    except Exception:
        _restore_anchor_only_state(
            model,
            wrappers=tuple(wrappers.values()),
            bindings=bindings,
            router_feature_extractor=router_feature_extractor,
            router_heads=router_heads,
        )
        router_heads.load_state_dict(original_router_state, strict=True)
        raise


class LocalMiniMoETrainer(Trainer):
    """One-run Hugging Face Trainer for a two-expert package-local mixture.

    Construction attaches the validated TRN-003 topology before
    Trainer/Accelerate device preparation. :meth:`train` reasserts route and
    optimizer ownership, executes the ordinary causal-language-model loss,
    then always restores exact forced-anchor routing and freezes the trained
    experts and router heads. Partial optimizer updates are not rolled back
    after a training failure.

    Full-model saving, Hub push, model reinitialization, and checkpoint resume
    are rejected. Trained experts and affine heads remain caller-owned and can
    be exported through their existing safetensors package APIs.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        args: TrainingArguments,
        backend: ModelBackend,
        expert_bindings: Sequence[LocalMiniMoEExpertBinding],
        router_feature_extractor: RouterFeatureExtractor,
        router_heads: AppendOnlyRouterHeads,
        config: LocalMiniMoETrainingConfig | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if not isinstance(args, TrainingArguments):
            raise TypeError("args must be transformers.TrainingArguments.")
        if args.save_strategy != SaveStrategy.NO:
            raise ForcedPathTrainingError(
                "TRN-003 requires save_strategy='no'; export the package "
                "experts and router heads through safe APIs."
            )
        if args.push_to_hub:
            raise ForcedPathTrainingError(
                "TRN-003 does not publish the immutable anchor model."
            )
        if trainer_kwargs.get("model_init") is not None:
            raise ForcedPathTrainingError(
                "model_init is incompatible with explicit local experts."
            )

        bindings = _normalize_bindings(expert_bindings)
        topology = prepare_local_mini_moe_training(
            model,
            backend=backend,
            expert_bindings=bindings,
            router_feature_extractor=router_feature_extractor,
            router_heads=router_heads,
            config=config,
        )
        wrappers = _routed_wrappers(model, backend=backend)
        self.backend = backend
        self.expert_bindings = bindings
        self.router_feature_extractor = router_feature_extractor
        self.router_heads = router_heads
        self.topology = topology
        self.training_report: LocalMiniMoETrainingReport | None = None
        self._wrappers = wrappers
        self._training_consumed = False
        self._expected_trainable_parameter_ids = frozenset(
            _expected_trainable_parameter_ids(
                bindings,
                router_heads=router_heads,
            )
        )
        try:
            super().__init__(model=model, args=args, **trainer_kwargs)
            self.assert_local_mini_moe_invariants()
        except Exception:
            self._cleanup()
            raise

    def assert_local_mini_moe_invariants(self) -> None:
        """Assert the exact local route and optimizer parameter identity."""

        by_key = {binding.key: binding.expert for binding in self.expert_bindings}
        invalid_layers: list[int] = []
        for layer_index, wrapper in self._wrappers.items():
            expected_experts = tuple(
                by_key[(layer_index, expert_id)]
                for expert_id in self.topology.expert_ids
            )
            if (
                wrapper.routing_mode != "local_mini_moe"
                or wrapper.expert_ids != self.topology.expert_ids
                or wrapper.router_feature_extractor
                is not self.router_feature_extractor
                or wrapper.router_heads is not self.router_heads
                or tuple(wrapper.experts) != expected_experts
            ):
                invalid_layers.append(layer_index)

        actual_ids = frozenset(
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if (
            invalid_layers
            or actual_ids != self._expected_trainable_parameter_ids
        ):
            raise ForcedPathTrainingError(
                "Local mini-MoE invariants changed before optimization: "
                f"invalid_layers={invalid_layers}, "
                f"expected_trainable={len(self._expected_trainable_parameter_ids)}, "
                f"actual_trainable={len(actual_ids)}."
            )

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Any = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Run one joint optimization and restore exact anchor routing."""

        if self._training_consumed:
            raise ForcedPathTrainingError(
                "LocalMiniMoETrainer supports exactly one training run."
            )
        if resume_from_checkpoint not in (None, False):
            raise ForcedPathTrainingError(
                "Checkpoint resume is not supported by TRN-003."
            )
        self._training_consumed = True
        try:
            self.assert_local_mini_moe_invariants()
            output = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
            self.training_report = LocalMiniMoETrainingReport(
                topology=self.topology,
                global_step=output.global_step,
                training_loss=output.training_loss,
            )
            return output
        finally:
            self._cleanup()

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        """Reject full-model saving across the immutable-anchor boundary."""

        raise ForcedPathTrainingError(
            "LocalMiniMoETrainer does not save the immutable anchor; export "
            "the trained experts and router heads explicitly."
        )

    def _cleanup(self) -> None:
        """Detach local routes, clear gradients, and freeze owned artifacts."""

        _restore_anchor_only_state(
            self.model,
            wrappers=tuple(self._wrappers.values()),
            bindings=self.expert_bindings,
            router_feature_extractor=self.router_feature_extractor,
            router_heads=self.router_heads,
        )


def _validate_local_topology(
    model: nn.Module,
    *,
    backend: ModelBackend,
    expert_bindings: Sequence[LocalMiniMoEExpertBinding],
    router_feature_extractor: RouterFeatureExtractor,
    router_heads: AppendOnlyRouterHeads,
) -> tuple[
    tuple[LocalMiniMoEExpertBinding, ...],
    dict[int, RoutedMLP],
    tuple[str, str],
]:
    """Validate rectangular ownership and zero router initialization."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must be a ModelBackend.")
    if not isinstance(router_feature_extractor, RouterFeatureExtractor):
        raise TypeError(
            "router_feature_extractor must be a RouterFeatureExtractor."
        )
    if not isinstance(router_heads, AppendOnlyRouterHeads):
        raise TypeError("router_heads must be AppendOnlyRouterHeads.")
    if router_feature_extractor.output_size != router_heads.feature_size:
        raise ForcedPathTrainingError(
            "Router feature and affine-head dimensions must match."
        )

    bindings = _normalize_bindings(expert_bindings)
    wrappers = _routed_wrappers(model, backend=backend)
    layer_indices = tuple(wrappers)
    binding_layers = tuple(sorted({binding.layer_index for binding in bindings}))
    if binding_layers != layer_indices:
        raise ForcedPathTrainingError(
            "Local experts must exactly cover converted RoutedMLP layers: "
            f"model={layer_indices}, bindings={binding_layers}."
        )
    invalid_modes = tuple(
        layer_index
        for layer_index, wrapper in wrappers.items()
        if wrapper.routing_mode != "forced_anchor"
    )
    if invalid_modes:
        raise ForcedPathTrainingError(
            "Local mini-MoE training requires forced_anchor mode; "
            f"invalid_layers={invalid_modes}."
        )

    first_ids = router_heads.expert_ids_for_layer(layer_indices[0])
    if len(first_ids) != 2 or len(set(first_ids)) != 2:
        raise ForcedPathTrainingError(
            "TRN-003 requires exactly two unique router experts per layer."
        )
    expert_ids: tuple[str, str] = (first_ids[0], first_ids[1])
    expected_keys = tuple(
        (layer_index, expert_id)
        for layer_index in layer_indices
        for expert_id in expert_ids
    )
    if router_heads.head_keys != expected_keys:
        raise ForcedPathTrainingError(
            "Affine router heads must form the exact layer-major two-expert "
            f"rectangle: expected={expected_keys}, actual={router_heads.head_keys}."
        )
    if any(
        router_heads.expert_ids_for_layer(layer_index) != expert_ids
        for layer_index in layer_indices
    ):
        raise ForcedPathTrainingError(
            "Every local layer must use the same ordered expert IDs."
        )
    binding_keys = tuple(binding.key for binding in bindings)
    if set(binding_keys) != set(expected_keys) or len(binding_keys) != len(
        expected_keys
    ):
        raise ForcedPathTrainingError(
            "Expert bindings must form the exact layer-by-expert rectangle."
        )

    for head in router_heads.heads:
        if head.temperature != 1.0 or head.threshold != 0.0:
            raise ForcedPathTrainingError(
                "Local training heads must begin uncalibrated with temperature "
                "1 and threshold 0."
            )
        if any(
            torch.count_nonzero(parameter.detach()).item()
            for parameter in head.parameters()
        ):
            raise ForcedPathTrainingError(
                "Local training affine heads must begin at exact zero."
            )

    _validate_parameter_ownership(
        model,
        bindings=bindings,
        router_feature_extractor=router_feature_extractor,
        router_heads=router_heads,
    )
    return bindings, wrappers, expert_ids


def _validate_parameter_ownership(
    model: nn.Module,
    *,
    bindings: tuple[LocalMiniMoEExpertBinding, ...],
    router_feature_extractor: RouterFeatureExtractor,
    router_heads: AppendOnlyRouterHeads,
) -> None:
    """Reject identity or storage aliases across every ownership boundary."""

    model_parameters = tuple(model.parameters())
    model_ids = {id(parameter) for parameter in model_parameters}
    model_storage = {_storage_key(parameter) for parameter in model_parameters}
    owned_ids: set[int] = set()
    owned_storage: set[tuple[str, int | None, int]] = set()

    modules: list[tuple[str, nn.Module]] = [
        *[
            (
                f"expert {binding.expert_id!r} layer {binding.layer_index}",
                binding.expert,
            )
            for binding in bindings
        ],
        ("router heads", router_heads),
        ("router feature extractor", router_feature_extractor),
    ]
    for owner, module in modules:
        parameters = tuple(module.parameters())
        parameter_ids = {id(parameter) for parameter in parameters}
        storage_keys = {_storage_key(parameter) for parameter in parameters}
        if len(parameter_ids) != len(parameters):
            raise ForcedPathTrainingError(f"{owner} repeats parameters.")
        if len(storage_keys) != len(parameters):
            raise ForcedPathTrainingError(f"{owner} shares parameter storage.")
        if parameter_ids & model_ids or storage_keys & model_storage:
            raise ForcedPathTrainingError(f"{owner} shares model state.")
        if parameter_ids & owned_ids or storage_keys & owned_storage:
            raise ForcedPathTrainingError(
                "Local mini-MoE components must not share parameter storage."
            )
        owned_ids.update(parameter_ids)
        owned_storage.update(storage_keys)


def _normalize_bindings(
    value: Sequence[LocalMiniMoEExpertBinding],
) -> tuple[LocalMiniMoEExpertBinding, ...]:
    """Normalize bindings into stable layer/expert lexical order."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "expert_bindings must be LocalMiniMoEExpertBinding values."
        )
    if not value or not all(
        isinstance(binding, LocalMiniMoEExpertBinding) for binding in value
    ):
        raise TypeError(
            "expert_bindings must be a non-empty sequence of "
            "LocalMiniMoEExpertBinding values."
        )
    normalized = tuple(
        sorted(value, key=lambda binding: (binding.layer_index, binding.expert_id))
    )
    keys = tuple(binding.key for binding in normalized)
    if len(set(keys)) != len(keys):
        raise ForcedPathTrainingError(
            "expert_bindings must not repeat layer/expert ownership keys."
        )
    return normalized


def _routed_wrappers(
    model: nn.Module,
    *,
    backend: ModelBackend,
) -> dict[int, RoutedMLP]:
    """Discover converted routed layers in deterministic backend order."""

    try:
        references = tuple(backend.iter_mlp_references(model))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ForcedPathTrainingError(
            f"Cannot discover model MLPs for local mini-MoE training: {error}"
        ) from error
    wrappers = {
        reference.layer_index: reference.module
        for reference in references
        if isinstance(reference.module, RoutedMLP)
    }
    if not wrappers:
        raise ForcedPathTrainingError(
            "Local mini-MoE training requires converted RoutedMLP layers."
        )
    return dict(sorted(wrappers.items()))


def _expected_trainable_parameter_ids(
    bindings: tuple[LocalMiniMoEExpertBinding, ...],
    *,
    router_heads: AppendOnlyRouterHeads,
) -> set[int]:
    """Return the exact expert-adapter and affine-head optimizer identity set."""

    return {
        id(parameter)
        for module in [
            *(binding.expert for binding in bindings),
            router_heads,
        ]
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def _restore_anchor_only_state(
    model: nn.Module,
    *,
    wrappers: tuple[RoutedMLP, ...],
    bindings: tuple[LocalMiniMoEExpertBinding, ...],
    router_feature_extractor: RouterFeatureExtractor,
    router_heads: AppendOnlyRouterHeads,
) -> None:
    """Restore exact anchor execution and freeze all caller-owned artifacts."""

    model.zero_grad(set_to_none=True)
    for wrapper in wrappers:
        wrapper.reset_to_forced_anchor()
    model.requires_grad_(False)
    for binding in bindings:
        binding.expert.set_trainable(False)
        for parameter in binding.expert.parameters():
            parameter.grad = None
    router_heads.requires_grad_(False)
    router_feature_extractor.requires_grad_(False)
    for module in (router_heads, router_feature_extractor):
        for parameter in module.parameters():
            parameter.grad = None


def _storage_key(parameter: nn.Parameter) -> tuple[str, int | None, int]:
    """Identify underlying storage so views cannot cross ownership boundaries."""

    return (
        parameter.device.type,
        parameter.device.index,
        parameter.untyped_storage().data_ptr(),
    )
