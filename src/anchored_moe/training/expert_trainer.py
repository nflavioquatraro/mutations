"""Forced-path single-expert training with exact parameter isolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SaveStrategy, TrainOutput

from anchored_moe.backends.base import ModelBackend
from anchored_moe.experts.base import Expert
from anchored_moe.modeling import RoutedMLP
from anchored_moe.training.objectives import (
    ExpertObjectiveConfig,
    ExpertObjectiveReport,
    ExpertWeightDrift,
    ReplayKLPreservation,
)


class ForcedPathTrainingError(RuntimeError):
    """Raised when forced-path routing or trainable parameters are unsafe."""


@dataclass(frozen=True)
class ForcedExpertBinding:
    """Bind one trainable expert object to one converted model layer.

    Attributes:
        layer_index: Zero-based converted decoder layer.
        expert_id: Stable package expert identifier shared across layers.
        expert: Independently owned expert module trained at this layer.
    """

    layer_index: int
    expert_id: str
    expert: Expert

    def __post_init__(self) -> None:
        """Validate ownership metadata and the expert interface."""

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
                "expert_id must be a non-empty string without surrounding "
                "whitespace."
            )
        if not isinstance(self.expert, Expert):
            raise TypeError("expert must implement the Expert interface.")


@dataclass(frozen=True)
class ForcedPathTrainingReport:
    """Resolved forced-path training topology and parameter counts.

    Attributes:
        layer_indices: Converted layers trained in deterministic order.
        expert_id: One package expert identifier used at every layer.
        trainable_parameter_names: Exact model parameter names eligible for
            optimization after preparation.
        total_parameter_count: Complete model parameter count.
        trainable_parameter_count: Sum of expert parameters being optimized.
    """

    layer_indices: tuple[int, ...]
    expert_id: str
    trainable_parameter_names: tuple[str, ...]
    total_parameter_count: int
    trainable_parameter_count: int

    def __post_init__(self) -> None:
        """Validate ordering and non-empty training state."""

        if not self.layer_indices:
            raise ValueError("layer_indices must not be empty.")
        if tuple(sorted(self.layer_indices)) != self.layer_indices:
            raise ValueError("layer_indices must be sorted.")
        if not self.trainable_parameter_names:
            raise ValueError("trainable_parameter_names must not be empty.")
        if self.total_parameter_count <= 0:
            raise ValueError("total_parameter_count must be positive.")
        if (
            self.trainable_parameter_count <= 0
            or self.trainable_parameter_count > self.total_parameter_count
        ):
            raise ValueError("trainable_parameter_count is invalid.")


def prepare_forced_path_training(
    model: nn.Module,
    *,
    backend: ModelBackend,
    expert_bindings: Sequence[ForcedExpertBinding],
) -> ForcedPathTrainingReport:
    """Freeze a converted model and attach one forced expert at every layer.

    The operation implements the TRN-001 execution topology:

    ``h_(l,t) -> E_(l,k)(h_(l,t))``

    for every token ``t`` and converted layer ``l``. It validates the entire
    topology and parameter ownership before mutation, freezes the full model,
    enables gradients only on the supplied experts, and then attaches those
    experts without a router.

    Args:
        model: Converted model whose routed wrappers are in forced-anchor mode.
        backend: Model-family adapter used for stable layer discovery.
        expert_bindings: Exactly one independent expert for every converted
            routed layer. All bindings must use the same expert identifier.

    Returns:
        Immutable report containing topology and exact trainable parameters.

    Raises:
        TypeError: If model, backend, or bindings have invalid types.
        ForcedPathTrainingError:
            If layer coverage, routing state, expert ownership, or trainable
            parameters violate the isolated forced-path contract.

    This is framework infrastructure rather than a paper-derived algorithm.
    Replay preservation and weight-drift objectives are deliberately excluded
    and remain TRN-005.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must be a ModelBackend.")
    bindings = _normalize_bindings(expert_bindings)

    try:
        references = tuple(backend.iter_mlp_references(model))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ForcedPathTrainingError(
            f"Cannot discover model MLPs for forced-path training: {error}"
        ) from error
    wrappers = {
        reference.layer_index: reference.module
        for reference in references
        if isinstance(reference.module, RoutedMLP)
    }
    wrapper_layers = tuple(sorted(wrappers))
    binding_layers = tuple(binding.layer_index for binding in bindings)
    if wrapper_layers != binding_layers:
        raise ForcedPathTrainingError(
            "Expert bindings must exactly cover converted RoutedMLP layers: "
            f"model={wrapper_layers}, bindings={binding_layers}."
        )

    expert_ids = {binding.expert_id for binding in bindings}
    if len(expert_ids) != 1:
        raise ForcedPathTrainingError(
            "A single-expert package must use one expert_id at every layer."
        )
    for binding in bindings:
        wrapper = wrappers[binding.layer_index]
        if wrapper.routing_mode != "forced_anchor":
            raise ForcedPathTrainingError(
                "Forced-path preparation requires every routed layer to be in "
                f"forced_anchor mode; layer {binding.layer_index} is "
                f"{wrapper.routing_mode!r}."
            )

    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    model_storage_keys = {
        _parameter_storage_key(parameter) for parameter in model.parameters()
    }
    owned_parameter_ids: set[int] = set()
    owned_storage_keys: set[tuple[str, int | None, int]] = set()
    for binding in bindings:
        parameters = tuple(binding.expert.parameters())
        if not parameters:
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} has no parameters."
            )
        parameter_ids = {id(parameter) for parameter in parameters}
        if len(parameter_ids) != len(parameters):
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} repeats parameters."
            )
        storage_keys = {
            _parameter_storage_key(parameter) for parameter in parameters
        }
        if len(storage_keys) != len(parameters):
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} shares parameter storage."
            )
        if parameter_ids & model_parameter_ids:
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} shares model parameters."
            )
        if storage_keys & model_storage_keys:
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} shares model storage."
            )
        if parameter_ids & owned_parameter_ids:
            raise ForcedPathTrainingError(
                "Experts at different layers must not share parameters."
            )
        if storage_keys & owned_storage_keys:
            raise ForcedPathTrainingError(
                "Experts at different layers must not share parameter storage."
            )
        owned_parameter_ids.update(parameter_ids)
        owned_storage_keys.update(storage_keys)

    # Freezing is mechanical and precedes explicit expert re-enablement. The
    # shared backbone, anchor MLPs, routers, and any incumbent modules cannot
    # enter the optimizer through stale gradient flags.
    model.requires_grad_(False)
    expected_trainable_parameter_ids: set[int] = set()
    for binding in bindings:
        binding.expert.set_trainable(True)
        optimizable = tuple(
            parameter
            for parameter in binding.expert.parameters()
            if parameter.requires_grad
        )
        if not optimizable:
            raise ForcedPathTrainingError(
                f"Expert at layer {binding.layer_index} exposes no trainable "
                "parameters after set_trainable(True)."
            )
        expected_trainable_parameter_ids.update(
            id(parameter) for parameter in optimizable
        )
        wrappers[binding.layer_index].configure_forced_expert(
            layer_index=binding.layer_index,
            expert_id=binding.expert_id,
            expert=binding.expert,
        )

    trainable = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    actual_parameter_ids = {id(parameter) for _, parameter in trainable}
    if actual_parameter_ids != expected_trainable_parameter_ids:
        raise ForcedPathTrainingError(
            "Prepared model trainable parameters do not exactly match the "
            "expert-declared optimizer subsets."
        )

    return ForcedPathTrainingReport(
        layer_indices=binding_layers,
        expert_id=bindings[0].expert_id,
        trainable_parameter_names=tuple(name for name, _ in trainable),
        total_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        trainable_parameter_count=sum(
            parameter.numel() for _, parameter in trainable
        ),
    )


class ForcedPathExpertTrainer(Trainer):
    """Hugging Face Trainer with isolated forced-path expert optimization.

    Construction prepares the converted model with
    :func:`prepare_forced_path_training`. Before every :meth:`train` call, the
    trainer reasserts both the forced route topology and the exact trainable
    parameter identity set. This catches accidental unfreezing, expert
    replacement, or route reconfiguration before an optimizer step.

    TRN-005 optionally adds token-masked KL preservation against precomputed
    frozen-anchor logits and normalized squared drift from immutable expert
    initialization snapshots. Both terms are independently weighted and leave
    TRN-001's route and optimizer ownership unchanged.

    Full-model checkpoint saving is disabled for TRN-001: ``args.save_strategy``
    must be ``"no"``. Call each expert's safe serialization API explicitly;
    complete package export remains NB-003.

    Args:
        model: Converted Hugging Face causal language model.
        args: Training arguments with ``save_strategy="no"``.
        backend: Model-family adapter for routed-layer discovery.
        expert_bindings: One expert binding per converted layer.
        objective_config:
            Optional replay-KL and drift configuration. The all-zero default
            reproduces the ordinary TRN-001 model loss.
        trainer_kwargs: Remaining keyword arguments accepted by
            :class:`transformers.Trainer`, such as datasets and collators.

    References:
        Hugging Face Transformers Trainer documentation:
        https://huggingface.co/docs/transformers/main_classes/trainer
        Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural
        Network." https://arxiv.org/abs/1503.02531
        Li et al., "Explicit Inductive Bias for Transfer Learning with
        Convolutional Networks." https://arxiv.org/abs/1802.01483

    Differences from the referenced methods:
        The standard Trainer may optimize any parameter with
        ``requires_grad=True`` and may save the complete model. This subclass
        asserts an expert-only identity set and rejects full-model checkpoint
        saving to preserve the immutable-anchor package boundary.
        Preservation uses precomputed immutable-anchor logits and a token mask.
        L2-SP is restricted to independently owned expert parameters and
        normalized by their scalar parameter count. Teacher logits are batch
        data rather than a second live model, and drift snapshots are detached
        tensors outside model state.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        args: TrainingArguments,
        backend: ModelBackend,
        expert_bindings: Sequence[ForcedExpertBinding],
        objective_config: ExpertObjectiveConfig | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if not isinstance(args, TrainingArguments):
            raise TypeError("args must be transformers.TrainingArguments.")
        if args.save_strategy != SaveStrategy.NO:
            raise ForcedPathTrainingError(
                "TRN-001 requires save_strategy='no'; export experts through "
                "safe package APIs."
            )
        if args.push_to_hub:
            raise ForcedPathTrainingError(
                "TRN-001 does not publish full models; export an expert package "
                "through the dedicated package workflow."
            )
        if trainer_kwargs.get("model_init") is not None:
            raise ForcedPathTrainingError(
                "model_init is incompatible with explicit expert bindings."
            )
        if objective_config is not None and not isinstance(
            objective_config,
            ExpertObjectiveConfig,
        ):
            raise TypeError(
                "objective_config must be an ExpertObjectiveConfig or None."
            )
        resolved_objectives = objective_config or ExpertObjectiveConfig()
        self.objective_config = resolved_objectives
        self._replay_kl = (
            ReplayKLPreservation(resolved_objectives.replay_temperature)
            if resolved_objectives.replay_kl_enabled
            else None
        )
        self._weight_drift: ExpertWeightDrift | None = None
        self.last_objective_report: ExpertObjectiveReport | None = None

        bindings = _normalize_bindings(expert_bindings)
        report = prepare_forced_path_training(
            model,
            backend=backend,
            expert_bindings=bindings,
        )
        if resolved_objectives.drift_enabled:
            drift_parameters = tuple(
                (
                    f"layer_{binding.layer_index}.{name}",
                    parameter,
                )
                for binding in bindings
                for name, parameter in binding.expert.named_parameters()
                if parameter.requires_grad
            )
            self._weight_drift = ExpertWeightDrift(drift_parameters)
        try:
            # Experts must be registered before Trainer/Accelerate moves the
            # model so every forced path follows the same device policy.
            super().__init__(model=model, args=args, **trainer_kwargs)
        except Exception:
            for binding in bindings:
                references = tuple(backend.iter_mlp_references(model))
                wrappers = {
                    reference.layer_index: reference.module
                    for reference in references
                    if isinstance(reference.module, RoutedMLP)
                }
                wrappers[binding.layer_index].reset_to_forced_anchor()
            raise

        self.backend = backend
        self.expert_bindings = bindings
        self.forced_path_report = report
        self._expected_trainable_parameter_ids = frozenset(
            id(parameter)
            for binding in bindings
            for parameter in binding.expert.parameters()
            if parameter.requires_grad
        )
        self.assert_forced_path_invariants()

    def _set_signature_columns_if_needed(self) -> None:
        """Keep configured teacher and replay-mask fields in dataset columns."""

        super()._set_signature_columns_if_needed()
        if not self.objective_config.replay_kl_enabled:
            return
        if self._signature_columns is None:
            raise ForcedPathTrainingError(
                "Trainer did not resolve model input columns."
            )
        objective_columns = [self.objective_config.teacher_logits_key]
        if self.objective_config.replay_token_mask_key is not None:
            objective_columns.append(
                self.objective_config.replay_token_mask_key
            )
        # Stable append order preserves standard Trainer columns while making
        # the two explicitly configured auxiliary fields available to
        # compute_loss. Dataset pruning therefore remains enabled.
        self._signature_columns = list(
            dict.fromkeys([*self._signature_columns, *objective_columns])
        )

    def assert_forced_path_invariants(self) -> None:
        """Assert route ownership and the exact trainable parameter identity."""

        references = tuple(self.backend.iter_mlp_references(self.model))
        wrappers = {
            reference.layer_index: reference.module
            for reference in references
            if isinstance(reference.module, RoutedMLP)
        }
        route_errors: list[str] = []
        for binding in self.expert_bindings:
            wrapper = wrappers.get(binding.layer_index)
            if (
                wrapper is None
                or wrapper.routing_mode != "forced_expert"
                or wrapper.expert_ids != (binding.expert_id,)
                or len(wrapper.experts) != 1
                or wrapper.experts[0] is not binding.expert
            ):
                route_errors.append(str(binding.layer_index))

        actual_parameter_ids = frozenset(
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if (
            route_errors
            or actual_parameter_ids != self._expected_trainable_parameter_ids
        ):
            raise ForcedPathTrainingError(
                "Forced-path training invariants changed before optimization: "
                f"invalid_layers={route_errors}, "
                f"expected_trainable={len(self._expected_trainable_parameter_ids)}, "
                f"actual_trainable={len(actual_parameter_ids)}."
            )

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Any = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Reassert isolated training state and run the standard Trainer loop."""

        if resume_from_checkpoint not in (None, False):
            raise ForcedPathTrainingError(
                "Checkpoint resume is not supported by TRN-001 expert-only "
                "training."
            )
        self.assert_forced_path_invariants()
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Add independently weighted TRN-005 terms to the model loss.

        The configured teacher-logit and replay-mask fields are removed before
        delegating ordinary model-loss computation to
        :class:`transformers.Trainer`. Missing replay masks fall back to the
        model ``attention_mask`` and then to all token positions.
        """

        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a dictionary.")
        model_inputs = dict(inputs)
        teacher_key = self.objective_config.teacher_logits_key
        anchor_logits = model_inputs.pop(teacher_key, None)
        mask_key = self.objective_config.replay_token_mask_key
        replay_token_mask = (
            model_inputs.pop(mask_key, None)
            if mask_key is not None
            else None
        )
        if replay_token_mask is None:
            replay_token_mask = model_inputs.get("attention_mask")

        if (
            self.objective_config.replay_kl_enabled
            and anchor_logits is None
        ):
            raise ForcedPathTrainingError(
                "Replay KL is enabled but the batch is missing teacher logits "
                f"under {teacher_key!r}."
            )

        specialization_loss, outputs = super().compute_loss(
            model,
            model_inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if not isinstance(specialization_loss, Tensor):
            raise ForcedPathTrainingError(
                "The base Trainer loss must be a Tensor."
            )

        zero = specialization_loss.new_zeros(())
        replay_kl_loss = zero
        if self._replay_kl is not None:
            student_logits = getattr(outputs, "logits", None)
            if not isinstance(student_logits, Tensor):
                raise ForcedPathTrainingError(
                    "Replay KL requires model outputs with floating logits."
                )
            replay_kl_loss = self._replay_kl(
                student_logits,
                anchor_logits,
                token_mask=replay_token_mask,
            )

        drift_loss = (
            self._weight_drift()
            if self._weight_drift is not None
            else zero
        )
        total_loss = (
            specialization_loss
            + self.objective_config.replay_kl_weight * replay_kl_loss
            + self.objective_config.drift_weight * drift_loss
        )
        self.last_objective_report = ExpertObjectiveReport(
            specialization_loss=float(
                specialization_loss.detach().float().cpu().item()
            ),
            replay_kl_loss=float(
                replay_kl_loss.detach().float().cpu().item()
            ),
            drift_loss=float(drift_loss.detach().float().cpu().item()),
            total_loss=float(total_loss.detach().float().cpu().item()),
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        """Reject full-model saving; serialize expert packages explicitly."""

        raise ForcedPathTrainingError(
            "ForcedPathExpertTrainer does not save the immutable anchor; "
            "serialize experts with their safe package APIs."
        )


def _normalize_bindings(
    value: Sequence[ForcedExpertBinding],
) -> tuple[ForcedExpertBinding, ...]:
    """Validate, sort, and reject duplicate layer ownership."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "expert_bindings must be a sequence of ForcedExpertBinding."
        )
    if not value or not all(
        isinstance(binding, ForcedExpertBinding) for binding in value
    ):
        raise TypeError(
            "expert_bindings must be a non-empty sequence of "
            "ForcedExpertBinding."
        )
    normalized = tuple(sorted(value, key=lambda binding: binding.layer_index))
    layer_indices = [binding.layer_index for binding in normalized]
    if len(set(layer_indices)) != len(layer_indices):
        raise ForcedPathTrainingError(
            "expert_bindings must not repeat converted layers."
        )
    return normalized


def _parameter_storage_key(
    parameter: nn.Parameter,
) -> tuple[str, int | None, int]:
    """Identify underlying storage to reject aliased expert parameters."""

    device = parameter.device
    return (
        device.type,
        device.index,
        parameter.untyped_storage().data_ptr(),
    )
