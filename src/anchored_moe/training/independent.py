"""Independent latent expert-set training against one immutable anchor."""

from __future__ import annotations

import math
from inspect import signature
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SaveStrategy

from anchored_moe.backends.base import ModelBackend
from anchored_moe.modeling import RoutedMLP
from anchored_moe.training.expert_trainer import (
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    ForcedPathTrainingError,
    ForcedPathTrainingReport,
)
from anchored_moe.training.objectives import (
    ExpertObjectiveConfig,
    ExpertObjectiveReport,
)


_RESERVED_TRAINER_KWARGS = frozenset(
    {
        "args",
        "backend",
        "expert_bindings",
        "model",
        "model_init",
        "objective_config",
    }
)
_TRAINER_INIT_KWARGS = frozenset(signature(Trainer.__init__).parameters) - {
    "self"
}


@dataclass(frozen=True)
class IndependentExpertTrainingRun:
    """Configuration for one independently optimized forced expert path.

    ``expert_bindings`` must form one complete layer path with a single stable
    expert identifier. ``args``, ``objective_config``, and ``trainer_kwargs``
    belong only to this run, so callers may supply a different dataset,
    collator, seed, schedule, and preservation objective for every latent
    expert.

    The coordinator preserves the declared run order. Data partitioning is
    deliberately caller-owned: a run receives ordinary Hugging Face Trainer
    inputs rather than embedding one partitioning algorithm into TRN-002.
    """

    expert_bindings: Sequence[ForcedExpertBinding]
    args: TrainingArguments
    objective_config: ExpertObjectiveConfig | None = None
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze run inputs and reject failures before model mutation."""

        bindings = _normalize_run_bindings(self.expert_bindings)
        if not isinstance(self.args, TrainingArguments):
            raise TypeError("args must be transformers.TrainingArguments.")
        if self.args.save_strategy != SaveStrategy.NO:
            raise ForcedPathTrainingError(
                "TRN-002 requires save_strategy='no' for every expert run."
            )
        if self.args.push_to_hub:
            raise ForcedPathTrainingError(
                "TRN-002 does not publish full models; export the completed "
                "latent expert set through the package workflow."
            )
        if self.objective_config is not None and not isinstance(
            self.objective_config,
            ExpertObjectiveConfig,
        ):
            raise TypeError(
                "objective_config must be an ExpertObjectiveConfig or None."
            )
        if not isinstance(self.trainer_kwargs, Mapping):
            raise TypeError("trainer_kwargs must be a mapping.")
        invalid_keys = sorted(
            repr(key)
            for key in self.trainer_kwargs
            if not isinstance(key, str) or key not in _TRAINER_INIT_KWARGS
        )
        if invalid_keys:
            raise ValueError(
                "trainer_kwargs contains arguments not accepted by "
                f"transformers.Trainer: {invalid_keys}."
            )
        reserved = sorted(_RESERVED_TRAINER_KWARGS & set(self.trainer_kwargs))
        if reserved:
            raise ValueError(
                "trainer_kwargs cannot override coordinator-owned arguments: "
                f"{reserved}."
            )

        object.__setattr__(self, "expert_bindings", bindings)
        object.__setattr__(
            self,
            "trainer_kwargs",
            MappingProxyType(dict(self.trainer_kwargs)),
        )

    @property
    def expert_id(self) -> str:
        """Return the stable expert identifier shared by every layer binding."""

        return self.expert_bindings[0].expert_id

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return this expert path's converted layers in deterministic order."""

        return tuple(
            binding.layer_index for binding in self.expert_bindings
        )


@dataclass(frozen=True)
class IndependentExpertTrainingReport:
    """Immutable result for one completed independent expert run.

    Attributes:
        expert_id: Stable latent-expert identifier.
        forced_path: Exact TRN-001 topology and trainable-parameter report.
        global_step: Number of optimizer steps completed by Trainer.
        training_loss: Final aggregate loss reported by Trainer.
        objective: Last detached TRN-005 component report, when available.
    """

    expert_id: str
    forced_path: ForcedPathTrainingReport
    global_step: int
    training_loss: float
    objective: ExpertObjectiveReport | None

    def __post_init__(self) -> None:
        """Validate identity, optimization progress, and finite loss."""

        if (
            not isinstance(self.expert_id, str)
            or not self.expert_id
            or self.expert_id != self.expert_id.strip()
        ):
            raise ValueError(
                "expert_id must be a non-empty string without surrounding "
                "whitespace."
            )
        if not isinstance(self.forced_path, ForcedPathTrainingReport):
            raise TypeError(
                "forced_path must be a ForcedPathTrainingReport."
            )
        if self.forced_path.expert_id != self.expert_id:
            raise ValueError(
                "forced_path expert_id must match the independent run."
            )
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step <= 0
        ):
            raise ValueError("global_step must be a positive integer.")
        if (
            isinstance(self.training_loss, bool)
            or not isinstance(self.training_loss, (int, float))
        ):
            raise TypeError("training_loss must be a real number.")
        normalized_loss = float(self.training_loss)
        if not math.isfinite(normalized_loss):
            raise ValueError("training_loss must be finite.")
        object.__setattr__(self, "training_loss", normalized_loss)
        if self.objective is not None and not isinstance(
            self.objective,
            ExpertObjectiveReport,
        ):
            raise TypeError(
                "objective must be an ExpertObjectiveReport or None."
            )


@dataclass(frozen=True)
class IndependentExpertSetTrainingReport:
    """Immutable aggregate report for a completed latent expert set.

    Reports retain the caller-declared expert order. Every expert must cover
    the same converted layers, and at least two reports are required because a
    one-expert path is already represented by TRN-001.
    """

    reports: tuple[IndependentExpertTrainingReport, ...]

    def __post_init__(self) -> None:
        """Validate expert uniqueness and rectangular layer coverage."""

        if (
            not isinstance(self.reports, tuple)
            or len(self.reports) < 2
            or not all(
                isinstance(report, IndependentExpertTrainingReport)
                for report in self.reports
            )
        ):
            raise TypeError(
                "reports must be a tuple of at least two "
                "IndependentExpertTrainingReport values."
            )
        expert_ids = tuple(report.expert_id for report in self.reports)
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("Independent expert reports must have unique IDs.")
        layer_indices = self.reports[0].forced_path.layer_indices
        if any(
            report.forced_path.layer_indices != layer_indices
            for report in self.reports[1:]
        ):
            raise ValueError(
                "Independent expert reports must cover identical layers."
            )

    @property
    def expert_ids(self) -> tuple[str, ...]:
        """Return completed latent experts in declared training order."""

        return tuple(report.expert_id for report in self.reports)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return the common converted-layer topology."""

        return self.reports[0].forced_path.layer_indices

    @property
    def total_global_steps(self) -> int:
        """Return optimizer steps summed across independent runs."""

        return sum(report.global_step for report in self.reports)


def train_independent_expert_set(
    model: nn.Module,
    *,
    backend: ModelBackend,
    runs: Sequence[IndependentExpertTrainingRun],
) -> IndependentExpertSetTrainingReport:
    """Train two or more expert paths independently on one frozen anchor.

    TRN-002 validates the complete layer-by-expert rectangle and all parameter
    ownership before constructing the first Trainer. It then executes each run
    through :class:`ForcedPathExpertTrainer` in declared order. Only one expert
    path is attached at a time; no router score, expert mixture, diversity
    loss, or cross-expert gradient is introduced.

    Routed wrappers are restored to exact ``forced_anchor`` mode and every
    model and expert parameter is frozen after each run and after exceptions.
    Cleanup is failure-safe but not transactional: if an optimizer fails after
    updating an expert, TRN-002 does not copy and roll back that expert's
    potentially very large weights.

    Args:
        model: Converted Hugging Face causal language model shared by all runs.
        backend: Model-family adapter for deterministic routed-layer discovery.
        runs: At least two independently configured expert training runs.

    Returns:
        Immutable per-expert reports in the declared run order.

    Raises:
        TypeError: If model, backend, runs, or their fields have invalid types.
        ForcedPathTrainingError:
            If layer coverage, expert identity, storage ownership, routing
            state, Trainer isolation, or optimization results are unsafe.

    This is framework infrastructure rather than a paper-derived training
    algorithm. It composes the TRN-001 Hugging Face Trainer boundary while
    leaving data partitioning and process-level parallelism to callers.
    """

    normalized_runs, wrappers = _validate_expert_set(
        model,
        backend=backend,
        runs=runs,
    )
    completed: list[IndependentExpertTrainingReport] = []
    try:
        for run in normalized_runs:
            trainer: ForcedPathExpertTrainer | None = None
            try:
                trainer = ForcedPathExpertTrainer(
                    model=model,
                    args=run.args,
                    backend=backend,
                    expert_bindings=run.expert_bindings,
                    objective_config=run.objective_config,
                    **dict(run.trainer_kwargs),
                )
                output = trainer.train()
                completed.append(
                    IndependentExpertTrainingReport(
                        expert_id=run.expert_id,
                        forced_path=trainer.forced_path_report,
                        global_step=output.global_step,
                        training_loss=output.training_loss,
                        objective=trainer.last_objective_report,
                    )
                )
            finally:
                # Trainer may fail after attaching or partially optimizing an
                # expert. Always detach the route before another expert run or
                # control returns to the caller.
                _restore_anchor_only_state(
                    model,
                    wrappers=wrappers,
                    runs=normalized_runs,
                )
                trainer = None
    finally:
        # The outer guard covers failures between runs and cleanup errors from
        # future Trainer integrations without weakening exact anchor fallback.
        _restore_anchor_only_state(
            model,
            wrappers=wrappers,
            runs=normalized_runs,
        )

    return IndependentExpertSetTrainingReport(reports=tuple(completed))


def _validate_expert_set(
    model: nn.Module,
    *,
    backend: ModelBackend,
    runs: Sequence[IndependentExpertTrainingRun],
) -> tuple[
    tuple[IndependentExpertTrainingRun, ...],
    tuple[tuple[int, RoutedMLP], ...],
]:
    """Validate the full rectangular topology and ownership before training."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must be a ModelBackend.")
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise TypeError(
            "runs must be a sequence of IndependentExpertTrainingRun values."
        )
    normalized_runs = tuple(runs)
    if len(normalized_runs) < 2 or not all(
        isinstance(run, IndependentExpertTrainingRun)
        for run in normalized_runs
    ):
        raise TypeError(
            "runs must contain at least two IndependentExpertTrainingRun "
            "values."
        )

    expert_ids = tuple(run.expert_id for run in normalized_runs)
    if len(set(expert_ids)) != len(expert_ids):
        raise ForcedPathTrainingError(
            "Independent expert runs must use unique expert IDs."
        )
    expected_layers = normalized_runs[0].layer_indices
    if any(run.layer_indices != expected_layers for run in normalized_runs[1:]):
        raise ForcedPathTrainingError(
            "Every independent expert must cover the same converted layers."
        )

    try:
        references = tuple(backend.iter_mlp_references(model))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ForcedPathTrainingError(
            f"Cannot discover model MLPs for independent training: {error}"
        ) from error
    wrappers = tuple(
        (reference.layer_index, reference.module)
        for reference in references
        if isinstance(reference.module, RoutedMLP)
    )
    wrapper_layers = tuple(layer_index for layer_index, _ in wrappers)
    if wrapper_layers != expected_layers:
        raise ForcedPathTrainingError(
            "Every independent expert must exactly cover converted RoutedMLP "
            f"layers: model={wrapper_layers}, bindings={expected_layers}."
        )
    invalid_modes = tuple(
        layer_index
        for layer_index, wrapper in wrappers
        if wrapper.routing_mode != "forced_anchor"
    )
    if invalid_modes:
        raise ForcedPathTrainingError(
            "Independent training requires exact forced_anchor mode before "
            f"the first run; invalid_layers={invalid_modes}."
        )

    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    model_storage_keys = {
        _parameter_storage_key(parameter) for parameter in model.parameters()
    }
    owned_parameter_ids: set[int] = set()
    owned_storage_keys: set[tuple[str, int | None, int]] = set()
    for run in normalized_runs:
        for binding in run.expert_bindings:
            parameters = tuple(binding.expert.parameters())
            if not parameters:
                raise ForcedPathTrainingError(
                    f"Expert {run.expert_id!r} at layer "
                    f"{binding.layer_index} has no parameters."
                )
            parameter_ids = {id(parameter) for parameter in parameters}
            storage_keys = {
                _parameter_storage_key(parameter) for parameter in parameters
            }
            if len(parameter_ids) != len(parameters):
                raise ForcedPathTrainingError(
                    f"Expert {run.expert_id!r} repeats parameters."
                )
            if len(storage_keys) != len(parameters):
                raise ForcedPathTrainingError(
                    f"Expert {run.expert_id!r} shares parameter storage."
                )
            if parameter_ids & model_parameter_ids:
                raise ForcedPathTrainingError(
                    f"Expert {run.expert_id!r} shares model parameters."
                )
            if storage_keys & model_storage_keys:
                raise ForcedPathTrainingError(
                    f"Expert {run.expert_id!r} shares model storage."
                )
            if parameter_ids & owned_parameter_ids:
                raise ForcedPathTrainingError(
                    "Independent experts must not share parameters."
                )
            if storage_keys & owned_storage_keys:
                raise ForcedPathTrainingError(
                    "Independent experts must not share parameter storage."
                )
            owned_parameter_ids.update(parameter_ids)
            owned_storage_keys.update(storage_keys)

    return normalized_runs, wrappers


def _restore_anchor_only_state(
    model: nn.Module,
    *,
    wrappers: tuple[tuple[int, RoutedMLP], ...],
    runs: tuple[IndependentExpertTrainingRun, ...],
) -> None:
    """Detach all expert paths, clear gradients, and freeze complete state."""

    model.zero_grad(set_to_none=True)
    for _, wrapper in wrappers:
        wrapper.reset_to_forced_anchor()
    model.requires_grad_(False)
    for run in runs:
        for binding in run.expert_bindings:
            binding.expert.requires_grad_(False)
            for parameter in binding.expert.parameters():
                parameter.grad = None


def _normalize_run_bindings(
    value: Sequence[ForcedExpertBinding],
) -> tuple[ForcedExpertBinding, ...]:
    """Normalize one complete expert path without touching a model."""

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
    layer_indices = tuple(binding.layer_index for binding in normalized)
    if len(set(layer_indices)) != len(layer_indices):
        raise ForcedPathTrainingError(
            "One independent expert run must not repeat converted layers."
        )
    expert_ids = {binding.expert_id for binding in normalized}
    if len(expert_ids) != 1:
        raise ForcedPathTrainingError(
            "One independent expert run must use one expert_id at every layer."
        )
    return normalized


def _parameter_storage_key(
    parameter: nn.Parameter,
) -> tuple[str, int | None, int]:
    """Identify parameter storage so views cannot cross expert boundaries."""

    device = parameter.device
    return (
        device.type,
        device.index,
        parameter.untyped_storage().data_ptr(),
    )
