"""Post-hoc mixture-aware refinement of independently trained expert pairs.

TRN-004 composes two existing training strategies without introducing a
second optimizer implementation:

1. TRN-002 independently trains two complete expert paths.
2. TRN-004 validates that completed topology against caller-owned expert
   bindings.
3. TRN-003 jointly refines the same experts and fresh package-local affine
   heads through the differentiable COM-001 score-softmax mixture.

The source report is provenance, not proof of Python object identity. Expert
weights remain independently owned artifacts, so the coordinator validates
their stable IDs and converted-layer rectangle while the delegated TRN-003
preflight enforces parameter and storage isolation.

This module implements repository-specific training orchestration rather than
a paper-derived algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from types import MappingProxyType
from typing import Any

from torch import nn
from transformers import Trainer, TrainingArguments

from anchored_moe.backends.base import ModelBackend
from anchored_moe.routers.base import RouterFeatureExtractor
from anchored_moe.routers.heads import AppendOnlyRouterHeads
from anchored_moe.training.expert_trainer import ForcedPathTrainingError
from anchored_moe.training.independent import (
    IndependentExpertSetTrainingReport,
)
from anchored_moe.training.mini_moe import (
    LocalMiniMoEExpertBinding,
    LocalMiniMoETrainer,
    LocalMiniMoETrainingConfig,
    LocalMiniMoETrainingReport,
)


_RESERVED_TRAINER_KWARGS = frozenset(
    {
        "args",
        "backend",
        "config",
        "expert_bindings",
        "model",
        "model_init",
        "router_feature_extractor",
        "router_heads",
    }
)
_TRAINER_INIT_KWARGS = frozenset(signature(Trainer.__init__).parameters) - {
    "self"
}


class MixtureAwareRefinementError(ForcedPathTrainingError):
    """Raised when TRN-004 provenance or refinement inputs are incompatible."""


@dataclass(frozen=True)
class MixtureAwareRefinementReport:
    """Immutable link between independent training and local refinement.

    Attributes:
        source_training:
            Completed TRN-002 report whose two expert paths provide the
            declared provenance for refinement.
        refinement:
            Completed TRN-003 report for the joint expert-and-router update.

    The report requires exact expert order and converted-layer equality. This
    makes the transition auditable without retaining models, datasets,
    optimizers, or mutable expert objects.
    """

    source_training: IndependentExpertSetTrainingReport
    refinement: LocalMiniMoETrainingReport

    def __post_init__(self) -> None:
        """Validate the exact two-expert topology across both training stages."""

        if not isinstance(
            self.source_training,
            IndependentExpertSetTrainingReport,
        ):
            raise TypeError(
                "source_training must be an "
                "IndependentExpertSetTrainingReport."
            )
        if not isinstance(self.refinement, LocalMiniMoETrainingReport):
            raise TypeError(
                "refinement must be a LocalMiniMoETrainingReport."
            )
        if len(self.source_training.expert_ids) != 2:
            raise MixtureAwareRefinementError(
                "TRN-004 requires exactly two independently trained experts."
            )
        if (
            self.source_training.expert_ids
            != self.refinement.topology.expert_ids
        ):
            raise MixtureAwareRefinementError(
                "Refinement expert order must exactly match the TRN-002 "
                "source report."
            )
        if (
            self.source_training.layer_indices
            != self.refinement.topology.layer_indices
        ):
            raise MixtureAwareRefinementError(
                "Refinement layers must exactly match the TRN-002 source "
                "report."
            )

    @property
    def expert_ids(self) -> tuple[str, str]:
        """Return the stable two-expert order shared by both stages."""

        first, second = self.source_training.expert_ids
        return first, second

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return the converted layers shared by both stages."""

        return self.source_training.layer_indices

    @property
    def total_global_steps(self) -> int:
        """Return independent and joint optimizer steps summed together."""

        return (
            self.source_training.total_global_steps
            + self.refinement.global_step
        )


def refine_independent_expert_set(
    model: nn.Module,
    *,
    backend: ModelBackend,
    source_training: IndependentExpertSetTrainingReport,
    expert_bindings: Sequence[LocalMiniMoEExpertBinding],
    router_feature_extractor: RouterFeatureExtractor,
    router_heads: AppendOnlyRouterHeads,
    args: TrainingArguments,
    config: LocalMiniMoETrainingConfig | None = None,
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> MixtureAwareRefinementReport:
    """Jointly refine one completed two-expert TRN-002 set with TRN-003.

    TRN-004 first validates the source report, exact expert order, complete
    converted-layer rectangle, and ordinary Trainer keyword boundary without
    changing model state. It then constructs one
    :class:`LocalMiniMoETrainer`, whose established preflight proves parameter
    and storage independence before attaching the differentiable local route.

    Every specialization token executes both experts at every converted layer:

    ``y[l,t] = alpha[l,t,0] * E[l,0](h[l,t])``
    ``         + alpha[l,t,1] * E[l,1](h[l,t])``

    where ``alpha`` is COM-001 score-softmax over fresh package-local affine
    heads. The immutable anchor and router feature transformation remain
    frozen. On success or failure, TRN-003 clears gradients, freezes the
    experts and heads, and restores exact forced-anchor routing. Partial
    optimizer updates are deliberately not rolled back.

    Args:
        model:
            Converted Hugging Face causal language model in forced-anchor mode.
        backend:
            Model-family adapter for deterministic routed-layer discovery.
        source_training:
            Completed TRN-002 report for exactly two experts.
        expert_bindings:
            Caller-owned experts covering every source expert/layer pair.
        router_feature_extractor:
            Frozen feature transformation consumed by fresh local heads.
        router_heads:
            Fresh zero-initialized affine heads in source expert order.
        args:
            Ordinary Hugging Face TrainingArguments with ``save_strategy="no"``.
        config:
            Existing TRN-003 mixture and optional BAL-003 configuration.
        trainer_kwargs:
            Optional ordinary ``Trainer`` inputs such as ``train_dataset`` and
            ``data_collator``. Coordinator-owned arguments are rejected.

    Returns:
        An immutable report linking the independent and joint training stages.

    Raises:
        TypeError:
            If public inputs do not implement their declared interfaces.
        MixtureAwareRefinementError:
            If source provenance, binding topology, or Trainer arguments are
            incompatible.
        ForcedPathTrainingError:
            If delegated TRN-003 ownership or lifecycle invariants fail.

    This function is framework orchestration rather than a paper-derived
    algorithm. It implements the architecture blueprint's independent-
    training-then-mixture-refinement strategy using existing public runtimes.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must be a ModelBackend.")
    if not isinstance(source_training, IndependentExpertSetTrainingReport):
        raise TypeError(
            "source_training must be an IndependentExpertSetTrainingReport."
        )
    bindings = _validate_source_topology(
        source_training,
        expert_bindings=expert_bindings,
        router_heads=router_heads,
    )
    normalized_trainer_kwargs = _validate_trainer_kwargs(trainer_kwargs)

    trainer = LocalMiniMoETrainer(
        model=model,
        args=args,
        backend=backend,
        expert_bindings=bindings,
        router_feature_extractor=router_feature_extractor,
        router_heads=router_heads,
        config=config,
        **dict(normalized_trainer_kwargs),
    )
    trainer.train()
    if not isinstance(trainer.training_report, LocalMiniMoETrainingReport):
        raise MixtureAwareRefinementError(
            "TRN-003 completed without a local refinement report."
        )
    return MixtureAwareRefinementReport(
        source_training=source_training,
        refinement=trainer.training_report,
    )


def _validate_source_topology(
    source_training: IndependentExpertSetTrainingReport,
    *,
    expert_bindings: Sequence[LocalMiniMoEExpertBinding],
    router_heads: AppendOnlyRouterHeads,
) -> tuple[LocalMiniMoEExpertBinding, ...]:
    """Validate TRN-002 provenance before delegated model mutation."""

    if len(source_training.expert_ids) != 2:
        raise MixtureAwareRefinementError(
            "TRN-004 requires exactly two independently trained experts; "
            f"source={source_training.expert_ids}."
        )
    if isinstance(expert_bindings, (str, bytes)) or not isinstance(
        expert_bindings,
        Sequence,
    ):
        raise TypeError(
            "expert_bindings must be a sequence of "
            "LocalMiniMoEExpertBinding values."
        )
    bindings = tuple(expert_bindings)
    if not bindings or not all(
        isinstance(binding, LocalMiniMoEExpertBinding)
        for binding in bindings
    ):
        raise TypeError(
            "expert_bindings must contain only "
            "LocalMiniMoEExpertBinding values."
        )

    by_key: dict[tuple[int, str], LocalMiniMoEExpertBinding] = {}
    for binding in bindings:
        if binding.key in by_key:
            raise MixtureAwareRefinementError(
                f"TRN-004 repeats expert binding {binding.key!r}."
            )
        by_key[binding.key] = binding

    expected_keys = tuple(
        (layer_index, expert_id)
        for layer_index in source_training.layer_indices
        for expert_id in source_training.expert_ids
    )
    if set(by_key) != set(expected_keys) or len(by_key) != len(expected_keys):
        raise MixtureAwareRefinementError(
            "Refinement bindings must exactly cover the TRN-002 expert/layer "
            f"rectangle: expected={expected_keys}, actual={tuple(by_key)}."
        )
    if not isinstance(router_heads, AppendOnlyRouterHeads):
        raise TypeError("router_heads must be AppendOnlyRouterHeads.")
    if router_heads.head_keys != expected_keys:
        raise MixtureAwareRefinementError(
            "Refinement router heads must preserve the exact TRN-002 "
            f"layer/expert order: expected={expected_keys}, "
            f"actual={router_heads.head_keys}."
        )
    # Layer-major order followed by the source report's stable expert order is
    # passed to TRN-003, preventing caller mapping order from changing router
    # symmetry breaking or the audit report.
    return tuple(by_key[key] for key in expected_keys)


def _validate_trainer_kwargs(
    trainer_kwargs: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Freeze and validate Trainer inputs before local route attachment."""

    if trainer_kwargs is None:
        return MappingProxyType({})
    if not isinstance(trainer_kwargs, Mapping):
        raise TypeError("trainer_kwargs must be a mapping or None.")

    values = dict(trainer_kwargs)
    invalid_keys = sorted(
        repr(key)
        for key in values
        if not isinstance(key, str) or key not in _TRAINER_INIT_KWARGS
    )
    if invalid_keys:
        raise MixtureAwareRefinementError(
            "trainer_kwargs contains arguments not accepted by "
            f"transformers.Trainer: {invalid_keys}."
        )
    reserved = sorted(_RESERVED_TRAINER_KWARGS & set(values))
    if reserved:
        raise MixtureAwareRefinementError(
            "trainer_kwargs cannot override coordinator-owned arguments: "
            f"{reserved}."
        )
    return MappingProxyType(values)
