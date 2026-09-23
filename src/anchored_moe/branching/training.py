"""Composable objectives and isolated, failure-atomic JAX branch training.

No JAX or Optax imports occur until an objective/strategy executes. This module
does not generate data, run online teachers, calibrate thresholds or export models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .parameters import checked_parameters
from .routing import LinearRequestScorer


class BranchObjective(Protocol):
    """A differentiable scalar objective over backend-native predictions."""

    def validate(self, *args, **kwargs):
        """Reject invalid targets before tracing; arguments follow the training path."""
        ...

    def loss(self, prediction, target, **kwargs):
        """Return a scalar loss without mutating tensors or detaching gradients."""
        ...


class BranchTrainingStrategy(Protocol):
    """Optimize only the explicitly supplied parameter tree, then return it."""

    def optimize(self, parameters, loss_fn: Callable):
        """Return candidate parameters and a report; never publish partial steps."""
        ...


@dataclass(frozen=True)
class BranchTrainingReport:
    """Finite optimization diagnostics; counts cover the exact optimizer tree."""

    steps: int
    trainable_parameters: int
    initial_loss: float
    final_loss: float


@dataclass(frozen=True)
class JaxAdamStrategy:
    """Full-batch Adam over caller-owned branch/head tensors only.

    steps and learning_rate must be positive. max_grad_norm clips global norm;
    jit compiles the update closure. No model/base tensors enter optimizer state.
    Failure returns no candidate and leaves the owner unchanged.
    """

    steps: int = 20
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    jit: bool = True

    def __post_init__(self):
        """Validate optimizer controls before any device allocation."""
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps <= 0
        ):
            raise ValueError("steps must be a positive integer.")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.learning_rate, self.max_grad_norm)
        ):
            raise ValueError(
                "Optimizer rate and clipping norm must be finite and positive."
            )

    def optimize(self, parameters, loss_fn):
        """Run Adam and reject nonfinite losses, gradients or resulting weights."""
        import jax
        import jax.numpy as jnp
        import optax

        parameters = jax.tree.map(jnp.asarray, parameters)
        count = sum(v.size for v in jax.tree.leaves(parameters))
        if not count:
            raise ValueError("An optimizer needs a nonempty owned parameter tree.")
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )
        opt_state = optimizer.init(parameters)
        initial = float(loss_fn(parameters))
        if not np.isfinite(initial):
            raise ValueError("Nonfinite initial objective.")

        def update(p, state):
            loss, grads = jax.value_and_grad(loss_fn)(p)
            finite = jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(grads)])
            )
            updates, state = optimizer.update(grads, state, p)
            return optax.apply_updates(p, updates), state, loss, finite

        step = jax.jit(update) if self.jit else update
        for _ in range(self.steps):
            parameters, opt_state, loss, finite = step(parameters, opt_state)
            if not np.isfinite(float(loss)) or not bool(finite):
                raise ValueError("Nonfinite optimization step; candidate discarded.")
        final = float(loss_fn(parameters))
        if not np.isfinite(final) or any(
            not np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(parameters)
        ):
            raise ValueError("Nonfinite optimized candidate.")
        return parameters, BranchTrainingReport(self.steps, count, initial, final)


@dataclass(frozen=True)
class ResponseObjective:
    """Masked causal-LM loss with optional precomputed-anchor replay KL.

    Labels [1, T] align with input tokens; -100 masks prompt/non-target positions.
    Logits [1, T, V] predict labels at the next position. Teacher logits share
    that exact shape and token alignment. KL uses the same shifted target mask.
    """

    replay_kl_weight: float = 0.0
    temperature: float = 1.0

    def __post_init__(self):
        """Reject negative replay weight and nonpositive/nonfinite temperature."""
        if (
            not np.isfinite(self.replay_kl_weight)
            or self.replay_kl_weight < 0
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Invalid replay objective configuration.")

    def validate(self, prediction, target, teacher_logits=None):
        """Check shifted label coverage, vocabulary bounds and teacher alignment."""
        target = np.asarray(target)
        if (
            target.shape != prediction.shape[:2]
            or target.dtype.kind not in "iu"
            or target.shape[1] < 2
        ):
            raise ValueError("Expected integer [1, sequence] labels aligned to logits.")
        valid = target[:, 1:] != -100
        if (
            not valid.any()
            or (
                (target[target != -100] < 0)
                | (target[target != -100] >= prediction.shape[-1])
            ).any()
        ):
            raise ValueError(
                "Need valid next-token labels, or -100 for masked positions."
            )
        if self.replay_kl_weight and (
            teacher_logits is None
            or np.shape(teacher_logits) != prediction.shape
            or not np.isfinite(np.asarray(teacher_logits)).all()
        ):
            raise ValueError(
                "Replay requires finite aligned precomputed teacher logits."
            )

    def loss(self, prediction, target, *, teacher_logits=None):
        """Compute shifted masked NLL plus T-squared-scaled teacher KL."""
        import jax
        import jax.numpy as jnp

        target = jnp.asarray(target)[:, 1:]
        mask = target != -100
        safe = jnp.where(mask, target, 0)
        logp = jax.nn.log_softmax(prediction[:, :-1].astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
        result = jnp.sum(jnp.where(mask, nll, 0)) / jnp.sum(mask)
        if self.replay_kl_weight:
            teacher = jax.nn.log_softmax(
                jnp.asarray(teacher_logits)[:, :-1] / self.temperature, axis=-1
            )
            student = jax.nn.log_softmax(prediction[:, :-1] / self.temperature, axis=-1)
            kl = jnp.sum(jnp.exp(teacher) * (teacher - student), axis=-1)
            result += (
                self.replay_kl_weight
                * self.temperature**2
                * jnp.sum(jnp.where(mask, kl, 0))
                / jnp.sum(mask)
            )
        return result


@dataclass(frozen=True)
class UtilityRegressionObjective:
    """Mean squared error against measured per-specialist improvement over base."""

    kind: str = "utility"

    def validate(self, target, candidates):
        """Require one finite utility per ordered specialist."""
        if np.shape(target) != (candidates,) or not np.isfinite(target).all():
            raise ValueError("Expected a finite utility vector in candidate order.")

    def loss(self, prediction, target):
        """Return scalar mean squared utility error."""
        import jax.numpy as jnp

        return jnp.mean((prediction - jnp.asarray(target)) ** 2)


@dataclass(frozen=True)
class CategoricalRouterObjective:
    """Cross-entropy on a declared candidate index, including the base class."""

    kind: str = "probability"

    def validate(self, target, candidates):
        """Require an integer candidate index in the declared class order."""
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, np.integer))
            or not 0 <= target < candidates
        ):
            raise ValueError("Expected a valid categorical route index.")

    def loss(self, prediction, target):
        """Return negative log probability of the supplied class index."""
        import jax

        return -jax.nn.log_softmax(prediction)[target]


@dataclass(frozen=True)
class ResponseTrainingExample:
    """One unpadded teacher-forced sequence, labels and optional frozen logits."""

    input_ids: object
    labels: object
    teacher_logits: object | None = None


@dataclass(frozen=True)
class RouterTrainingExample:
    """Pre-response tokens plus utility vector or categorical branch index.

    input_ids must exclude the response being predicted. Caller owns rendering
    and data provenance; the trainer cannot infer answer leakage from tokens.
    """

    input_ids: object
    target: object


class BranchTrainer:
    """Compose backend execution, objective and optimizer with atomic publication.

    Prefix states are computed once per example and held frozen for the run.
    The default strategy uses full-batch JAX Adam; other strategies implement
    optimize(parameters, loss_fn). Only owned branch/head tensors are passed in.
    """

    def __init__(self, strategy: BranchTrainingStrategy | None = None):
        """Bind an independent optimizer strategy."""
        self.strategy = strategy or JaxAdamStrategy()

    def train_specialist(
        self,
        model,
        branch_id: str,
        examples,
        *,
        data_id: str,
        objective: BranchObjective | None = None,
        drift_weight: float = 0.0,
    ) -> BranchTrainingReport:
        """Train one forced response suffix, preserving every other branch.

        drift_weight penalizes parameter-count-normalized squared departure from
        the start of this call. Failure publishes nothing. data_id is recorded to
        reject reuse of a known fitting dataset for calibration.
        """
        import jax.numpy as jnp

        objective = objective or ResponseObjective()
        if (
            not isinstance(data_id, str)
            or not data_id.strip()
            or not np.isfinite(drift_weight)
            or drift_weight < 0
        ):
            raise ValueError(
                "Provide a data identity and nonnegative finite drift weight."
            )
        with model.operation():
            self._check_data(model, data_id)
            if (
                branch_id not in model.response_ids
                or model._configs[branch_id].parameterization == "frozen"
            ):
                raise ValueError("Training requires a trainable response branch.")
            initial = model.backend.native_parameters(
                model.branch_parameters(branch_id)
            )
            prepared = []
            for example in examples:
                state = model.backend.prefill(example.input_ids)
                labels = np.array(example.labels, copy=True)
                teacher = (
                    None
                    if example.teacher_logits is None
                    else np.array(example.teacher_logits, copy=True)
                )
                pred = model.backend.logits(state, model.effective_suffix(branch_id))
                objective.validate(pred, labels, teacher)
                prepared.append(
                    (
                        state,
                        jnp.asarray(labels),
                        None if teacher is None else jnp.asarray(teacher),
                    )
                )
            if not prepared:
                raise ValueError("Training examples cannot be empty.")

            def loss_fn(parameters):
                suffix = model.effective_suffix(branch_id, parameters)
                losses = [
                    objective.loss(
                        model.backend.logits(state, suffix),
                        labels,
                        teacher_logits=teacher,
                    )
                    for state, labels, teacher in prepared
                ]
                loss = sum(losses) / len(losses)
                if drift_weight:
                    loss += (
                        drift_weight
                        * sum(
                            jnp.sum((parameters[k] - initial[k]) ** 2) for k in initial
                        )
                        / sum(v.size for v in initial.values())
                    )
                return loss

            candidate, report = self.strategy.optimize(initial, loss_fn)
            model._publish_branch(branch_id, candidate)
            model._training_data_ids.add(data_id)
            return report

    def train_router(
        self,
        model,
        examples,
        *,
        data_id: str,
        objective=None,
        train_suffix: bool = True,
    ) -> BranchTrainingReport:
        """Train the affine head and optionally its private router suffix.

        Utility and categorical objectives must match scorer semantics. Response
        branches and the prefix are never optimizer arguments. Publishing both
        the suffix and head is atomic, after complete finite/topology validation.
        """
        import jax.numpy as jnp

        if not isinstance(data_id, str) or not data_id.strip():
            raise ValueError("Provide a nonempty fitting data identity.")
        with model.operation():
            self._check_data(model, data_id)
            scorer = model.scorer
            if not isinstance(scorer, LinearRequestScorer):
                raise TypeError("Builtin router training requires an affine scorer.")
            objective = objective or (
                UtilityRegressionObjective()
                if scorer.kind == "utility"
                else CategoricalRouterObjective()
            )
            if objective.kind != scorer.kind:
                raise ValueError("Router objective and score semantics differ.")
            router_id = (
                model.config.router_branch.branch_id
                if model.config.router_branch
                else None
            )
            train_router_suffix = bool(
                train_suffix
                and router_id
                and model._configs[router_id].parameterization != "frozen"
            )
            initial = {"head": model.backend.native_parameters(scorer.parameters)}
            if train_router_suffix:
                initial["suffix"] = model.backend.native_parameters(
                    model.branch_parameters(router_id)
                )
            prepared = []
            for example in examples:
                objective.validate(example.target, len(scorer.branch_ids))
                target = (
                    np.array(example.target, copy=True)
                    if scorer.kind == "utility"
                    else int(example.target)
                )
                prepared.append((model.backend.prefill(example.input_ids), target))
            if not prepared:
                raise ValueError("Router training examples cannot be empty.")

            def loss_fn(parameters):
                suffix = (
                    model.effective_suffix(router_id, parameters["suffix"])
                    if train_router_suffix
                    else (model.effective_suffix(router_id) if router_id else None)
                )
                losses = []
                for state, target in prepared:
                    features = model.backend.features(state, suffix)
                    raw = (
                        features @ parameters["head"]["weight"]
                        + parameters["head"]["bias"]
                    )
                    if scorer.kind == "probability":
                        raw = raw / scorer.temperature
                    losses.append(objective.loss(raw, target))
                return jnp.mean(jnp.stack(losses))

            candidate, report = self.strategy.optimize(initial, loss_fn)
            if set(candidate) != set(initial):
                raise ValueError(
                    "Optimized router tree does not match declared ownership."
                )
            checked_head = checked_parameters(candidate["head"], scorer.parameters)
            new_head = LinearRequestScorer(
                scorer.branch_ids,
                checked_head["weight"],
                checked_head["bias"],
                kind=scorer.kind,
                temperature=scorer.temperature,
            )
            model._validate_scorer(new_head)
            new_suffix = (
                checked_parameters(candidate["suffix"], model._parameters[router_id])
                if train_router_suffix
                else None
            )
            if new_suffix is not None:
                model._parameters[router_id] = new_suffix
            model._scorer = new_head
            model._training_data_ids.add(data_id)
            return report

    @staticmethod
    def _check_data(model, data_id):
        if getattr(model.selection, "calibration_data_id", None) == data_id:
            raise ValueError(
                "Fitting data must be disjoint from the current calibration dataset."
            )
