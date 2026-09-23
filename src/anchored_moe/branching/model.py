"""Own response branches and route once using a reusable prefix snapshot."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import numpy as np

from anchored_moe.utils.hashing import canonical_json_bytes, sha256_json

from .backend import BranchBackend
from .configuration import BranchedModelConfig
from .parameters import checked_parameters, parameterization
from .routing import (
    BranchRoute,
    FixedBranchSelection,
    LinearRequestScorer,
    RequestScorer,
    RequestSelectionPolicy,
    ThresholdSelection,
)


def tensor_digest(tensors) -> str:
    """Hash ordered tensor names, shapes, dtypes and exact contiguous bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = np.asarray(value)
        digest.update(canonical_json_bytes([name, list(value.shape), str(value.dtype)]))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BranchedForwardResult:
    """Selected route and native [1, sequence, vocabulary] response logits."""

    route: BranchRoute
    logits: object


@dataclass(frozen=True)
class BranchGenerationResult:
    """Generated token IDs and one locked route, without implied JSON validity.

    prefix_evaluations counts full-sequence prefix passes. Cached generation uses
    one prefill and incremental token updates; uncached generation recomputes
    history. A caller may supply a grammar-aware logits processor.
    """

    route: BranchRoute
    token_ids: tuple[int, ...]
    prefix_evaluations: int
    cached: bool = False


class BranchedModel:
    """Own a frozen backend, independent suffix parameters, scorer and policy.

    Construction does not mutate the backend. All operations on one owner are
    serialized by a fail-fast lock. Parameter updates are validated before
    publication; calibrated policies become stale when their composition changes.
    Configuration and backend internals must not be mutated through private fields.
    """

    def __init__(
        self,
        backend: BranchBackend,
        config: BranchedModelConfig,
        scorer: RequestScorer,
        selection: RequestSelectionPolicy,
    ):
        """Build private branches and validate topology/scorer compatibility."""
        if backend.identity["shared_prefix_layers"] != config.shared_prefix_layers:
            raise ValueError("Backend and owner split points differ.")
        self._backend, self._config = backend, config
        self._lock = Lock()
        self._configs = {b.branch_id: b for b in config.response_branches}
        if config.router_branch:
            self._configs[config.router_branch.branch_id] = config.router_branch
        self._parameters = {}
        for branch_id, branch in self._configs.items():
            tensors = parameterization(branch).initialize(
                backend.suffix_parameters, branch
            )
            self._parameters[branch_id] = checked_parameters(tensors, tensors)
        self._scorer, self._selection = scorer, selection
        self._training_data_ids: set[str] = set()
        self._validate_scorer(scorer)
        self._validate_selection(selection)

    @property
    def backend(self) -> BranchBackend:
        """Return the immutable execution backend."""
        return self._backend

    @property
    def config(self) -> BranchedModelConfig:
        """Return immutable resolved branch declarations."""
        return self._config

    @property
    def scorer(self) -> RequestScorer:
        """Return the scoring strategy; replace it with set_scorer()."""
        return self._scorer

    @property
    def selection(self) -> RequestSelectionPolicy:
        """Return the selection strategy, including any calibration binding."""
        return self._selection

    @property
    def response_ids(self) -> tuple[str, ...]:
        """Return the immutable response candidate order."""
        return tuple(b.branch_id for b in self.config.response_branches)

    @property
    def composition_key(self) -> str:
        """Fingerprint backend, branch versions and scorer, excluding policy.

        Builtin affine scorers are fully hashed. Custom scorers must implement
        identity() returning JSON-compatible version metadata for calibration.
        """
        scorer = self.scorer
        if isinstance(scorer, LinearRequestScorer):
            score_identity = {
                "ids": scorer.branch_ids,
                "kind": scorer.kind,
                "temperature": scorer.temperature,
                "tensors": tensor_digest(scorer.parameters),
            }
        elif callable(getattr(scorer, "identity", None)):
            score_identity = scorer.identity()
        else:
            raise TypeError(
                "Custom scorers need identity() for calibration and packaging."
            )
        return sha256_json(
            {
                "backend": self.backend.identity,
                "config": self.config.to_dict(),
                "branches": {k: tensor_digest(v) for k, v in self._parameters.items()},
                "scorer": score_identity,
            }
        )

    @property
    def calibration_stale(self) -> bool:
        """Return whether a fitted threshold policy targets a different model."""
        return (
            isinstance(self.selection, ThresholdSelection)
            and self.selection.composition_key is not None
            and self.selection.composition_key != self.composition_key
        )

    @contextmanager
    def operation(self):
        """Reserve this owner for inference, training or publication; always release."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent or nested operations on one branch owner are forbidden."
            )
        try:
            yield
        finally:
            self._lock.release()

    def branch_parameters(self, branch_id: str) -> dict:
        """Return defensive copies of one branch's trainable tensors."""
        return {k: v.copy() for k, v in self._parameters[branch_id].items()}

    def effective_suffix(self, branch_id: str, parameters=None):
        """Compose a branch with the base, preserving native training gradients."""
        config = self._configs[branch_id]
        learned = self._parameters[branch_id] if parameters is None else parameters
        return parameterization(config).materialize(
            self.backend.suffix_parameters, learned, config
        )

    def replace_branch_parameters(self, branch_id: str, parameters) -> None:
        """Publish a validated update atomically; retain any stale calibration marker."""
        with self.operation():
            self._publish_branch(branch_id, parameters)

    def _publish_branch(self, branch_id, parameters):
        if self._configs[branch_id].parameterization == "frozen":
            raise ValueError("Cannot train or replace a frozen branch.")
        checked = checked_parameters(parameters, self._parameters[branch_id])
        self._parameters[branch_id] = checked

    def _validate_scorer(self, scorer):
        if isinstance(scorer, LinearRequestScorer):
            expected = (
                tuple(i for i in self.response_ids if i != self.config.fallback_branch)
                if scorer.kind == "utility"
                else self.response_ids
            )
            if (
                scorer.branch_ids != expected
                or scorer.feature_size != self.backend.feature_size
            ):
                raise ValueError(
                    "Scorer candidates/order/feature width do not match the composition."
                )

    def _validate_selection(self, selection, scorer=None):
        scorer = self.scorer if scorer is None else scorer
        if (
            isinstance(selection, FixedBranchSelection)
            and selection.branch_id not in self.response_ids
        ):
            raise ValueError("Fixed selection must name an installed response branch.")
        if isinstance(selection, ThresholdSelection):
            if set(dict(selection.thresholds)) != set(self.response_ids) - {
                self.config.fallback_branch
            }:
                raise ValueError(
                    "Thresholds must cover exactly the response specialists."
                )
            if (
                isinstance(scorer, LinearRequestScorer)
                and scorer.kind != selection.kind
            ):
                raise ValueError("Selection and scoring semantics differ.")

    def set_scorer(self, scorer: RequestScorer) -> None:
        """Publish a scorer after validation; previous calibration may become stale."""
        with self.operation():
            self._validate_scorer(scorer)
            self._validate_selection(self.selection, scorer)
            self._scorer = scorer

    def set_selection(self, selection: RequestSelectionPolicy) -> None:
        """Select a policy; reject known fitting/calibration dataset overlap."""
        with self.operation():
            self._validate_selection(selection)
            if (
                isinstance(selection, ThresholdSelection)
                and selection.calibration_data_id in self._training_data_ids
            ):
                raise ValueError(
                    "Calibration data must be disjoint from known training data."
                )
            self._selection = selection

    def _route(self, state):
        if isinstance(self.selection, FixedBranchSelection):
            route = self.selection.select(None, self.config.fallback_branch)
        else:
            if self.calibration_stale:
                raise RuntimeError(
                    "Router calibration is stale; recalibrate or explicitly choose a manual policy."
                )
            suffix = (
                self.effective_suffix(self.config.router_branch.branch_id)
                if self.config.router_branch
                else None
            )
            features = self.backend.features(state, suffix)
            scores = self.scorer.score(features)
            expected = set(self.response_ids) - (
                {self.config.fallback_branch} if scores.kind == "utility" else set()
            )
            if set(scores.branch_ids) != expected:
                raise ValueError(
                    "Router scores do not cover the declared response branches."
                )
            route = self.selection.select(scores, self.config.fallback_branch)
        if route.branch_id not in self.response_ids:
            raise ValueError("Router selected an unknown response branch.")
        return route

    def forward(self, input_ids) -> BranchedForwardResult:
        """Prefill once, score once, then execute exactly one response suffix."""
        with self.operation():
            state = self.backend.prefill(input_ids)
            route = self._route(state)
            return BranchedForwardResult(
                route,
                self.backend.logits(state, self.effective_suffix(route.branch_id)),
            )

    def route_request(self, input_ids) -> BranchRoute:
        """Select a response without executing it, for external native dispatch.

        Runs the shared prefix and optional router suffix only. The caller must
        render/tokenize the same request context used to fit the router. A stock
        native response engine computes its own full prefill; no JAX cache is
        transferred. Source calibration still needs native-outcome validation.
        """
        with self.operation():
            return self._route(self.backend.prefill(input_ids))

    def forward_branch(self, input_ids, branch_id: str):
        """Force one response branch for training-data scoring or equivalence."""
        with self.operation():
            if branch_id not in self.response_ids:
                raise ValueError("Expected a response branch ID.")
            return self.backend.logits(
                self.backend.prefill(input_ids), self.effective_suffix(branch_id)
            )

    def generate_tokens(
        self,
        input_ids,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        logits_processor=None,
        use_cache: bool | None = None,
    ) -> BranchGenerationResult:
        """Greedy decoding with a route locked for the complete response.

        input_ids is [1, sequence], unpadded. logits_processor, if supplied, takes
        (all_token_ids, next_logits) and returns a same-shape vector; -inf may
        mask tokens. It must retain at least one finite candidate. This API does
        not implement Needle's native grammar or confidence gate. use_cache=None
        selects caching when the backend supports it; True requires support and
        False retains the full-history reference path. Caches live for one call.
        """
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        if use_cache is not None and type(use_cache) is not bool:
            raise ValueError("use_cache must be None, True or False.")
        cached = (
            self.backend.supports_cached_decoding if use_cache is None else use_cache
        )
        if cached and not self.backend.supports_cached_decoding:
            raise NotImplementedError("This backend does not support cached decoding.")
        with self.operation():
            ids = np.array(input_ids, copy=True)
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("Generation requires [1, sequence] token IDs.")
            limit = self.backend.max_sequence_length
            if limit is not None and ids.shape[1] + max_new_tokens > limit:
                raise ValueError(
                    "Prompt plus reserved generation exceeds the backend context limit."
                )
            state = (
                self.backend.prefill_cached(ids, capacity=ids.shape[1] + max_new_tokens)
                if cached
                else self.backend.prefill(ids)
            )
            route = self._route(state)
            suffix = self.effective_suffix(route.branch_id)
            if cached:
                next_logits, decode_state = self.backend.start_decode(state, suffix)
                # The incremental snapshot retains KV and bounded histories; the
                # full prompt's residual lanes and engram arrays can be released.
                state = None
            output = []
            for step in range(max_new_tokens):
                if cached:
                    if step:
                        next_logits, decode_state = self.backend.decode_step(
                            decode_state, output[-1]
                        )
                elif step:
                    state = self.backend.prefill(ids)
                if not cached:
                    next_logits = self.backend.logits(state, suffix)
                logits = np.asarray(next_logits)[0, -1].copy()
                if logits_processor is not None:
                    processed = np.asarray(logits_processor(ids.copy(), logits.copy()))
                    if processed.shape != logits.shape:
                        raise ValueError("Logits processor changed vocabulary shape.")
                    logits = processed.astype(logits.dtype, copy=True)
                for forbidden in self.backend.forbidden_generation_ids:
                    logits[forbidden] = -np.inf
                if (
                    np.isnan(logits).any()
                    or np.isposinf(logits).any()
                    or not np.isfinite(logits).any()
                ):
                    raise ValueError("No valid finite next-token scores.")
                token = int(logits.argmax())
                output.append(token)
                if token == eos_token_id:
                    break
                ids = np.concatenate(
                    (ids, np.array([[token]], dtype=ids.dtype)), axis=1
                )
            return BranchGenerationResult(
                route, tuple(output), 1 if cached else len(output), cached
            )
