"""Counterfactual branch utilities and synchronized request latency measurements."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class BranchUtilityReport:
    """Ordered offline losses and specialist gains over the original base.

    losses: [requests, response branches]. utilities: [requests, specialists].
    Lower loss means better performance. The caller supplies a task-aware loss
    and a dataset identity; this report does not itself calibrate a router.
    """

    data_id: str
    response_ids: tuple[str, ...]
    specialist_ids: tuple[str, ...]
    losses: tuple[tuple[float, ...], ...]
    utilities: tuple[tuple[float, ...], ...]
    composition_key: str


def evaluate_branch_utilities(
    model, requests, loss_fn, *, data_id: str
) -> BranchUtilityReport:
    """Evaluate every response suffix against each shared request prefix offline.

    loss_fn(request_index, logits) returns a finite scalar, lower is better.
    It may close over labels, but requests used later for router fitting must
    contain only pre-response text. Prefix states are not returned or persisted.
    """
    if not isinstance(data_id, str) or not data_id:
        raise ValueError("A nonempty evaluation dataset identity is required.")
    with model.operation():
        rows = []
        for index, ids in enumerate(requests):
            state = model.backend.prefill(ids)
            row = tuple(
                float(
                    loss_fn(
                        index,
                        model.backend.logits(state, model.effective_suffix(branch)),
                    )
                )
                for branch in model.response_ids
            )
            if not np.isfinite(row).all():
                raise ValueError("Counterfactual loss must be finite.")
            rows.append(row)
        if not rows:
            raise ValueError("Evaluation requests cannot be empty.")
        base_index = model.response_ids.index(model.config.fallback_branch)
        specialists = tuple(
            i for i in model.response_ids if i != model.config.fallback_branch
        )
        utilities = tuple(
            tuple(
                row[base_index] - row[model.response_ids.index(i)] for i in specialists
            )
            for row in rows
        )
        return BranchUtilityReport(
            data_id,
            model.response_ids,
            specialists,
            tuple(rows),
            utilities,
            model.composition_key,
        )


@dataclass(frozen=True)
class BranchLatencyReport:
    """Warm request-prefill timings including routing and the selected suffix.

    Samples include conversion of logits to host memory to synchronize device
    work. They exclude model loading and tokenization, and do not measure decode.
    """

    samples_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float


def benchmark_branch_request(
    model, input_ids, *, warmup: int = 1, repeats: int = 5
) -> BranchLatencyReport:
    """Measure the complete forward path; never infer cost from parameter count."""
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (warmup, repeats))
        or warmup < 0
        or repeats <= 0
    ):
        raise ValueError("warmup must be nonnegative and repeats positive integers.")
    for _ in range(warmup):
        np.asarray(model.forward(input_ids).logits)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        np.asarray(model.forward(input_ids).logits)
        samples.append(perf_counter() - start)
    return BranchLatencyReport(
        tuple(samples), float(np.median(samples)), float(np.percentile(samples, 95))
    )


def benchmark_branch_generation(
    model,
    input_ids,
    *,
    max_new_tokens: int,
    use_cache: bool | None = None,
    warmup: int = 1,
    repeats: int = 5,
) -> BranchLatencyReport:
    """Measure end-to-end greedy generation with a fixed token budget.

    Includes routing, prefill, cache setup and every decode step, excluding
    loading and tokenization. Host token selection synchronizes device work.
    Warmup excludes JIT compilation for this exact reserved capacity/shape.
    Compare identical prompts/budgets with use_cache=True and False; these
    timings do not measure the separate quantized native engine.
    """
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (warmup, repeats))
        or warmup < 0
        or repeats <= 0
    ):
        raise ValueError("warmup must be nonnegative and repeats positive integers.")
    for _ in range(warmup):
        model.generate_tokens(
            input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache
        )
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        model.generate_tokens(
            input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache
        )
        samples.append(perf_counter() - start)
    return BranchLatencyReport(
        tuple(samples), float(np.median(samples)), float(np.percentile(samples, 95))
    )
