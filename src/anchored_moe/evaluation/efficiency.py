"""Reproducible efficiency measurements for anchored routing strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median
from time import perf_counter_ns
from typing import Any

import torch
from torch import nn

from anchored_moe.policies.selection import (
    AdaptiveRouteDecision,
    RouteDecision,
    Top2RouteDecision,
)


@dataclass(frozen=True)
class EfficiencyBenchmarkConfig:
    """Configuration for one warmup-aware operation benchmark.

    Attributes:
        warmup_runs:
            Executions excluded from latency and throughput statistics.
        measured_runs:
            Executions included in the immutable result.
        work_items_per_run:
            Tokens, examples, or other explicitly named work units processed
            by each operation call. Throughput uses this count as its numerator.
        device:
            Device whose asynchronous work must be synchronized around timing.
            CPU, CUDA, and MPS devices are supported when available.
        record_cuda_peak_memory:
            Whether a CUDA benchmark resets allocator peak statistics after
            warmup and records the maximum allocated bytes during measured
            runs. Non-CUDA reports always use ``None`` for this metric.
    """

    warmup_runs: int = 1
    measured_runs: int = 5
    work_items_per_run: int = 1
    device: str = "cpu"
    record_cuda_peak_memory: bool = True

    def __post_init__(self) -> None:
        """Validate counts and normalize the device string."""

        for field_name in ("warmup_runs", "measured_runs", "work_items_per_run"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative.")
        if self.measured_runs <= 0:
            raise ValueError("measured_runs must be positive.")
        if self.work_items_per_run <= 0:
            raise ValueError("work_items_per_run must be positive.")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")
        try:
            normalized_device = str(torch.device(self.device))
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark device: {self.device!r}.") from error
        object.__setattr__(self, "device", normalized_device)
        if not isinstance(self.record_cuda_peak_memory, bool):
            raise TypeError("record_cuda_peak_memory must be a boolean.")


@dataclass(frozen=True)
class ModuleStateFootprint:
    """Resident bytes owned by unique parameter and buffer storages.

    Shared storages are counted once even when several modules or tensor views
    reference them. A storage referenced by both a parameter and a buffer is
    classified as parameter state. Optimizer state, activations, allocator
    caching, Python objects, and package files are deliberately excluded.
    """

    parameter_bytes: int
    trainable_parameter_bytes: int
    buffer_bytes: int

    def __post_init__(self) -> None:
        """Validate non-negative integer byte counts."""

        for field_name in (
            "parameter_bytes",
            "trainable_parameter_bytes",
            "buffer_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.trainable_parameter_bytes > self.parameter_bytes:
            raise ValueError(
                "trainable_parameter_bytes cannot exceed parameter_bytes."
            )

    @property
    def total_bytes(self) -> int:
        """Return parameter plus buffer storage bytes."""

        return self.parameter_bytes + self.buffer_bytes

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible representation."""

        return {
            "buffer_bytes": self.buffer_bytes,
            "parameter_bytes": self.parameter_bytes,
            "total_bytes": self.total_bytes,
            "trainable_parameter_bytes": self.trainable_parameter_bytes,
        }


@dataclass(frozen=True)
class ActiveMLPMetrics:
    """Exact per-token MLP work implied by one route decision.

    Attributes:
        route_type:
            ``"top1"``, ``"top2"``, or ``"adaptive"``.
        token_count:
            Number of routed tokens.
        anchor_token_executions:
            Tokens for which the immutable anchor MLP executes.
        specialist_token_executions:
            Sum of active specialist executions across tokens.
        sparse_module_calls:
            Number of distinct non-empty anchor or specialist groups. This is
            the ideal sparse dispatcher call count for one layer and batch.
        per_expert_token_executions:
            Stable ``(expert_id, count)`` pairs in registry order.
    """

    route_type: str
    token_count: int
    anchor_token_executions: int
    specialist_token_executions: int
    sparse_module_calls: int
    per_expert_token_executions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Validate counts and the stable per-expert breakdown."""

        if self.route_type not in {"top1", "top2", "adaptive"}:
            raise ValueError("route_type must be top1, top2, or adaptive.")
        for field_name in (
            "token_count",
            "anchor_token_executions",
            "specialist_token_executions",
            "sparse_module_calls",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        normalized_counts = tuple(self.per_expert_token_executions)
        object.__setattr__(self, "per_expert_token_executions", normalized_counts)
        expert_ids: list[str] = []
        for item in normalized_counts:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
            ):
                raise TypeError(
                    "per_expert_token_executions must contain "
                    "(non-empty string, integer) pairs."
                )
            if isinstance(item[1], bool) or not isinstance(item[1], int):
                raise TypeError("Per-expert execution counts must be integers.")
            if item[1] < 0:
                raise ValueError(
                    "Per-expert execution counts must be non-negative."
                )
            expert_ids.append(item[0])
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("Per-expert execution identifiers must be unique.")
        if sum(count for _, count in normalized_counts) != (
            self.specialist_token_executions
        ):
            raise ValueError(
                "Per-expert counts must sum to specialist_token_executions."
            )
        expected_calls = int(self.anchor_token_executions > 0) + sum(
            int(count > 0) for _, count in normalized_counts
        )
        if self.sparse_module_calls != expected_calls:
            raise ValueError(
                "sparse_module_calls must count non-empty anchor and expert groups."
            )

    @property
    def total_mlp_token_executions(self) -> int:
        """Return anchor plus specialist token executions."""

        return self.anchor_token_executions + self.specialist_token_executions

    @property
    def mean_active_mlps_per_token(self) -> float:
        """Return total MLP token executions divided by routed tokens."""

        if self.token_count == 0:
            return 0.0
        return self.total_mlp_token_executions / self.token_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "anchor_token_executions": self.anchor_token_executions,
            "mean_active_mlps_per_token": self.mean_active_mlps_per_token,
            "per_expert_token_executions": dict(
                self.per_expert_token_executions
            ),
            "route_type": self.route_type,
            "sparse_module_calls": self.sparse_module_calls,
            "specialist_token_executions": self.specialist_token_executions,
            "token_count": self.token_count,
            "total_mlp_token_executions": self.total_mlp_token_executions,
        }


@dataclass(frozen=True)
class EfficiencyBenchmarkReport:
    """Immutable latency, throughput, and memory report.

    Latencies contain one synchronized wall-clock duration per measured run.
    Throughput is total declared work divided by total measured duration.
    CUDA peak memory is allocator state for the configured device after the
    warmup boundary; CPU and MPS reports expose ``None`` rather than an
    inaccurate Python-heap proxy.
    """

    device: str
    warmup_runs: int
    work_items_per_run: int
    latencies_seconds: tuple[float, ...]
    module_state: ModuleStateFootprint | None
    peak_cuda_memory_bytes: int | None

    def __post_init__(self) -> None:
        """Validate report values and detach mutable latency sequences."""

        try:
            normalized_device = str(torch.device(self.device))
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid report device: {self.device!r}.") from error
        object.__setattr__(self, "device", normalized_device)
        for field_name in ("warmup_runs", "work_items_per_run"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative.")
        if self.work_items_per_run <= 0:
            raise ValueError("work_items_per_run must be positive.")
        latencies = tuple(self.latencies_seconds)
        object.__setattr__(self, "latencies_seconds", latencies)
        if not latencies:
            raise ValueError("latencies_seconds must contain at least one run.")
        if any(not isfinite(value) or value <= 0 for value in latencies):
            raise ValueError("Every measured latency must be finite and positive.")
        if self.module_state is not None and not isinstance(
            self.module_state, ModuleStateFootprint
        ):
            raise TypeError("module_state must be ModuleStateFootprint or None.")
        if self.peak_cuda_memory_bytes is not None:
            if isinstance(self.peak_cuda_memory_bytes, bool) or not isinstance(
                self.peak_cuda_memory_bytes, int
            ):
                raise TypeError("peak_cuda_memory_bytes must be an integer or None.")
            if self.peak_cuda_memory_bytes < 0:
                raise ValueError("peak_cuda_memory_bytes must be non-negative.")
            if torch.device(self.device).type != "cuda":
                raise ValueError(
                    "peak_cuda_memory_bytes is defined only for CUDA reports."
                )

    @property
    def measured_runs(self) -> int:
        """Return the number of recorded executions."""

        return len(self.latencies_seconds)

    @property
    def mean_latency_seconds(self) -> float:
        """Return arithmetic mean synchronized latency."""

        return sum(self.latencies_seconds) / self.measured_runs

    @property
    def median_latency_seconds(self) -> float:
        """Return median synchronized latency."""

        return float(median(self.latencies_seconds))

    @property
    def p95_latency_seconds(self) -> float:
        """Return deterministic nearest-rank 95th-percentile latency."""

        ordered = sorted(self.latencies_seconds)
        nearest_rank = max(1, (95 * len(ordered) + 99) // 100)
        return ordered[nearest_rank - 1]

    @property
    def throughput_items_per_second(self) -> float:
        """Return declared work items divided by total measured time."""

        total_items = self.work_items_per_run * self.measured_runs
        return total_items / sum(self.latencies_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate metrics as a JSON-compatible mapping."""

        return {
            "device": self.device,
            "mean_latency_seconds": self.mean_latency_seconds,
            "measured_runs": self.measured_runs,
            "median_latency_seconds": self.median_latency_seconds,
            "module_state": (
                None if self.module_state is None else self.module_state.to_dict()
            ),
            "p95_latency_seconds": self.p95_latency_seconds,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "throughput_items_per_second": self.throughput_items_per_second,
            "warmup_runs": self.warmup_runs,
            "work_items_per_run": self.work_items_per_run,
        }


def measure_module_state_footprint(
    modules: nn.Module | Sequence[nn.Module],
) -> ModuleStateFootprint:
    """Count unique resident parameter and buffer storage bytes.

    Args:
        modules:
            One module or a finite sequence of modules. Passing both a parent
            and one of its children is safe because shared storage is counted
            once.

    Returns:
        Exact storage-byte counts for parameters, trainable parameters, and
        buffers currently referenced by the supplied modules.
    """

    normalized_modules = _normalize_modules(modules)
    parameter_storages: dict[tuple[Any, ...], tuple[int, bool]] = {}
    for module in normalized_modules:
        for parameter in module.parameters():
            key, storage_bytes = _storage_identity(parameter)
            previous = parameter_storages.get(key)
            parameter_storages[key] = (
                storage_bytes,
                parameter.requires_grad
                if previous is None
                else previous[1] or parameter.requires_grad,
            )

    buffer_storages: dict[tuple[Any, ...], int] = {}
    for module in normalized_modules:
        for buffer in module.buffers():
            key, storage_bytes = _storage_identity(buffer)
            if key not in parameter_storages:
                buffer_storages.setdefault(key, storage_bytes)

    return ModuleStateFootprint(
        parameter_bytes=sum(size for size, _ in parameter_storages.values()),
        trainable_parameter_bytes=sum(
            size for size, trainable in parameter_storages.values() if trainable
        ),
        buffer_bytes=sum(buffer_storages.values()),
    )


def measure_active_mlp_metrics(
    routes: RouteDecision | Top2RouteDecision | AdaptiveRouteDecision,
) -> ActiveMLPMetrics:
    """Count exact MLP token executions implied by route metadata.

    Top-1 always executes one MLP per token. Pair-shaped top-2 executes two
    specialists only when neither selected slot is the anchor; an anchor veto
    executes the anchor alone. Adaptive routing executes the anchor for arity
    zero and exactly ``route_arity`` specialists otherwise.
    """

    if isinstance(routes, RouteDecision):
        route_type = "top1"
        token_count = routes.selected_indices.numel()
        anchor_count = int(routes.anchor_mask.sum().item())
        per_expert = tuple(
            (
                expert_id,
                int(routes.mask_for_expert(expert_id).sum().item()),
            )
            for expert_id in routes.expert_ids
        )
    elif isinstance(routes, Top2RouteDecision):
        route_type = "top2"
        token_count = routes.anchor_mask.numel()
        pair_mask = routes.expert_pair_mask
        anchor_count = int(routes.anchor_mask.sum().item())
        per_expert = tuple(
            (
                expert_id,
                int(
                    (
                        (routes.selected_indices == candidate_index)
                        & pair_mask.unsqueeze(-1)
                    ).sum().item()
                ),
            )
            for candidate_index, expert_id in enumerate(routes.expert_ids, start=1)
        )
    elif isinstance(routes, AdaptiveRouteDecision):
        route_type = "adaptive"
        token_count = routes.route_arity.numel()
        anchor_count = int(routes.anchor_mask.sum().item())
        per_expert = tuple(
            (
                expert_id,
                int(routes.mask_for_expert(expert_id).sum().item()),
            )
            for expert_id in routes.expert_ids
        )
    else:
        raise TypeError(
            "routes must be RouteDecision, Top2RouteDecision, or "
            "AdaptiveRouteDecision."
        )

    specialist_count = sum(count for _, count in per_expert)
    sparse_calls = int(anchor_count > 0) + sum(
        int(count > 0) for _, count in per_expert
    )
    return ActiveMLPMetrics(
        route_type=route_type,
        token_count=token_count,
        anchor_token_executions=anchor_count,
        specialist_token_executions=specialist_count,
        sparse_module_calls=sparse_calls,
        per_expert_token_executions=per_expert,
    )


def benchmark_operation(
    operation: Callable[[], Any],
    *,
    config: EfficiencyBenchmarkConfig | None = None,
    modules: nn.Module | Sequence[nn.Module] | None = None,
) -> EfficiencyBenchmarkReport:
    """Benchmark a zero-argument operation with explicit synchronization.

    Args:
        operation:
            Callable that performs one complete measured unit. Inputs should
            already be materialized so setup cost is included only when the
            caller intentionally places it inside the callable.
        config:
            Warmup, repetition, work-unit, device, and CUDA-memory settings.
        modules:
            Optional module state whose unique resident storage bytes should be
            reported alongside runtime measurements.

    Returns:
        An immutable synchronized efficiency report.

    Raises:
        TypeError:
            If the operation, configuration, or modules are invalid.
        RuntimeError:
            If the requested accelerator is unavailable.

    Notes:
        The function does not change module training mode, gradient mode,
        autocast, random seeds, or thread counts. CUDA peak-stat reset affects
        the configured process-global device allocator statistics.
    """

    if not callable(operation):
        raise TypeError("operation must be callable.")
    resolved_config = config or EfficiencyBenchmarkConfig()
    if not isinstance(resolved_config, EfficiencyBenchmarkConfig):
        raise TypeError("config must be EfficiencyBenchmarkConfig or None.")
    device = torch.device(resolved_config.device)
    _validate_device_available(device)

    for _ in range(resolved_config.warmup_runs):
        operation()
    _synchronize_device(device)

    if device.type == "cuda" and resolved_config.record_cuda_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    for _ in range(resolved_config.measured_runs):
        _synchronize_device(device)
        start_ns = perf_counter_ns()
        operation()
        _synchronize_device(device)
        elapsed_ns = perf_counter_ns() - start_ns
        # perf_counter_ns is monotonic. Preserve a positive duration even on a
        # hypothetical clock whose resolution is coarser than this operation.
        latencies.append(max(elapsed_ns, 1) / 1_000_000_000)

    peak_cuda_memory = None
    if device.type == "cuda" and resolved_config.record_cuda_peak_memory:
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(device))
    # Count state after execution so lazily initialized parameters or buffers
    # are represented without placing state inspection inside timed regions.
    module_state = (
        None if modules is None else measure_module_state_footprint(modules)
    )

    return EfficiencyBenchmarkReport(
        device=str(device),
        warmup_runs=resolved_config.warmup_runs,
        work_items_per_run=resolved_config.work_items_per_run,
        latencies_seconds=tuple(latencies),
        module_state=module_state,
        peak_cuda_memory_bytes=peak_cuda_memory,
    )


def _normalize_modules(
    modules: nn.Module | Sequence[nn.Module],
) -> tuple[nn.Module, ...]:
    """Normalize and validate a module-or-sequence input."""

    if isinstance(modules, nn.Module):
        return (modules,)
    if not isinstance(modules, Sequence) or isinstance(
        modules, (str, bytes, bytearray)
    ):
        raise TypeError("modules must be a module or a sequence of modules.")
    normalized = tuple(modules)
    if not all(isinstance(module, nn.Module) for module in normalized):
        raise TypeError("Every modules item must be torch.nn.Module.")
    return normalized


def _storage_identity(tensor: torch.Tensor) -> tuple[tuple[Any, ...], int]:
    """Return a device-aware storage identity and its resident byte size."""

    storage = tensor.untyped_storage()
    storage_bytes = int(storage.nbytes())
    if tensor.device.type == "meta":
        # Meta tensors describe logical shapes but own no resident allocation.
        # Object identity still prevents unrelated storages collapsing.
        return (("meta", id(storage)), 0)
    device = tensor.device
    return (
        (
            device.type,
            device.index,
            int(storage.data_ptr()),
            storage_bytes,
        ),
        storage_bytes,
    )


def _validate_device_available(device: torch.device) -> None:
    """Reject unavailable asynchronous benchmark devices explicitly."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA benchmark device is unavailable: {device}.")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"CUDA benchmark device is unavailable: {device}.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError(f"MPS benchmark device is unavailable: {device}.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "Benchmark synchronization supports only CPU, CUDA, and MPS devices."
        )


def _synchronize_device(device: torch.device) -> None:
    """Wait for queued work on one supported asynchronous device."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
