"""Acceptance tests for EVAL-005 efficiency measurements."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anchored_moe import (
    ActiveMLPMetrics,
    AdaptiveRouteDecision,
    EfficiencyBenchmarkConfig,
    ModuleStateFootprint,
    RouteDecision,
    Top2RouteDecision,
    benchmark_operation,
    measure_active_mlp_metrics,
    measure_module_state_footprint,
)


class _StatefulLinear(nn.Linear):
    """Small module with parameter, frozen parameter, and buffer state."""

    def __init__(self) -> None:
        super().__init__(3, 2)
        self.bias.requires_grad_(False)
        self.register_buffer("scale_bank", torch.ones(5, dtype=torch.float16))


def test_module_footprint_counts_unique_storage_and_trainability() -> None:
    module = _StatefulLinear()

    footprint = measure_module_state_footprint((module, module))

    assert footprint == ModuleStateFootprint(
        parameter_bytes=32,
        trainable_parameter_bytes=24,
        buffer_bytes=10,
    )
    assert footprint.total_bytes == 42
    assert footprint.to_dict() == {
        "buffer_bytes": 10,
        "parameter_bytes": 32,
        "total_bytes": 42,
        "trainable_parameter_bytes": 24,
    }


def test_module_footprint_treats_meta_state_as_non_resident() -> None:
    module = nn.Linear(3, 2, device="meta")

    footprint = measure_module_state_footprint(module)

    assert footprint.parameter_bytes == 0
    assert footprint.total_bytes == 0


def test_active_mlp_metrics_cover_top1_top2_and_adaptive_routes() -> None:
    top1 = RouteDecision(
        selected_indices=torch.tensor([0, 1, 2, 1]),
        selected_scores=torch.tensor([0.0, 2.0, 1.0, 3.0]),
        expert_ids=("code", "math"),
    )
    top2 = Top2RouteDecision(
        selected_indices=torch.tensor([[1, 2], [2, 1], [1, 0]]),
        selected_scores=torch.tensor([[2.0, 1.0], [3.0, 2.0], [0.5, 0.0]]),
        expert_ids=("code", "math"),
    )
    adaptive = AdaptiveRouteDecision(
        ranked_indices=torch.tensor([[1, 2], [2, 1], [1, 2]]),
        ranked_scores=torch.tensor([[2.0, 1.0], [3.0, 2.0], [1.5, 1.0]]),
        route_arity=torch.tensor([0, 1, 2]),
        expert_ids=("code", "math"),
    )

    top1_metrics = measure_active_mlp_metrics(top1)
    assert top1_metrics == ActiveMLPMetrics(
        route_type="top1",
        token_count=4,
        anchor_token_executions=1,
        specialist_token_executions=3,
        sparse_module_calls=3,
        per_expert_token_executions=(("code", 2), ("math", 1)),
    )
    assert top1_metrics.total_mlp_token_executions == 4
    assert top1_metrics.mean_active_mlps_per_token == 1.0

    top2_metrics = measure_active_mlp_metrics(top2)
    assert top2_metrics.anchor_token_executions == 1
    assert top2_metrics.per_expert_token_executions == (
        ("code", 2),
        ("math", 2),
    )
    assert top2_metrics.total_mlp_token_executions == 5
    assert top2_metrics.mean_active_mlps_per_token == pytest.approx(5.0 / 3.0)
    assert top2_metrics.sparse_module_calls == 3

    adaptive_metrics = measure_active_mlp_metrics(adaptive)
    assert adaptive_metrics.anchor_token_executions == 1
    assert adaptive_metrics.per_expert_token_executions == (
        ("code", 1),
        ("math", 2),
    )
    assert adaptive_metrics.total_mlp_token_executions == 4
    assert adaptive_metrics.mean_active_mlps_per_token == pytest.approx(4.0 / 3.0)
    assert adaptive_metrics.to_dict()["route_type"] == "adaptive"


def test_empty_routes_have_zero_work_without_fabricating_calls() -> None:
    routes = RouteDecision(
        selected_indices=torch.empty(0, dtype=torch.long),
        selected_scores=torch.empty(0),
        expert_ids=(),
    )

    metrics = measure_active_mlp_metrics(routes)

    assert metrics.token_count == 0
    assert metrics.total_mlp_token_executions == 0
    assert metrics.mean_active_mlps_per_token == 0.0
    assert metrics.sparse_module_calls == 0


def test_cpu_benchmark_honors_warmup_repetitions_and_work_units() -> None:
    module = nn.Linear(2, 2, bias=False)
    inputs = torch.ones(4, 2)
    calls: list[torch.Tensor] = []

    def operation() -> torch.Tensor:
        output = module(inputs)
        calls.append(output)
        return output

    report = benchmark_operation(
        operation,
        config=EfficiencyBenchmarkConfig(
            warmup_runs=2,
            measured_runs=4,
            work_items_per_run=inputs.shape[0],
            device="cpu",
        ),
        modules=module,
    )

    assert len(calls) == 6
    assert report.device == "cpu"
    assert report.measured_runs == 4
    assert len(report.latencies_seconds) == 4
    assert report.mean_latency_seconds > 0
    assert report.median_latency_seconds > 0
    assert report.p95_latency_seconds == max(report.latencies_seconds)
    assert report.throughput_items_per_second > 0
    assert report.peak_cuda_memory_bytes is None
    assert report.module_state == ModuleStateFootprint(
        parameter_bytes=16,
        trainable_parameter_bytes=16,
        buffer_bytes=0,
    )
    assert report.to_dict()["work_items_per_run"] == 4


def test_benchmark_measures_module_state_after_lazy_initialization() -> None:
    module = nn.LazyLinear(2, bias=False)
    inputs = torch.ones(1, 2)

    report = benchmark_operation(
        lambda: module(inputs),
        config=EfficiencyBenchmarkConfig(warmup_runs=0, measured_runs=1),
        modules=module,
    )

    assert report.module_state == ModuleStateFootprint(
        parameter_bytes=16,
        trainable_parameter_bytes=16,
        buffer_bytes=0,
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"warmup_runs": -1}, ValueError, "non-negative"),
        ({"measured_runs": 0}, ValueError, "positive"),
        ({"work_items_per_run": True}, TypeError, "integer"),
        ({"device": "not-a-device"}, ValueError, "Invalid benchmark device"),
    ],
)
def test_benchmark_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        EfficiencyBenchmarkConfig(**kwargs)


def test_efficiency_helpers_reject_wrong_public_types() -> None:
    with pytest.raises(TypeError, match="routes must be"):
        measure_active_mlp_metrics(torch.tensor([0]))
    with pytest.raises(TypeError, match="sequence of modules"):
        measure_module_state_footprint("model")
    with pytest.raises(TypeError, match="operation must be callable"):
        benchmark_operation(3)  # type: ignore[arg-type]
