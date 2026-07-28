"""Integration of EVAL-005 with top-1 and top-2 sparse dispatch."""

from __future__ import annotations

import torch
from torch import nn

from anchored_moe import (
    AnchoredTop1Dispatcher,
    AnchoredTop2Dispatcher,
    EfficiencyBenchmarkConfig,
    benchmark_operation,
    measure_active_mlp_metrics,
)


def test_dispatch_strategies_share_state_but_report_different_active_work() -> None:
    hidden_states = torch.ones(1, 3, 2)
    expert_scores = torch.tensor([[[-1.0, -2.0], [2.0, 1.0], [0.5, 1.5]]])
    expert_ids = ("code", "math")
    anchor = nn.Linear(2, 2)
    code = nn.Linear(2, 2)
    math = nn.Linear(2, 2)
    modules = (anchor, code, math)
    top1_dispatcher = AnchoredTop1Dispatcher()
    top2_dispatcher = AnchoredTop2Dispatcher()

    top1_operation = lambda: top1_dispatcher.dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=(code, math),
    )
    top2_operation = lambda: top2_dispatcher.dispatch(
        hidden_states,
        expert_scores=expert_scores,
        expert_ids=expert_ids,
        anchor_mlp=anchor,
        experts=(code, math),
    )
    config = EfficiencyBenchmarkConfig(
        warmup_runs=0,
        measured_runs=2,
        work_items_per_run=3,
    )
    with torch.no_grad():
        top1_routes = top1_operation().selected_routes
        top2_routes = top2_operation().selected_routes
        top1_benchmark = benchmark_operation(
            top1_operation,
            config=config,
            modules=modules,
        )
        top2_benchmark = benchmark_operation(
            top2_operation,
            config=config,
            modules=modules,
        )

    top1_work = measure_active_mlp_metrics(top1_routes)
    top2_work = measure_active_mlp_metrics(top2_routes)

    assert top1_work.total_mlp_token_executions == 3
    assert top2_work.total_mlp_token_executions == 5
    assert top1_benchmark.module_state == top2_benchmark.module_state
    assert top1_benchmark.module_state is not None
    assert top1_benchmark.module_state.total_bytes == 72
    assert top1_benchmark.throughput_items_per_second > 0
    assert top2_benchmark.throughput_items_per_second > 0
