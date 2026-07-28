"""Compare anchored top-1 and top-2 quality and efficiency offline."""

from __future__ import annotations

import json

import torch
from torch import nn

from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Dispatcher,
    AnchoredTop1Selection,
    AnchoredTop2Dispatcher,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    EfficiencyBenchmarkConfig,
    benchmark_operation,
    compare_top1_top2_to_oracle,
    measure_active_mlp_metrics,
)


def main() -> None:
    """Benchmark deterministic local routes and print a JSON summary."""

    torch.manual_seed(13)
    expert_ids = ("code", "math")
    hidden_states = torch.randn(2, 4, 4)
    expert_scores = torch.tensor(
        [
            [[-1.0, -2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 1.0]],
            [[-0.5, -0.1], [0.8, 0.7], [1.2, 0.9], [0.2, -0.4]],
        ]
    )
    anchor = nn.Linear(4, 4).eval()
    code = nn.Linear(4, 4).eval()
    math = nn.Linear(4, 4).eval()
    modules = (anchor, code, math)
    for module in modules:
        module.requires_grad_(False)
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
    benchmark_config = EfficiencyBenchmarkConfig(
        warmup_runs=1,
        measured_runs=5,
        work_items_per_run=hidden_states[..., 0].numel(),
        device=str(hidden_states.device),
    )
    with torch.no_grad():
        top1_result = top1_operation()
        top2_result = top2_operation()
        top1_benchmark = benchmark_operation(
            top1_operation,
            config=benchmark_config,
            modules=modules,
        )
        top2_benchmark = benchmark_operation(
            top2_operation,
            config=benchmark_config,
            modules=modules,
        )

    top1_work = measure_active_mlp_metrics(top1_result.selected_routes)
    top2_work = measure_active_mlp_metrics(top2_result.selected_routes)

    # These deterministic counterfactual losses demonstrate EVAL-004 beside
    # EVAL-005. They are measured data in a production evaluation workflow.
    losses = CounterfactualRouteLosses(
        torch.tensor(
            [
                [0.5, 0.9, 1.0, 0.8],
                [1.0, 0.8, 0.9, 0.2],
                [1.0, 0.9, 0.8, 0.2],
                [1.0, 0.4, 0.6, 0.7],
                [0.4, 0.8, 0.7, 0.6],
                [1.0, 0.6, 0.8, 0.4],
                [1.0, 0.5, 0.7, 0.3],
                [0.8, 0.6, 0.9, 0.7],
            ]
        ).reshape(2, 4, 4),
        expert_ids=expert_ids,
        routes=(
            CounterfactualRoute.anchor(),
            CounterfactualRoute.single("code"),
            CounterfactualRoute.single("math"),
            CounterfactualRoute.pair("code", "math"),
        ),
        loss_name="cross_entropy",
    )
    top1_routes = AnchoredTop1Selection()(expert_scores, expert_ids=expert_ids)
    top2_routes = AnchorVetoTop2Selection()(expert_scores, expert_ids=expert_ids)
    quality = compare_top1_top2_to_oracle(losses, top1_routes, top2_routes)

    summary = {
        "quality": {
            "mean_top1_regret": quality.mean_top1_regret,
            "mean_top2_regret": quality.mean_top2_regret,
        },
        "top1": {
            "efficiency": top1_benchmark.to_dict(),
            "work": top1_work.to_dict(),
        },
        "top2": {
            "efficiency": top2_benchmark.to_dict(),
            "work": top2_work.to_dict(),
        },
    }
    assert top1_work.mean_active_mlps_per_token == 1.0
    assert top2_work.mean_active_mlps_per_token > 1.0
    assert top1_benchmark.module_state == top2_benchmark.module_state
    assert top1_benchmark.module_state is not None
    assert top1_benchmark.module_state.trainable_parameter_bytes == 0
    print("EVAL-005 validation passed: " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
