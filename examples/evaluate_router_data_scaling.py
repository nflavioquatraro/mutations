"""Build a plot-ready EVAL-007 report from completed training measurements."""

from __future__ import annotations

import json

from anchored_moe import (
    GlobalRouterRecalibrationReport,
    LayerGlobalRouterRecalibrationReport,
    LocalMiniMoEObjectiveReport,
    LocalMiniMoETrainingConfig,
    LocalMiniMoETrainingReport,
    LocalMiniMoETrainingTopology,
    RouterDataScalingMetrics,
    RouterDataScalingTrial,
    evaluate_router_data_scaling,
)


STRATA = (
    "specialization_interior",
    "boundary",
    "mixed",
)


def metrics(
    performance: tuple[float, float, float],
    regret: tuple[float, float, float],
) -> tuple[RouterDataScalingMetrics, ...]:
    """Create one complete held-out three-stratum measurement block."""

    return tuple(
        RouterDataScalingMetrics(
            stratum=stratum,
            evaluation_row_count=2,
            performance=metric,
            mean_router_regret=mean_regret,
        )
        for stratum, metric, mean_regret in zip(
            STRATA,
            performance,
            regret,
            strict=True,
        )
    )


def recalibration_report(
    sample_count: int,
) -> GlobalRouterRecalibrationReport:
    """Create compact RTR-004 provenance for one completed replay subset."""

    return GlobalRouterRecalibrationReport(
        registry_sha256="a" * 64,
        compatibility_sha256="b" * 64,
        feature_bank_metadata_sha256=f"{sample_count:064x}",
        utility_label_metadata_sha256=f"{sample_count + 100:064x}",
        loss_name="token_nll",
        learning_rate=0.05,
        optimization_steps=64,
        drift_weight=0.001,
        layer_reports=(
            LayerGlobalRouterRecalibrationReport(
                layer_index=2,
                expert_ids=("code:0", "code:1"),
                row_count=sample_count,
                anchor_target_count=0,
                initial_cross_entropy=1.0,
                final_cross_entropy=0.5,
                initial_total_loss=1.0,
                final_total_loss=0.5,
                initial_correct_count=0,
                final_correct_count=sample_count,
                changed_route_count=sample_count,
            ),
        ),
    )


def trial(
    sample_count: int,
    performance: tuple[float, float, float],
    regret: tuple[float, float, float],
) -> RouterDataScalingTrial:
    """Bind measurements to their exact nested RTR-004 replay subset."""

    return RouterDataScalingTrial(
        router_training_sample_count=sample_count,
        router_training_row_ids=tuple(range(sample_count)),
        metrics=metrics(performance, regret),
        recalibration_report=recalibration_report(sample_count),
    )


matched_training = LocalMiniMoETrainingReport(
    topology=LocalMiniMoETrainingTopology(
        layer_indices=(2,),
        expert_ids=("code:0", "code:1"),
        router_head_keys=((2, "code:0"), (2, "code:1")),
        trainable_parameter_names=("experts", "router_heads"),
        total_parameter_count=1_000,
        trainable_parameter_count=500,
        config=LocalMiniMoETrainingConfig(),
    ),
    global_step=100,
    training_loss=0.72,
    final_objectives=LocalMiniMoEObjectiveReport(
        specialization_loss=0.7,
        load_balancing_loss=0.0,
        total_loss=0.7,
    ),
)

report = evaluate_router_data_scaling(
    metric_name="mean_token_nll",
    higher_is_better=False,
    evaluation_row_ids_by_stratum={
        "specialization_interior": (100, 101),
        "boundary": (102, 103),
        "mixed": (104, 105),
    },
    append_only_metrics=metrics(
        (1.0, 1.2, 1.4),
        (0.3, 0.5, 0.7),
    ),
    router_only_trials=(
        trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6)),
        trial(4, (0.8, 1.0, 1.25), (0.1, 0.3, 0.5)),
    ),
    matched_partial_moe_metrics=metrics(
        (0.7, 0.8, 1.0),
        (0.05, 0.1, 0.2),
    ),
    matched_partial_moe_training=matched_training,
)

assert report.router_training_sample_counts == (2, 4)
assert len(report.gaps) == len(STRATA) * 2
assert report.gaps[0].performance_gain_over_append_only > 0
assert report.gaps[0].performance_gap_to_partial_moe > 0

print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
