"""Deterministic contract tests for EVAL-007 router-data scaling."""

from __future__ import annotations

from dataclasses import replace

import pytest

from anchored_moe import (
    GlobalRouterRecalibrationReport,
    LayerGlobalRouterRecalibrationReport,
    LocalMiniMoEObjectiveReport,
    LocalMiniMoETrainingConfig,
    LocalMiniMoETrainingReport,
    LocalMiniMoETrainingTopology,
    RouterDataScalingMetrics,
    RouterDataScalingReport,
    RouterDataScalingTrial,
    evaluate_router_data_scaling,
)


_STRATA = (
    "specialization_interior",
    "boundary",
    "mixed",
)


def _training_report(
    *,
    layer_indices: tuple[int, ...] = (2,),
    expert_ids: tuple[str, str] = ("code:0", "code:1"),
) -> LocalMiniMoETrainingReport:
    topology = LocalMiniMoETrainingTopology(
        layer_indices=layer_indices,
        expert_ids=expert_ids,
        router_head_keys=tuple(
            (layer_index, expert_id)
            for layer_index in layer_indices
            for expert_id in expert_ids
        ),
        trainable_parameter_names=("expert.weight", "router.weight"),
        total_parameter_count=12,
        trainable_parameter_count=6,
        config=LocalMiniMoETrainingConfig(),
    )
    return LocalMiniMoETrainingReport(
        topology=topology,
        global_step=3,
        training_loss=0.75,
        final_objectives=LocalMiniMoEObjectiveReport(
            specialization_loss=0.7,
            load_balancing_loss=0.0,
            total_loss=0.7,
        ),
    )


def _recalibration_report(
    sample_count: int,
    *,
    registry: str = "a" * 64,
    compatibility: str = "b" * 64,
    layer_index: int = 2,
    expert_ids: tuple[str, ...] = ("code:0", "code:1"),
) -> GlobalRouterRecalibrationReport:
    return GlobalRouterRecalibrationReport(
        registry_sha256=registry,
        compatibility_sha256=compatibility,
        feature_bank_metadata_sha256=f"{sample_count:064x}",
        utility_label_metadata_sha256=f"{sample_count + 100:064x}",
        loss_name="token_nll",
        learning_rate=0.05,
        optimization_steps=16,
        drift_weight=0.001,
        layer_reports=(
            LayerGlobalRouterRecalibrationReport(
                layer_index=layer_index,
                expert_ids=expert_ids,
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


def _metrics(
    performance: tuple[float, float, float],
    regret: tuple[float, float, float],
) -> tuple[RouterDataScalingMetrics, ...]:
    return tuple(
        RouterDataScalingMetrics(
            stratum=stratum,
            evaluation_row_count=2,
            performance=metric,
            mean_router_regret=mean_regret,
        )
        for stratum, metric, mean_regret in zip(
            _STRATA,
            performance,
            regret,
            strict=True,
        )
    )


def _trial(
    sample_count: int,
    performance: tuple[float, float, float],
    regret: tuple[float, float, float],
    **report_kwargs: object,
) -> RouterDataScalingTrial:
    return RouterDataScalingTrial(
        router_training_sample_count=sample_count,
        router_training_row_ids=tuple(range(sample_count)),
        metrics=_metrics(performance, regret),
        recalibration_report=_recalibration_report(
            sample_count,
            **report_kwargs,
        ),
    )


def _evaluate(
    *,
    trials: tuple[RouterDataScalingTrial, ...] | None = None,
    evaluation_rows: dict[str, tuple[int, ...]] | None = None,
    training_report: LocalMiniMoETrainingReport | None = None,
) -> RouterDataScalingReport:
    return evaluate_router_data_scaling(
        metric_name="mean_token_nll",
        higher_is_better=False,
        evaluation_row_ids_by_stratum=evaluation_rows
        or {
            "specialization_interior": (100, 101),
            "boundary": (102, 103),
            "mixed": (104, 105),
        },
        append_only_metrics=_metrics(
            (1.0, 1.2, 1.4),
            (0.3, 0.5, 0.7),
        ),
        router_only_trials=trials
        or (
            _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6)),
            _trial(4, (0.8, 1.0, 1.25), (0.1, 0.3, 0.5)),
        ),
        matched_partial_moe_metrics=_metrics(
            (0.7, 0.8, 1.0),
            (0.05, 0.1, 0.2),
        ),
        matched_partial_moe_training=training_report or _training_report(),
    )


def test_scaling_report_calculates_directional_gaps_and_round_trips() -> None:
    report = _evaluate()

    assert report.router_training_sample_counts == (2, 4)
    assert report.registry_sha256 == "a" * 64
    assert report.compatibility_sha256 == "b" * 64
    first = report.gaps[0]
    assert first.stratum == "specialization_interior"
    assert first.performance_gain_over_append_only == pytest.approx(0.1)
    assert first.performance_gap_to_partial_moe == pytest.approx(0.2)
    assert first.regret_reduction_over_append_only == pytest.approx(0.1)
    assert first.regret_gap_to_partial_moe == pytest.approx(0.15)
    assert RouterDataScalingReport.from_dict(report.to_dict()) == report


def test_scaling_report_canonicalizes_trial_and_metric_order() -> None:
    small = _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6))
    large = _trial(4, (0.8, 1.0, 1.25), (0.1, 0.3, 0.5))
    reversed_large = replace(large, metrics=large.metrics[::-1])

    report = _evaluate(trials=(reversed_large, small))

    assert report.router_training_sample_counts == (2, 4)
    assert tuple(metric.stratum for metric in report.router_only_trials[1].metrics) == (
        _STRATA
    )


def test_scaling_report_rejects_training_evaluation_overlap() -> None:
    with pytest.raises(ValueError, match="training and held-out"):
        _evaluate(
            evaluation_rows={
                "specialization_interior": (0, 101),
                "boundary": (102, 103),
                "mixed": (104, 105),
            }
        )


def test_scaling_report_rejects_non_nested_replay_subsets() -> None:
    first = _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6))
    second = replace(
        _trial(4, (0.8, 1.0, 1.25), (0.1, 0.3, 0.5)),
        router_training_row_ids=(2, 3, 4, 5),
    )

    with pytest.raises(ValueError, match="grow strictly"):
        _evaluate(trials=(first, second))


def test_scaling_report_rejects_registry_or_topology_drift() -> None:
    first = _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6))
    wrong_registry = _trial(
        4,
        (0.8, 1.0, 1.25),
        (0.1, 0.3, 0.5),
        registry="c" * 64,
    )
    with pytest.raises(ValueError, match="one registry"):
        _evaluate(trials=(first, wrong_registry))

    wrong_experts = _trial(
        4,
        (0.8, 1.0, 1.25),
        (0.1, 0.3, 0.5),
        expert_ids=("math:0", "math:1"),
    )
    with pytest.raises(ValueError, match="expert topologies"):
        _evaluate(trials=(first, wrong_experts))


def test_scaling_trial_rejects_mismatched_recalibration_row_provenance() -> None:
    with pytest.raises(ValueError, match="layer row_count"):
        RouterDataScalingTrial(
            router_training_sample_count=4,
            router_training_row_ids=(0, 1, 2, 3),
            metrics=_metrics((0.8, 1.0, 1.25), (0.1, 0.3, 0.5)),
            recalibration_report=_recalibration_report(2),
        )


def test_scaling_report_rejects_metric_row_count_or_derived_gap_tampering() -> None:
    trial = _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6))
    bad_metric = replace(trial.metrics[0], evaluation_row_count=3)
    trial = replace(trial, metrics=(bad_metric, *trial.metrics[1:]))
    with pytest.raises(ValueError, match="row count"):
        _evaluate(trials=(trial,))

    report = _evaluate()
    with pytest.raises(ValueError, match="gaps must exactly match"):
        replace(
            report,
            gaps=(
                replace(
                    report.gaps[0],
                    performance_gap_to_partial_moe=999.0,
                ),
                *report.gaps[1:],
            ),
        )


def test_scaling_report_does_not_require_monotonic_improvement() -> None:
    worse_large_trial = _trial(
        4,
        (0.95, 1.15, 1.35),
        (0.25, 0.45, 0.65),
    )

    report = _evaluate(
        trials=(
            _trial(2, (0.9, 1.1, 1.3), (0.2, 0.4, 0.6)),
            worse_large_trial,
        )
    )

    assert report.router_training_sample_counts == (2, 4)
    assert report.gaps[3].performance_gain_over_append_only == pytest.approx(
        0.05
    )
