"""Router-data scaling evaluation against append-only and partial-MoE baselines.

EVAL-007 records a deterministic experiment grid after training has completed.
It compares one strict append-only router, router-only global recalibration at
increasing replay-subset sizes, and one matched TRN-003 local partial MoE. The
evaluator owns validation, provenance, and metric aggregation; it deliberately
does not collect data, fit routers, train experts, or render plots.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from anchored_moe.calibration.global_recalibration import (
    GlobalRouterRecalibrationReport,
)
from anchored_moe.utils.hashing import sha256_json

if TYPE_CHECKING:
    from anchored_moe.training.mini_moe import LocalMiniMoETrainingReport


ROUTER_DATA_SCALING_FORMAT_VERSION = 1
ROUTER_DATA_STRATA = (
    "specialization_interior",
    "boundary",
    "mixed",
)


class RouterDataScalingEvaluationError(ValueError):
    """Raised when an EVAL-007 experiment grid or provenance is invalid."""


@dataclass(frozen=True)
class RouterDataScalingMetrics:
    """Held-out performance and mean router regret for one data stratum.

    Args:
        stratum:
            One of ``specialization_interior``, ``boundary``, or ``mixed``.
        evaluation_row_count:
            Number of held-out rows represented by both scalar metrics.
        performance:
            Finite aggregate value for the report's declared performance
            metric. Its direction is declared once by the enclosing report.
        mean_router_regret:
            Finite non-negative mean excess loss relative to the
            counterfactual best route on the same held-out rows.
    """

    stratum: str
    evaluation_row_count: int
    performance: float
    mean_router_regret: float

    def __post_init__(self) -> None:
        """Normalize JSON scalars and validate stratum metric semantics."""

        if self.stratum not in ROUTER_DATA_STRATA:
            raise ValueError(
                "stratum must be one of "
                f"{ROUTER_DATA_STRATA}, got {self.stratum!r}."
            )
        count = _positive_integer(
            self.evaluation_row_count,
            field_name="evaluation_row_count",
        )
        performance = _finite_number(self.performance, field_name="performance")
        regret = _non_negative_number(
            self.mean_router_regret,
            field_name="mean_router_regret",
        )
        object.__setattr__(self, "evaluation_row_count", count)
        object.__setattr__(self, "performance", performance)
        object.__setattr__(self, "mean_router_regret", regret)

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-compatible stratum measurement."""

        return {
            "evaluation_row_count": self.evaluation_row_count,
            "mean_router_regret": self.mean_router_regret,
            "performance": self.performance,
            "stratum": self.stratum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouterDataScalingMetrics:
        """Strictly reconstruct one stratum measurement."""

        _require_exact_fields(
            value,
            {
                "evaluation_row_count",
                "mean_router_regret",
                "performance",
                "stratum",
            },
            field_name="router-data scaling metrics",
        )
        return cls(
            stratum=value["stratum"],
            evaluation_row_count=value["evaluation_row_count"],
            performance=value["performance"],
            mean_router_regret=value["mean_router_regret"],
        )


@dataclass(frozen=True)
class MatchedPartialMoEProvenance:
    """Stable audit summary extracted from a completed TRN-003 report."""

    strategy: str
    layer_indices: tuple[int, ...]
    expert_ids: tuple[str, ...]
    global_step: int
    training_loss: float
    training_config_sha256: str

    def __post_init__(self) -> None:
        """Validate the matched two-expert topology and training outcome."""

        if self.strategy != "local_mini_moe_top2":
            raise ValueError(
                "matched partial MoE strategy must be 'local_mini_moe_top2'."
            )
        if (
            not isinstance(self.layer_indices, tuple)
            or not self.layer_indices
            or self.layer_indices != tuple(sorted(set(self.layer_indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in self.layer_indices
            )
        ):
            raise ValueError(
                "layer_indices must be unique ascending non-negative integers."
            )
        if (
            not isinstance(self.expert_ids, tuple)
            or len(self.expert_ids) != 2
            or len(set(self.expert_ids)) != 2
            or any(
                not isinstance(expert_id, str)
                or not expert_id
                or expert_id != expert_id.strip()
                for expert_id in self.expert_ids
            )
        ):
            raise ValueError(
                "expert_ids must contain two unique non-empty identifiers."
            )
        object.__setattr__(
            self,
            "global_step",
            _positive_integer(self.global_step, field_name="global_step"),
        )
        object.__setattr__(
            self,
            "training_loss",
            _finite_number(self.training_loss, field_name="training_loss"),
        )
        _sha256(self.training_config_sha256, field_name="training_config_sha256")

    @classmethod
    def from_training_report(
        cls,
        report: LocalMiniMoETrainingReport,
    ) -> MatchedPartialMoEProvenance:
        """Extract portable provenance without retaining model or expert state."""

        # Import lazily because the root package exposes evaluation before
        # modeling/training; eager cross-subpackage imports would create a
        # modeling -> inference -> evaluation -> training cycle.
        from anchored_moe.training.mini_moe import LocalMiniMoETrainingReport

        if not isinstance(report, LocalMiniMoETrainingReport):
            raise TypeError(
                "matched_partial_moe_training must be a "
                "LocalMiniMoETrainingReport."
            )
        return cls(
            strategy=report.topology.config.strategy,
            layer_indices=report.topology.layer_indices,
            expert_ids=report.topology.expert_ids,
            global_step=report.global_step,
            training_loss=report.training_loss,
            training_config_sha256=sha256_json(
                report.topology.config.to_dict()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return portable matched-baseline training provenance."""

        return {
            "expert_ids": list(self.expert_ids),
            "global_step": self.global_step,
            "layer_indices": list(self.layer_indices),
            "strategy": self.strategy,
            "training_config_sha256": self.training_config_sha256,
            "training_loss": self.training_loss,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> MatchedPartialMoEProvenance:
        """Strictly reconstruct matched-baseline provenance."""

        _require_exact_fields(
            value,
            {
                "expert_ids",
                "global_step",
                "layer_indices",
                "strategy",
                "training_config_sha256",
                "training_loss",
            },
            field_name="matched partial-MoE provenance",
        )
        return cls(
            strategy=value["strategy"],
            layer_indices=_integer_tuple(
                value["layer_indices"],
                field_name="layer_indices",
            ),
            expert_ids=_string_tuple(
                value["expert_ids"],
                field_name="expert_ids",
            ),
            global_step=value["global_step"],
            training_loss=value["training_loss"],
            training_config_sha256=value["training_config_sha256"],
        )


@dataclass(frozen=True)
class RouterDataScalingTrial:
    """One router-only recalibration at a declared nested replay-subset size."""

    router_training_sample_count: int
    router_training_row_ids: tuple[int, ...]
    metrics: tuple[RouterDataScalingMetrics, ...]
    recalibration_report: GlobalRouterRecalibrationReport

    strategy_name: ClassVar[str] = "router_only_global"

    def __post_init__(self) -> None:
        """Canonicalize metrics and validate sample/report provenance."""

        sample_count = _positive_integer(
            self.router_training_sample_count,
            field_name="router_training_sample_count",
        )
        row_ids = _row_id_tuple(
            self.router_training_row_ids,
            field_name="router_training_row_ids",
        )
        if len(row_ids) != sample_count:
            raise RouterDataScalingEvaluationError(
                "router_training_sample_count must equal the number of unique "
                "router_training_row_ids."
            )
        metrics = _canonical_metrics(self.metrics, field_name="trial metrics")
        if not isinstance(
            self.recalibration_report,
            GlobalRouterRecalibrationReport,
        ):
            raise TypeError(
                "recalibration_report must be a "
                "GlobalRouterRecalibrationReport."
            )
        if self.recalibration_report.strategy_name != self.strategy_name:
            raise ValueError(
                "recalibration_report must use router_only_global strategy."
            )
        if any(
            layer.row_count != sample_count
            for layer in self.recalibration_report.layer_reports
        ):
            raise RouterDataScalingEvaluationError(
                "every RTR-004 layer row_count must equal the declared router "
                "training sample count."
            )
        object.__setattr__(
            self,
            "router_training_sample_count",
            sample_count,
        )
        object.__setattr__(self, "router_training_row_ids", row_ids)
        object.__setattr__(self, "metrics", metrics)

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-compatible scaling trial with RTR-004 provenance."""

        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "recalibration_report": self.recalibration_report.to_dict(),
            "router_training_row_ids": list(self.router_training_row_ids),
            "router_training_sample_count": self.router_training_sample_count,
            "strategy": self.strategy_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouterDataScalingTrial:
        """Strictly reconstruct one router-only scaling trial."""

        _require_exact_fields(
            value,
            {
                "metrics",
                "recalibration_report",
                "router_training_row_ids",
                "router_training_sample_count",
                "strategy",
            },
            field_name="router-data scaling trial",
        )
        if value["strategy"] != cls.strategy_name:
            raise ValueError("Unsupported router-data scaling trial strategy.")
        raw_metrics = value["metrics"]
        if not isinstance(raw_metrics, list):
            raise TypeError("trial metrics must be a list.")
        return cls(
            router_training_sample_count=value[
                "router_training_sample_count"
            ],
            router_training_row_ids=_integer_tuple(
                value["router_training_row_ids"],
                field_name="router_training_row_ids",
            ),
            metrics=tuple(
                RouterDataScalingMetrics.from_dict(metric)
                for metric in raw_metrics
            ),
            recalibration_report=GlobalRouterRecalibrationReport.from_dict(
                value["recalibration_report"]
            ),
        )


@dataclass(frozen=True)
class RouterDataScalingGap:
    """Direction-normalized EVAL-007 comparison for one trial and stratum.

    Positive ``performance_gain_over_append_only`` and
    ``regret_reduction_over_append_only`` favor router-only recalibration.
    Positive ``performance_gap_to_partial_moe`` and
    ``regret_gap_to_partial_moe`` mean the matched partial MoE remains better.
    """

    router_training_sample_count: int
    stratum: str
    performance_gain_over_append_only: float
    performance_gap_to_partial_moe: float
    regret_reduction_over_append_only: float
    regret_gap_to_partial_moe: float

    def __post_init__(self) -> None:
        """Validate finite comparison scalars and their grid identity."""

        object.__setattr__(
            self,
            "router_training_sample_count",
            _positive_integer(
                self.router_training_sample_count,
                field_name="router_training_sample_count",
            ),
        )
        if self.stratum not in ROUTER_DATA_STRATA:
            raise ValueError(f"Unsupported scaling stratum: {self.stratum!r}.")
        for field_name in (
            "performance_gain_over_append_only",
            "performance_gap_to_partial_moe",
            "regret_reduction_over_append_only",
            "regret_gap_to_partial_moe",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-compatible pointwise comparison."""

        return {
            "performance_gain_over_append_only": (
                self.performance_gain_over_append_only
            ),
            "performance_gap_to_partial_moe": (
                self.performance_gap_to_partial_moe
            ),
            "regret_gap_to_partial_moe": self.regret_gap_to_partial_moe,
            "regret_reduction_over_append_only": (
                self.regret_reduction_over_append_only
            ),
            "router_training_sample_count": self.router_training_sample_count,
            "stratum": self.stratum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouterDataScalingGap:
        """Strictly reconstruct one pointwise comparison."""

        expected = {
            "performance_gain_over_append_only",
            "performance_gap_to_partial_moe",
            "regret_gap_to_partial_moe",
            "regret_reduction_over_append_only",
            "router_training_sample_count",
            "stratum",
        }
        _require_exact_fields(
            value,
            expected,
            field_name="router-data scaling gap",
        )
        return cls(**{field_name: value[field_name] for field_name in expected})


@dataclass(frozen=True)
class RouterDataScalingReport:
    """Complete immutable EVAL-007 experiment grid and derived scaling gaps."""

    metric_name: str
    higher_is_better: bool
    evaluation_row_ids_by_stratum: tuple[
        tuple[str, tuple[int, ...]],
        ...,
    ]
    append_only_metrics: tuple[RouterDataScalingMetrics, ...]
    router_only_trials: tuple[RouterDataScalingTrial, ...]
    matched_partial_moe_metrics: tuple[RouterDataScalingMetrics, ...]
    matched_partial_moe_provenance: MatchedPartialMoEProvenance
    gaps: tuple[RouterDataScalingGap, ...]

    format_version: ClassVar[int] = ROUTER_DATA_SCALING_FORMAT_VERSION
    feature_id: ClassVar[str] = "EVAL-007"
    append_only_strategy: ClassVar[str] = "strict_append_only"
    matched_strategy: ClassVar[str] = "matched_local_partial_moe"

    def __post_init__(self) -> None:
        """Validate exact grid coverage, splits, topology, and derived gaps."""

        _identifier(self.metric_name, field_name="metric_name")
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a bool.")
        evaluation_rows = _evaluation_row_items(
            self.evaluation_row_ids_by_stratum
        )
        append_metrics = _canonical_metrics(
            self.append_only_metrics,
            field_name="append_only_metrics",
        )
        matched_metrics = _canonical_metrics(
            self.matched_partial_moe_metrics,
            field_name="matched_partial_moe_metrics",
        )
        if (
            not isinstance(self.router_only_trials, tuple)
            or not self.router_only_trials
            or not all(
                isinstance(trial, RouterDataScalingTrial)
                for trial in self.router_only_trials
            )
        ):
            raise TypeError(
                "router_only_trials must be a non-empty tuple of trials."
            )
        trials = tuple(
            sorted(
                self.router_only_trials,
                key=lambda trial: trial.router_training_sample_count,
            )
        )
        sample_counts = tuple(
            trial.router_training_sample_count for trial in trials
        )
        if len(set(sample_counts)) != len(sample_counts):
            raise RouterDataScalingEvaluationError(
                "router_only_trials must have unique training sample counts."
            )
        if not isinstance(
            self.matched_partial_moe_provenance,
            MatchedPartialMoEProvenance,
        ):
            raise TypeError(
                "matched_partial_moe_provenance must be "
                "MatchedPartialMoEProvenance."
            )

        evaluation_sets = {
            stratum: set(row_ids) for stratum, row_ids in evaluation_rows
        }
        all_evaluation_rows: set[int] = set()
        for stratum in ROUTER_DATA_STRATA:
            overlap = all_evaluation_rows & evaluation_sets[stratum]
            if overlap:
                raise RouterDataScalingEvaluationError(
                    "evaluation row strata must be disjoint; overlap includes "
                    f"{min(overlap)}."
                )
            all_evaluation_rows.update(evaluation_sets[stratum])

        previous_training_rows: set[int] = set()
        registry_sha256: str | None = None
        compatibility_sha256: str | None = None
        feature_bank_hashes: set[str] = set()
        utility_label_hashes: set[str] = set()
        matched_layers = self.matched_partial_moe_provenance.layer_indices
        matched_experts = self.matched_partial_moe_provenance.expert_ids
        for trial in trials:
            training_rows = set(trial.router_training_row_ids)
            if previous_training_rows and not previous_training_rows < training_rows:
                raise RouterDataScalingEvaluationError(
                    "router training row subsets must grow strictly and retain "
                    "every row from the preceding sample count."
                )
            overlap = training_rows & all_evaluation_rows
            if overlap:
                raise RouterDataScalingEvaluationError(
                    "router training and held-out evaluation rows must be "
                    f"disjoint; overlap includes {min(overlap)}."
                )
            previous_training_rows = training_rows

            recalibration = trial.recalibration_report
            if registry_sha256 is None:
                registry_sha256 = recalibration.registry_sha256
                compatibility_sha256 = recalibration.compatibility_sha256
            elif (
                recalibration.registry_sha256 != registry_sha256
                or recalibration.compatibility_sha256 != compatibility_sha256
            ):
                raise RouterDataScalingEvaluationError(
                    "all router-only trials must target one registry and "
                    "compatibility identity."
                )
            if (
                recalibration.feature_bank_metadata_sha256
                in feature_bank_hashes
                or recalibration.utility_label_metadata_sha256
                in utility_label_hashes
            ):
                raise RouterDataScalingEvaluationError(
                    "each sample count must carry distinct RTR-004 feature and "
                    "utility provenance."
                )
            feature_bank_hashes.add(
                recalibration.feature_bank_metadata_sha256
            )
            utility_label_hashes.add(
                recalibration.utility_label_metadata_sha256
            )
            layer_indices = tuple(
                layer.layer_index for layer in recalibration.layer_reports
            )
            if layer_indices != matched_layers:
                raise RouterDataScalingEvaluationError(
                    "RTR-004 and matched TRN-003 layer topologies differ."
                )
            if any(
                layer.expert_ids != matched_experts
                for layer in recalibration.layer_reports
            ):
                raise RouterDataScalingEvaluationError(
                    "RTR-004 and matched TRN-003 expert topologies differ."
                )

        expected_counts = {
            stratum: len(row_ids) for stratum, row_ids in evaluation_rows
        }
        for metrics in (
            append_metrics,
            matched_metrics,
            *(trial.metrics for trial in trials),
        ):
            for metric in metrics:
                if metric.evaluation_row_count != expected_counts[metric.stratum]:
                    raise RouterDataScalingEvaluationError(
                        f"{metric.stratum} metric row count does not match its "
                        "held-out row provenance."
                    )

        expected_gaps = _calculate_gaps(
            append_metrics,
            trials,
            matched_metrics,
            higher_is_better=self.higher_is_better,
        )
        if self.gaps != expected_gaps:
            raise RouterDataScalingEvaluationError(
                "gaps must exactly match the declared strategy metrics."
            )

        object.__setattr__(
            self,
            "evaluation_row_ids_by_stratum",
            evaluation_rows,
        )
        object.__setattr__(self, "append_only_metrics", append_metrics)
        object.__setattr__(self, "router_only_trials", trials)
        object.__setattr__(
            self,
            "matched_partial_moe_metrics",
            matched_metrics,
        )

    @property
    def router_training_sample_counts(self) -> tuple[int, ...]:
        """Return the deterministic increasing replay-subset schedule."""

        return tuple(
            trial.router_training_sample_count
            for trial in self.router_only_trials
        )

    @property
    def registry_sha256(self) -> str:
        """Return the common active-registry identity of RTR-004 trials."""

        return self.router_only_trials[0].recalibration_report.registry_sha256

    @property
    def compatibility_sha256(self) -> str:
        """Return the common anchor/feature compatibility identity."""

        return self.router_only_trials[
            0
        ].recalibration_report.compatibility_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible, plot-ready report."""

        return {
            "append_only": {
                "metrics": [
                    metric.to_dict() for metric in self.append_only_metrics
                ],
                "strategy": self.append_only_strategy,
            },
            "compatibility_sha256": self.compatibility_sha256,
            "evaluation_row_ids_by_stratum": {
                stratum: list(row_ids)
                for stratum, row_ids in self.evaluation_row_ids_by_stratum
            },
            "feature_id": self.feature_id,
            "format_version": self.format_version,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "higher_is_better": self.higher_is_better,
            "matched_partial_moe": {
                "metrics": [
                    metric.to_dict()
                    for metric in self.matched_partial_moe_metrics
                ],
                "provenance": self.matched_partial_moe_provenance.to_dict(),
                "strategy": self.matched_strategy,
            },
            "metric_name": self.metric_name,
            "registry_sha256": self.registry_sha256,
            "router_only_trials": [
                trial.to_dict() for trial in self.router_only_trials
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouterDataScalingReport:
        """Strictly reconstruct and revalidate an EVAL-007 report."""

        _require_exact_fields(
            value,
            {
                "append_only",
                "compatibility_sha256",
                "evaluation_row_ids_by_stratum",
                "feature_id",
                "format_version",
                "gaps",
                "higher_is_better",
                "matched_partial_moe",
                "metric_name",
                "registry_sha256",
                "router_only_trials",
            },
            field_name="router-data scaling report",
        )
        if value["feature_id"] != cls.feature_id:
            raise ValueError("Unsupported router-data scaling feature ID.")
        if value["format_version"] != cls.format_version:
            raise ValueError("Unsupported router-data scaling format version.")
        append_only = value["append_only"]
        matched = value["matched_partial_moe"]
        _require_exact_fields(
            append_only,
            {"metrics", "strategy"},
            field_name="append-only result",
        )
        _require_exact_fields(
            matched,
            {"metrics", "provenance", "strategy"},
            field_name="matched partial-MoE result",
        )
        if append_only["strategy"] != cls.append_only_strategy:
            raise ValueError("Unsupported append-only strategy.")
        if matched["strategy"] != cls.matched_strategy:
            raise ValueError("Unsupported matched partial-MoE strategy.")
        evaluation_rows = value["evaluation_row_ids_by_stratum"]
        if not isinstance(evaluation_rows, Mapping):
            raise TypeError(
                "evaluation_row_ids_by_stratum must be a mapping."
            )
        if set(evaluation_rows) != set(ROUTER_DATA_STRATA):
            raise ValueError(
                "evaluation_row_ids_by_stratum must contain exactly "
                f"{ROUTER_DATA_STRATA}."
            )
        report = cls(
            metric_name=value["metric_name"],
            higher_is_better=value["higher_is_better"],
            evaluation_row_ids_by_stratum=tuple(
                (
                    stratum,
                    _integer_tuple(
                        evaluation_rows[stratum],
                        field_name=f"evaluation rows for {stratum}",
                    ),
                )
                for stratum in ROUTER_DATA_STRATA
            ),
            append_only_metrics=_metrics_from_json(append_only["metrics"]),
            router_only_trials=_trials_from_json(
                value["router_only_trials"]
            ),
            matched_partial_moe_metrics=_metrics_from_json(
                matched["metrics"]
            ),
            matched_partial_moe_provenance=(
                MatchedPartialMoEProvenance.from_dict(matched["provenance"])
            ),
            gaps=_gaps_from_json(value["gaps"]),
        )
        if report.registry_sha256 != value["registry_sha256"]:
            raise ValueError("Serialized registry_sha256 is inconsistent.")
        if report.compatibility_sha256 != value["compatibility_sha256"]:
            raise ValueError(
                "Serialized compatibility_sha256 is inconsistent."
            )
        return report


def evaluate_router_data_scaling(
    *,
    metric_name: str,
    higher_is_better: bool,
    evaluation_row_ids_by_stratum: Mapping[str, Sequence[int]],
    append_only_metrics: Sequence[RouterDataScalingMetrics],
    router_only_trials: Sequence[RouterDataScalingTrial],
    matched_partial_moe_metrics: Sequence[RouterDataScalingMetrics],
    matched_partial_moe_training: LocalMiniMoETrainingReport,
) -> RouterDataScalingReport:
    """Validate and aggregate the EVAL-007 router-subset experiment.

    Training remains outside this function. Each router-only trial must carry
    its immutable RTR-004 report, and the fixed matched baseline must carry a
    completed TRN-003 report. The function verifies that replay subsets are
    strictly nested, that held-out strata and training rows never overlap, and
    that the compared router and partial MoE have identical layer/expert
    topology.

    Performance gaps are direction-normalized:

    ``gain_append = s * (P_router - P_append)``

    ``gap_partial = s * (P_partial - P_router)``

    where ``s = 1`` for higher-is-better metrics and ``s = -1`` otherwise.
    Router-regret differences always use lower-is-better semantics.

    Args:
        metric_name:
            Stable name of the held-out performance metric.
        higher_is_better:
            Whether a larger performance value is favorable.
        evaluation_row_ids_by_stratum:
            Exact disjoint held-out row IDs for all required strata.
        append_only_metrics:
            Fixed strict append-only baseline metrics.
        router_only_trials:
            Increasing replay-subset RTR-004 trials.
        matched_partial_moe_metrics:
            Fixed metrics from the matched local partial MoE.
        matched_partial_moe_training:
            Completed TRN-003 report proving the baseline topology and run.

    Returns:
        Immutable, deterministic, JSON-compatible scaling report.

    Raises:
        TypeError:
            If a public input does not use its declared contract.
        ValueError:
            If split, metric, topology, provenance, or grid invariants fail.

    References:
        The repository architecture blueprint, section 19.3,
        "Router-subset hypothesis test."

    Differences from the blueprint:
        This initial experiment contract consumes already completed training
        and evaluation measurements. It does not prescribe a dataset, train a
        jointly initialized conventional MoE, require monotonic improvement,
        or add a plotting dependency. TRN-003 supplies the matched two-expert
        partial-MoE baseline supported by the current framework.
    """

    _identifier(metric_name, field_name="metric_name")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be a bool.")
    if not isinstance(evaluation_row_ids_by_stratum, Mapping):
        raise TypeError(
            "evaluation_row_ids_by_stratum must be a mapping."
        )
    if set(evaluation_row_ids_by_stratum) != set(ROUTER_DATA_STRATA):
        raise ValueError(
            "evaluation_row_ids_by_stratum must contain exactly "
            f"{ROUTER_DATA_STRATA}."
        )
    evaluation_rows = tuple(
        (
            stratum,
            _row_id_tuple(
                evaluation_row_ids_by_stratum[stratum],
                field_name=f"evaluation rows for {stratum}",
            ),
        )
        for stratum in ROUTER_DATA_STRATA
    )
    append = _canonical_metrics(
        append_only_metrics,
        field_name="append_only_metrics",
    )
    matched = _canonical_metrics(
        matched_partial_moe_metrics,
        field_name="matched_partial_moe_metrics",
    )
    trials = _canonical_trials(router_only_trials)
    provenance = MatchedPartialMoEProvenance.from_training_report(
        matched_partial_moe_training
    )
    gaps = _calculate_gaps(
        append,
        trials,
        matched,
        higher_is_better=higher_is_better,
    )
    return RouterDataScalingReport(
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        evaluation_row_ids_by_stratum=evaluation_rows,
        append_only_metrics=append,
        router_only_trials=trials,
        matched_partial_moe_metrics=matched,
        matched_partial_moe_provenance=provenance,
        gaps=gaps,
    )


def _calculate_gaps(
    append_only_metrics: tuple[RouterDataScalingMetrics, ...],
    trials: tuple[RouterDataScalingTrial, ...],
    matched_metrics: tuple[RouterDataScalingMetrics, ...],
    *,
    higher_is_better: bool,
) -> tuple[RouterDataScalingGap, ...]:
    """Calculate the complete deterministic sample-count/stratum grid."""

    append_by_stratum = {
        metric.stratum: metric for metric in append_only_metrics
    }
    matched_by_stratum = {
        metric.stratum: metric for metric in matched_metrics
    }
    direction = 1.0 if higher_is_better else -1.0
    gaps: list[RouterDataScalingGap] = []
    for trial in trials:
        router_by_stratum = {
            metric.stratum: metric for metric in trial.metrics
        }
        for stratum in ROUTER_DATA_STRATA:
            append = append_by_stratum[stratum]
            router = router_by_stratum[stratum]
            matched = matched_by_stratum[stratum]
            gaps.append(
                RouterDataScalingGap(
                    router_training_sample_count=(
                        trial.router_training_sample_count
                    ),
                    stratum=stratum,
                    performance_gain_over_append_only=direction
                    * (router.performance - append.performance),
                    performance_gap_to_partial_moe=direction
                    * (matched.performance - router.performance),
                    regret_reduction_over_append_only=(
                        append.mean_router_regret
                        - router.mean_router_regret
                    ),
                    regret_gap_to_partial_moe=(
                        router.mean_router_regret
                        - matched.mean_router_regret
                    ),
                )
            )
    return tuple(gaps)


def _canonical_metrics(
    value: Sequence[RouterDataScalingMetrics],
    *,
    field_name: str,
) -> tuple[RouterDataScalingMetrics, ...]:
    """Return one exact, canonical three-stratum metric block."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of metrics.")
    metrics = tuple(value)
    if not all(isinstance(metric, RouterDataScalingMetrics) for metric in metrics):
        raise TypeError(
            f"{field_name} must contain RouterDataScalingMetrics."
        )
    by_stratum = {metric.stratum: metric for metric in metrics}
    if len(by_stratum) != len(metrics):
        raise ValueError(f"{field_name} must not repeat a stratum.")
    if set(by_stratum) != set(ROUTER_DATA_STRATA):
        raise ValueError(
            f"{field_name} must cover exactly {ROUTER_DATA_STRATA}."
        )
    return tuple(by_stratum[stratum] for stratum in ROUTER_DATA_STRATA)


def _canonical_trials(
    value: Sequence[RouterDataScalingTrial],
) -> tuple[RouterDataScalingTrial, ...]:
    """Return router trials in increasing sample-count order."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("router_only_trials must be a sequence.")
    trials = tuple(value)
    if not trials or not all(
        isinstance(trial, RouterDataScalingTrial) for trial in trials
    ):
        raise TypeError(
            "router_only_trials must contain at least one "
            "RouterDataScalingTrial."
        )
    return tuple(
        sorted(
            trials,
            key=lambda trial: trial.router_training_sample_count,
        )
    )


def _evaluation_row_items(
    value: Any,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Validate the immutable canonical evaluation-row representation."""

    if not isinstance(value, tuple) or len(value) != len(ROUTER_DATA_STRATA):
        raise TypeError(
            "evaluation_row_ids_by_stratum must be a canonical tuple."
        )
    if any(
        not isinstance(item, tuple) or len(item) != 2 for item in value
    ):
        raise TypeError(
            "evaluation row entries must be (stratum, row_ids) tuples."
        )
    raw = {stratum: row_ids for stratum, row_ids in value}
    if len(raw) != len(value) or set(raw) != set(ROUTER_DATA_STRATA):
        raise ValueError(
            "evaluation row entries must cover each required stratum once."
        )
    return tuple(
        (
            stratum,
            _row_id_tuple(
                raw[stratum],
                field_name=f"evaluation rows for {stratum}",
            ),
        )
        for stratum in ROUTER_DATA_STRATA
    )


def _metrics_from_json(value: Any) -> tuple[RouterDataScalingMetrics, ...]:
    """Parse a JSON metric list."""

    if not isinstance(value, list):
        raise TypeError("serialized metrics must be a list.")
    return tuple(RouterDataScalingMetrics.from_dict(metric) for metric in value)


def _trials_from_json(value: Any) -> tuple[RouterDataScalingTrial, ...]:
    """Parse a JSON trial list."""

    if not isinstance(value, list):
        raise TypeError("router_only_trials must be a list.")
    return tuple(RouterDataScalingTrial.from_dict(trial) for trial in value)


def _gaps_from_json(value: Any) -> tuple[RouterDataScalingGap, ...]:
    """Parse a JSON gap list."""

    if not isinstance(value, list):
        raise TypeError("gaps must be a list.")
    return tuple(RouterDataScalingGap.from_dict(gap) for gap in value)


def _row_id_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    """Normalize unique non-negative row identifiers."""

    row_ids = _integer_tuple(value, field_name=field_name)
    if not row_ids:
        raise ValueError(f"{field_name} must not be empty.")
    if any(row_id < 0 for row_id in row_ids):
        raise ValueError(f"{field_name} must be non-negative.")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{field_name} must contain unique values.")
    return row_ids


def _integer_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    """Normalize a JSON-style integer sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of integers.")
    normalized = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in normalized):
        raise TypeError(f"{field_name} must contain integers.")
    return normalized


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Normalize a JSON-style string sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(value)
    if any(not isinstance(item, str) for item in normalized):
        raise TypeError(f"{field_name} must contain strings.")
    return normalized


def _positive_integer(value: Any, *, field_name: str) -> int:
    """Validate a strictly positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _finite_number(value: Any, *, field_name: str) -> float:
    """Normalize one finite real scalar."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite real number.")
    return float(value)


def _non_negative_number(value: Any, *, field_name: str) -> float:
    """Normalize one finite non-negative real scalar."""

    normalized = _finite_number(value, field_name=field_name)
    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized


def _identifier(value: Any, *, field_name: str) -> str:
    """Validate a stable non-empty identifier."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace."
        )
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    """Validate a lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _require_exact_fields(
    value: Any,
    expected: set[str],
    *,
    field_name: str,
) -> None:
    """Require one mapping with no missing or unexpected JSON fields."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field_name} fields are incompatible: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
