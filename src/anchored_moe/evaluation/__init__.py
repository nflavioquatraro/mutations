"""Evaluation helpers for architectural invariants."""

from anchored_moe.evaluation.equivalence import (
    TensorEquivalence,
    assert_tensors_equivalent,
    compare_tensors,
    fingerprint_module,
)
from anchored_moe.evaluation.efficiency import (
    ActiveMLPMetrics,
    EfficiencyBenchmarkConfig,
    EfficiencyBenchmarkReport,
    ModuleStateFootprint,
    benchmark_operation,
    measure_active_mlp_metrics,
    measure_module_state_footprint,
)
from anchored_moe.evaluation.lifecycle import (
    RegistryLifecycleEvaluation,
    evaluate_registry_lifecycle,
)
from anchored_moe.evaluation.registry_composition import (
    RegistryCompositionEvaluation,
    RegistryCompositionEvaluationError,
    RegistryCompositionTrial,
    RegistrySubsetEvaluation,
    evaluate_registry_compositions,
)
from anchored_moe.evaluation.quality import (
    Top1Top2OracleComparison,
    compare_top1_top2_to_oracle,
)
from anchored_moe.evaluation.routing import (
    RouteAgreement,
    assert_route_decisions_equal,
    compare_route_decisions,
)

__all__ = [
    "ActiveMLPMetrics",
    "EfficiencyBenchmarkConfig",
    "EfficiencyBenchmarkReport",
    "ModuleStateFootprint",
    "RouteAgreement",
    "RegistryLifecycleEvaluation",
    "RegistryCompositionEvaluation",
    "RegistryCompositionEvaluationError",
    "RegistryCompositionTrial",
    "RegistrySubsetEvaluation",
    "TensorEquivalence",
    "Top1Top2OracleComparison",
    "assert_route_decisions_equal",
    "assert_tensors_equivalent",
    "benchmark_operation",
    "compare_route_decisions",
    "compare_top1_top2_to_oracle",
    "compare_tensors",
    "evaluate_registry_lifecycle",
    "evaluate_registry_compositions",
    "fingerprint_module",
    "measure_active_mlp_metrics",
    "measure_module_state_footprint",
]
