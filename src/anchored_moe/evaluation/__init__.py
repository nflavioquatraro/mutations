"""Evaluation helpers for architectural invariants."""

from anchored_moe.evaluation.equivalence import (
    TensorEquivalence,
    assert_tensors_equivalent,
    compare_tensors,
    fingerprint_module,
)
from anchored_moe.evaluation.lifecycle import (
    RegistryLifecycleEvaluation,
    evaluate_registry_lifecycle,
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
    "RouteAgreement",
    "RegistryLifecycleEvaluation",
    "TensorEquivalence",
    "Top1Top2OracleComparison",
    "assert_route_decisions_equal",
    "assert_tensors_equivalent",
    "compare_route_decisions",
    "compare_top1_top2_to_oracle",
    "compare_tensors",
    "evaluate_registry_lifecycle",
    "fingerprint_module",
]
