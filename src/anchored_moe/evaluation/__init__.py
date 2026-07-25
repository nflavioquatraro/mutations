"""Evaluation helpers for architectural invariants."""

from anchored_moe.evaluation.equivalence import (
    TensorEquivalence,
    assert_tensors_equivalent,
    compare_tensors,
    fingerprint_module,
)

__all__ = [
    "TensorEquivalence",
    "assert_tensors_equivalent",
    "compare_tensors",
    "fingerprint_module",
]
