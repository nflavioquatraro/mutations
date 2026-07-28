"""Integration of EVAL-006 with strict package and registry lifecycle APIs."""

from __future__ import annotations

from collections.abc import Callable

from anchored_moe import (
    ExpertPackage,
    ExpertRegistry,
    evaluate_registry_compositions,
)
from anchored_moe.utils import sha256_json


def _package_set_fingerprint(registry: ExpertRegistry) -> str:
    """Fingerprint active package content without installation order."""

    identities = sorted(registry.snapshot().active_packages)
    return sha256_json(
        {"packages": [identity.to_dict() for identity in identities]}
    )


def _ordered_registry_fingerprint(registry: ExpertRegistry) -> str:
    """Fingerprint active package content in runtime registry order."""

    identities = registry.snapshot().active_packages
    return sha256_json(
        {"packages": [identity.to_dict() for identity in identities]}
    )


def test_every_subset_and_order_are_deterministic_and_json_compatible(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    code = expert_package_factory("code")
    math = expert_package_factory("math")

    first = evaluate_registry_compositions(
        (math, code),
        behavior_fingerprint=_package_set_fingerprint,
    )
    second = evaluate_registry_compositions(
        (code, math),
        behavior_fingerprint=_package_set_fingerprint,
    )

    assert first == second
    assert first.package_ids == (code.identity, math.identity)
    assert first.subset_count == 4
    assert first.trial_count == 5
    assert tuple(
        subset.package_count for subset in first.subset_evaluations
    ) == (0, 1, 1, 2)
    assert tuple(
        subset.installation_order_count
        for subset in first.subset_evaluations
    ) == (1, 1, 1, 2)
    assert first.registry_order_sensitive_subset_count == 1
    assert first.behavior_order_sensitive_subset_count == 0
    assert first.behavior_order_invariant is True

    pair_subset = first.subset_evaluations[-1]
    assert pair_subset.registry_order_sensitive
    assert pair_subset.distinct_registry_hash_count == 2
    assert pair_subset.distinct_behavior_count == 1
    assert pair_subset.behavior_order_invariant is True
    assert pair_subset.trials[0].installation_order == (
        code.identity,
        math.identity,
    )
    assert pair_subset.trials[1].installation_order == (
        math.identity,
        code.identity,
    )

    serialized = first.to_dict()
    assert serialized["subset_count"] == 4
    assert serialized["trial_count"] == 5
    assert len(serialized["subset_evaluations"]) == 4


def test_behavior_fingerprint_exposes_installation_order_sensitivity(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    code = expert_package_factory("code")
    math = expert_package_factory("math")

    report = evaluate_registry_compositions(
        (code, math),
        behavior_fingerprint=_ordered_registry_fingerprint,
    )

    assert report.behavior_order_invariant is False
    assert report.behavior_order_sensitive_subset_count == 1
    assert (
        report.subset_evaluations[-1].distinct_behavior_count == 2
    )
    # Evaluation uses fresh internal registries and leaves package descriptors
    # valid for later lifecycle operations.
    code.validate_live_files()
    math.validate_live_files()
