"""Lifecycle evaluation for install, removal, and deterministic reinsertion."""

from __future__ import annotations

from collections.abc import Callable

from anchored_moe import ExpertPackage, ExpertRegistry
from anchored_moe.evaluation import evaluate_registry_lifecycle


def test_registry_lifecycle_restores_identity_order_and_compatibility(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    code = expert_package_factory("code")
    math = expert_package_factory("math")
    registry = ExpertRegistry()
    registry.install(code)
    registry.install(math)
    installed = registry.snapshot()

    registry.unload(code.identity)
    removed = registry.snapshot()
    registry.install(
        ExpertPackage.from_directory(code.package_directory)
    )
    reinserted = registry.snapshot()

    evaluation = evaluate_registry_lifecycle(
        installed,
        removed,
        reinserted,
    )

    assert evaluation.exact
    assert evaluation.removal_changed_registry
    assert evaluation.removal_reduced_active_packages
    assert evaluation.reinsertion_restored_hash
    assert evaluation.reinsertion_restored_order
    assert evaluation.compatibility_remained_bound
