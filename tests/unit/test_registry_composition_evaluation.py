"""Unit contracts for bounded EVAL-006 registry composition experiments."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from anchored_moe import (
    ExpertPackage,
    RegistryCompositionEvaluationError,
    RegistryLifecycleError,
    evaluate_registry_compositions,
)


def test_empty_package_collection_evaluates_anchor_registry_once() -> None:
    report = evaluate_registry_compositions(())

    assert report.package_ids == ()
    assert report.subset_count == 1
    assert report.trial_count == 1
    assert report.registry_order_sensitive_subset_count == 0
    assert report.behavior_order_invariant is None
    assert report.behavior_order_sensitive_subset_count is None
    empty = report.subset_evaluations[0]
    assert empty.package_set == ()
    assert empty.installation_order_count == 1
    assert empty.trials[0].snapshot.active_packages == ()
    assert empty.to_dict()["behavior_order_invariant"] is None


def test_trial_budget_rejects_factorial_work_before_observation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    packages = tuple(
        expert_package_factory(name) for name in ("code", "math", "bio")
    )
    callback_calls = 0

    def observer(_registry: object) -> str:
        nonlocal callback_calls
        callback_calls += 1
        return "0" * 64

    with pytest.raises(
        RegistryCompositionEvaluationError,
        match=r"16 trials.*maximum_trials=15",
    ):
        evaluate_registry_compositions(
            packages,
            behavior_fingerprint=observer,
            maximum_trials=15,
        )

    assert callback_calls == 0


def test_behavior_callback_cannot_mutate_temporary_registry(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code")

    def mutating_observer(registry: object) -> str:
        active_identities = registry.active_identities
        if active_identities:
            registry.unload(active_identities[0])
        return "0" * 64

    with pytest.raises(
        RegistryCompositionEvaluationError,
        match="mutated the evaluated registry",
    ):
        evaluate_registry_compositions(
            (package,),
            behavior_fingerprint=mutating_observer,
        )


@pytest.mark.parametrize("fingerprint", ["not-a-digest", "A" * 64, 7])
def test_behavior_callback_requires_lowercase_sha256(
    expert_package_factory: Callable[..., ExpertPackage],
    fingerprint: object,
) -> None:
    package = expert_package_factory("code")

    with pytest.raises(
        RegistryCompositionEvaluationError,
        match="lowercase 64-character",
    ):
        evaluate_registry_compositions(
            (package,),
            behavior_fingerprint=lambda _registry: fingerprint,
        )


def test_duplicate_versions_and_incompatibility_fail_during_preflight(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    first = expert_package_factory("code", tensor_offset=0.0)
    changed = expert_package_factory("code", tensor_offset=1.0)
    incompatible = expert_package_factory(
        "math",
        base_model_revision="different",
    )
    callback_calls = 0

    def observer(_registry: object) -> str:
        nonlocal callback_calls
        callback_calls += 1
        return "0" * 64

    with pytest.raises(ValueError, match="conflicting content"):
        evaluate_registry_compositions((first, changed))
    with pytest.raises(RegistryLifecycleError, match="incompatible"):
        evaluate_registry_compositions(
            (first, incompatible),
            behavior_fingerprint=observer,
        )

    assert callback_calls == 0


@pytest.mark.parametrize(
    ("packages", "kwargs", "error_type", "message"),
    [
        ("package", {}, TypeError, "sequence"),
        ((object(),), {}, TypeError, "only ExpertPackage"),
        ((), {"behavior_fingerprint": 1}, TypeError, "callable"),
        ((), {"maximum_trials": True}, TypeError, "integer"),
        ((), {"maximum_trials": 0}, ValueError, "positive"),
    ],
)
def test_evaluator_rejects_invalid_public_inputs(
    packages: object,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        evaluate_registry_compositions(packages, **kwargs)
