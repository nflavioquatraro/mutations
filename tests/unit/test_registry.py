"""Acceptance tests for atomic expert-registry lifecycle."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from anchored_moe import (
    ExpertPackage,
    ExpertRegistry,
    PackageIdentity,
    PackageValidationError,
    RegistryLifecycleError,
    RegistryUpgradeChange,
)


def test_install_unload_and_reinsert_restore_order_and_hash(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    code = expert_package_factory("code")
    math = expert_package_factory("math")
    registry = ExpertRegistry()
    empty_hash = registry.active_registry_sha256

    code_change = registry.install(code)
    installed_change = registry.install(math)
    installed = registry.snapshot()

    assert code_change.operation == "install"
    assert code_change.before.registry_sha256 == empty_hash
    assert installed.active_packages == (code.identity, math.identity)
    assert installed.registry_sha256 != empty_hash
    assert registry.active_packages == (code, math)

    removed_change = registry.unload(code.identity)
    removed = registry.snapshot()
    assert removed_change.operation == "unload"
    assert removed.active_packages == (math.identity,)

    reinserted_change = registry.install(
        ExpertPackage.from_directory(code.package_directory)
    )
    reinserted = registry.snapshot()
    assert reinserted_change.operation == "install"
    assert reinserted.active_packages == installed.active_packages
    assert reinserted.registry_sha256 == installed.registry_sha256
    assert reinserted.compatibility_sha256 == installed.compatibility_sha256


def test_failed_operations_leave_registry_unchanged(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    code = expert_package_factory("code")
    registry = ExpertRegistry()
    registry.install(code)
    installed = registry.snapshot()

    with pytest.raises(RegistryLifecycleError, match="already active"):
        registry.install(code)
    assert registry.snapshot() == installed

    unknown = PackageIdentity("math", "1.0.0", "a" * 64)
    with pytest.raises(RegistryLifecycleError, match="not active"):
        registry.unload(unknown)
    assert registry.snapshot() == installed


@pytest.mark.parametrize(
    "incompatible_field",
    ["base_model_revision", "router_output_size"],
)
def test_registry_rejects_incompatible_anchor_or_feature_space_atomically(
    expert_package_factory: Callable[..., ExpertPackage],
    incompatible_field: str,
) -> None:
    code = expert_package_factory("code")
    kwargs = (
        {"base_model_revision": "different-revision"}
        if incompatible_field == "base_model_revision"
        else {"router_output_size": 8}
    )
    incompatible = expert_package_factory("math", **kwargs)
    registry = ExpertRegistry()
    registry.install(code)
    installed = registry.snapshot()

    with pytest.raises(RegistryLifecycleError, match="incompatible"):
        registry.install(incompatible)

    assert registry.snapshot() == installed
    assert registry.active_packages == (code,)


def test_install_revalidates_live_files_before_mutation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory("code")
    payload = package.package_directory / package.manifest.expert_weight_files[0]
    payload.write_bytes(payload.read_bytes() + b"tampered")
    registry = ExpertRegistry()
    empty = registry.snapshot()

    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        registry.install(package)

    assert registry.snapshot() == empty


def test_registry_requires_strict_package_validation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    package = expert_package_factory(
        "code",
        strict=False,
        add_untracked_file=True,
    )
    registry = ExpertRegistry()

    with pytest.raises(RegistryLifecycleError, match="strict"):
        registry.install(package)

    assert registry.snapshot().active_package_count == 0


def test_immutable_version_content_cannot_change_after_unload(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    first = expert_package_factory("code", tensor_offset=0.0)
    changed = expert_package_factory("code", tensor_offset=1.0)
    registry = ExpertRegistry()
    registry.install(first)
    registry.unload(first.identity)
    empty_bound = registry.snapshot()

    with pytest.raises(RegistryLifecycleError, match="content changed"):
        registry.install(changed)

    assert registry.snapshot() == empty_bound
    assert registry.compatibility_sha256 is not None


def test_empty_registry_hash_is_deterministic() -> None:
    first = ExpertRegistry().snapshot()
    second = ExpertRegistry().snapshot()

    assert first.registry_sha256 == second.registry_sha256
    assert first.active_packages == second.active_packages == ()
    assert first.compatibility_sha256 is second.compatibility_sha256 is None


def test_upgrade_and_rollback_replace_one_version_in_the_same_slot(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """An upgrade must preserve order and an exact rollback must restore hash."""

    code_v1 = expert_package_factory(
        "code",
        specialization_version="1.0.0",
        tensor_offset=0.0,
    )
    code_v2 = expert_package_factory(
        "code",
        specialization_version="2.0.0",
        tensor_offset=1.0,
    )
    math = expert_package_factory("math")
    registry = ExpertRegistry()
    registry.install(code_v1)
    registry.install(math)
    before = registry.snapshot()

    upgrade = registry.upgrade(code_v1.identity, code_v2)

    assert isinstance(upgrade, RegistryUpgradeChange)
    assert upgrade.operation == "upgrade"
    assert upgrade.removed_package == code_v1.identity
    assert upgrade.activated_package == code_v2.identity
    assert upgrade.before == before
    assert upgrade.after == registry.snapshot()
    assert upgrade.after.active_packages == (code_v2.identity, math.identity)
    assert registry.active_packages == (code_v2, math)

    rollback = registry.rollback(upgrade, code_v1)

    assert rollback.operation == "rollback"
    assert rollback.removed_package == code_v2.identity
    assert rollback.activated_package == code_v1.identity
    assert rollback.before == upgrade.after
    assert rollback.after == before
    assert registry.active_packages == (code_v1, math)
    assert registry.active_registry_sha256 == before.registry_sha256

    # A rolled-back replacement remains immutable and can be activated again
    # through a new explicit upgrade without receiving a second slot.
    repeated = registry.upgrade(code_v1.identity, code_v2)
    assert repeated.after.active_packages == (code_v2.identity, math.identity)


@pytest.mark.parametrize(
    ("replacement_kwargs", "error_match"),
    [
        (
            {
                "specialization_name": "math",
                "specialization_version": "2.0.0",
            },
            "same specialization_name",
        ),
        (
            {
                "specialization_name": "code",
                "specialization_version": "1.0.0",
                "tensor_offset": 1.0,
            },
            "different specialization_version",
        ),
        (
            {
                "specialization_name": "code",
                "specialization_version": "2.0.0",
                "router_output_size": 8,
            },
            "incompatible",
        ),
        (
            {
                "specialization_name": "code",
                "specialization_version": "2.0.0",
                "strict": False,
                "add_untracked_file": True,
            },
            "strict",
        ),
    ],
)
def test_invalid_upgrade_is_rejected_without_mutation(
    expert_package_factory: Callable[..., ExpertPackage],
    replacement_kwargs: dict[str, object],
    error_match: str,
) -> None:
    """All upgrade compatibility failures must precede registry mutation."""

    current = expert_package_factory("code")
    replacement = expert_package_factory(**replacement_kwargs)
    registry = ExpertRegistry()
    registry.install(current)
    before = registry.snapshot()

    with pytest.raises(RegistryLifecycleError, match=error_match):
        registry.upgrade(current.identity, replacement)

    assert registry.snapshot() == before
    assert registry.active_packages == (current,)


def test_upgrade_revalidates_live_files_before_mutation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """A replacement tampered after loading must never enter active state."""

    current = expert_package_factory("code")
    replacement = expert_package_factory(
        "code",
        specialization_version="2.0.0",
    )
    payload = (
        replacement.package_directory
        / replacement.manifest.expert_weight_files[0]
    )
    payload.write_bytes(payload.read_bytes() + b"tampered")
    registry = ExpertRegistry()
    registry.install(current)
    before = registry.snapshot()

    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        registry.upgrade(current.identity, replacement)

    assert registry.snapshot() == before
    assert registry.active_packages == (current,)


def test_rollback_rejects_stale_registry_state_without_mutation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Rollback must not erase an intervening successful registry change."""

    code_v1 = expert_package_factory("code")
    code_v2 = expert_package_factory(
        "code",
        specialization_version="2.0.0",
    )
    math = expert_package_factory("math")
    registry = ExpertRegistry()
    registry.install(code_v1)
    upgrade = registry.upgrade(code_v1.identity, code_v2)
    registry.install(math)
    intervening = registry.snapshot()

    with pytest.raises(RegistryLifecycleError, match="stale"):
        registry.rollback(upgrade, code_v1)

    assert registry.snapshot() == intervening
    assert registry.active_packages == (code_v2, math)


def test_rollback_revalidates_previous_package_without_mutation(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Rollback must fail atomically when the retained old package changed."""

    code_v1 = expert_package_factory("code")
    code_v2 = expert_package_factory(
        "code",
        specialization_version="2.0.0",
    )
    registry = ExpertRegistry()
    registry.install(code_v1)
    upgrade = registry.upgrade(code_v1.identity, code_v2)
    payload = (
        code_v1.package_directory
        / code_v1.manifest.expert_weight_files[0]
    )
    payload.write_bytes(payload.read_bytes() + b"tampered")
    upgraded = registry.snapshot()

    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        registry.rollback(upgrade, code_v1)

    assert registry.snapshot() == upgraded
    assert registry.active_packages == (code_v2,)


def test_package_and_registry_identity_exclude_local_paths(
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    first_copy = expert_package_factory("code")
    second_copy = expert_package_factory("code")
    first_registry = ExpertRegistry()
    second_registry = ExpertRegistry()

    first_registry.install(first_copy)
    second_registry.install(second_copy)

    assert first_copy.package_directory != second_copy.package_directory
    assert first_copy.identity == second_copy.identity
    assert (
        first_registry.active_registry_sha256
        == second_registry.active_registry_sha256
    )
