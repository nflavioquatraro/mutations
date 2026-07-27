"""Registry removal and reinsertion evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from anchored_moe.registry.registry import RegistrySnapshot


@dataclass(frozen=True)
class RegistryLifecycleEvaluation:
    """Comparison of installed, removed, and reinserted registry states.

    Attributes:
        removal_changed_registry:
            Whether unloading changed the active registry identity.
        removal_reduced_active_packages:
            Whether exactly one active package was removed.
        reinsertion_restored_hash:
            Whether reinsertion restored the installed registry digest.
        reinsertion_restored_order:
            Whether all active package identities returned in the same order.
        compatibility_remained_bound:
            Whether all snapshots retain one compatibility identity.
    """

    removal_changed_registry: bool
    removal_reduced_active_packages: bool
    reinsertion_restored_hash: bool
    reinsertion_restored_order: bool
    compatibility_remained_bound: bool

    @property
    def exact(self) -> bool:
        """Return whether the complete lifecycle contract was preserved."""

        return all(
            (
                self.removal_changed_registry,
                self.removal_reduced_active_packages,
                self.reinsertion_restored_hash,
                self.reinsertion_restored_order,
                self.compatibility_remained_bound,
            )
        )


def evaluate_registry_lifecycle(
    installed: RegistrySnapshot,
    removed: RegistrySnapshot,
    reinserted: RegistrySnapshot,
) -> RegistryLifecycleEvaluation:
    """Evaluate one install/unload/reinsert snapshot sequence.

    The function is read-only. It compares registry identities and package
    ordering without loading payload tensors or inspecting token content.

    Args:
        installed: State immediately before unloading one package.
        removed: State immediately after that unload.
        reinserted: State after installing identical package content again.

    Returns:
        A structured lifecycle evaluation suitable for tests and reports.

    Raises:
        TypeError: If any input is not a :class:`RegistrySnapshot`.
    """

    snapshots = (installed, removed, reinserted)
    if not all(isinstance(snapshot, RegistrySnapshot) for snapshot in snapshots):
        raise TypeError("Lifecycle inputs must be RegistrySnapshot values.")
    compatibility = installed.compatibility_sha256
    return RegistryLifecycleEvaluation(
        removal_changed_registry=(
            removed.registry_sha256 != installed.registry_sha256
        ),
        removal_reduced_active_packages=(
            removed.active_package_count
            == installed.active_package_count - 1
        ),
        reinsertion_restored_hash=(
            reinserted.registry_sha256 == installed.registry_sha256
        ),
        reinsertion_restored_order=(
            reinserted.active_packages == installed.active_packages
        ),
        compatibility_remained_bound=(
            compatibility is not None
            and removed.compatibility_sha256 == compatibility
            and reinserted.compatibility_sha256 == compatibility
        ),
    )
