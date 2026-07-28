"""Atomic in-memory lifecycle for validated expert packages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.utils.hashing import sha256_json

if TYPE_CHECKING:
    from anchored_moe.registry.compatibility import MixtureCompatibilityIndex


REGISTRY_FORMAT_VERSION = 1


class RegistryLifecycleError(ValueError):
    """Raised when a registry lifecycle operation violates its invariants."""


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable active-registry state used for audit and evaluation.

    Attributes:
        registry_format_version: Hash-payload schema version.
        registry_sha256: Deterministic identity of the ordered active packages.
        compatibility_sha256: Immutable anchor/router-feature compatibility
            identity established by the first successful installation, or
            ``None`` for a registry that has never installed a package.
        active_packages: Package identities in stable registry slot order.
    """

    registry_format_version: int
    registry_sha256: str
    compatibility_sha256: str | None
    active_packages: tuple[PackageIdentity, ...]

    def __post_init__(self) -> None:
        """Validate format, hashes, ordering, and active-package uniqueness."""

        if self.registry_format_version != REGISTRY_FORMAT_VERSION:
            raise ValueError(
                "Unsupported registry_format_version in RegistrySnapshot."
            )
        if not isinstance(self.active_packages, tuple) or not all(
            isinstance(package, PackageIdentity)
            for package in self.active_packages
        ):
            raise TypeError("active_packages must be a tuple of PackageIdentity.")
        version_keys = [package.version_key for package in self.active_packages]
        if len(set(version_keys)) != len(version_keys):
            raise ValueError("active_packages must not repeat a package version.")
        expected_hash = _registry_sha256(self.active_packages)
        if self.registry_sha256 != expected_hash:
            raise ValueError(
                "registry_sha256 does not match ordered active_packages."
            )
        if self.compatibility_sha256 is not None and (
            not isinstance(self.compatibility_sha256, str)
            or len(self.compatibility_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.compatibility_sha256
            )
        ):
            raise ValueError(
                "compatibility_sha256 must be a lowercase 64-character digest."
            )
        if self.active_packages and self.compatibility_sha256 is None:
            raise ValueError(
                "An active registry must have a compatibility identity."
            )

    @property
    def active_package_count(self) -> int:
        """Return the number of active package versions."""

        return len(self.active_packages)


@dataclass(frozen=True)
class RegistryChange:
    """Auditable result of one successful install or unload operation.

    Attributes:
        operation: Either ``"install"`` or ``"unload"``.
        package: Exact package identity added or removed.
        before: Registry snapshot before the atomic transition.
        after: Registry snapshot after the atomic transition.
    """

    operation: str
    package: PackageIdentity
    before: RegistrySnapshot
    after: RegistrySnapshot

    def __post_init__(self) -> None:
        """Validate operation name, types, membership, and count transition."""

        if self.operation not in {"install", "unload"}:
            raise ValueError("operation must be 'install' or 'unload'.")
        if not isinstance(self.package, PackageIdentity):
            raise TypeError("package must be a PackageIdentity.")
        if not isinstance(self.before, RegistrySnapshot) or not isinstance(
            self.after,
            RegistrySnapshot,
        ):
            raise TypeError("before and after must be RegistrySnapshot values.")
        before_has_package = self.package in self.before.active_packages
        after_has_package = self.package in self.after.active_packages
        if self.operation == "install" and (
            before_has_package
            or not after_has_package
            or self.after.active_package_count
            != self.before.active_package_count + 1
        ):
            raise ValueError("Install change does not describe one added package.")
        if self.operation == "unload" and (
            not before_has_package
            or after_has_package
            or self.after.active_package_count
            != self.before.active_package_count - 1
        ):
            raise ValueError("Unload change does not describe one removed package.")


@dataclass(frozen=True)
class RegistryUpgradeChange:
    """Auditable result of one atomic package-version replacement.

    Attributes:
        operation: Either ``"upgrade"`` or ``"rollback"``.
        removed_package: Exact package identity deactivated by the transition.
        activated_package: Exact package identity activated in the same slot.
        before: Registry snapshot before the atomic transition.
        after: Registry snapshot after the atomic transition.

    Upgrade and rollback preserve the active package count and specialization
    ordering. A rollback is deliberately represented as another replacement:
    its removed and activated identities are the reverse of the corresponding
    upgrade.
    """

    operation: str
    removed_package: PackageIdentity
    activated_package: PackageIdentity
    before: RegistrySnapshot
    after: RegistrySnapshot

    def __post_init__(self) -> None:
        """Validate operation, identities, and exact replacement semantics."""

        if self.operation not in {"upgrade", "rollback"}:
            raise ValueError("operation must be 'upgrade' or 'rollback'.")
        if not isinstance(self.removed_package, PackageIdentity) or not isinstance(
            self.activated_package,
            PackageIdentity,
        ):
            raise TypeError(
                "removed_package and activated_package must be PackageIdentity."
            )
        if not isinstance(self.before, RegistrySnapshot) or not isinstance(
            self.after,
            RegistrySnapshot,
        ):
            raise TypeError("before and after must be RegistrySnapshot values.")
        if (
            self.removed_package.specialization_name
            != self.activated_package.specialization_name
        ):
            raise ValueError(
                "A version replacement must preserve specialization_name."
            )
        if (
            self.removed_package.specialization_version
            == self.activated_package.specialization_version
        ):
            raise ValueError(
                "A version replacement must change specialization_version."
            )
        if self.before.active_package_count != self.after.active_package_count:
            raise ValueError(
                "A version replacement must preserve active package count."
            )

        before_packages = self.before.active_packages
        after_packages = self.after.active_packages
        if (
            before_packages.count(self.removed_package) != 1
            or self.activated_package in before_packages
            or after_packages.count(self.activated_package) != 1
            or self.removed_package in after_packages
        ):
            raise ValueError(
                "Version change snapshots do not describe one exact replacement."
            )
        removed_index = before_packages.index(self.removed_package)
        activated_index = after_packages.index(self.activated_package)
        if removed_index != activated_index:
            raise ValueError(
                "A version replacement must preserve the registry slot."
            )
        if (
            before_packages[:removed_index] != after_packages[:activated_index]
            or before_packages[removed_index + 1 :]
            != after_packages[activated_index + 1 :]
        ):
            raise ValueError(
                "A version replacement must not alter other active packages."
            )
        if self.before.compatibility_sha256 != self.after.compatibility_sha256:
            raise ValueError(
                "A version replacement must preserve compatibility identity."
            )


class ExpertRegistry:
    """Manage validated active packages without hidden model mutation.

    Installation revalidates live local files, binds the registry permanently
    to the first package's anchor and router-feature compatibility identity,
    and activates the package in a stable historical slot. Unloading removes
    the active descriptor but retains only its identity-to-slot ledger, so
    reinserting identical content restores the original ordering and registry
    hash.

    PKG-004 upgrades replace one active package with a different immutable
    version of the same specialization in its current slot. Rollback requires
    the exact upgrade record and old package descriptor, and is accepted only
    while the registry still equals the upgrade's resulting snapshot.

    Registry hashes cover ordered portable package identities, not local paths:

    ``SHA256(CanonicalJSON({format, packages}))``.

    All state transitions are protected by a re-entrant lock and validate every
    failure condition before mutating active state. PKG-002 does not load
    tensors, attach modules, recalibrate routers, upgrade versions, or persist
    registry state.
    """

    def __init__(self) -> None:
        """Create an empty registry with no established compatibility binding."""

        self._lock = RLock()
        self._active: dict[tuple[str, str], ExpertPackage] = {}
        self._slots: list[tuple[str, str]] = []
        self._known_manifest_hashes: dict[tuple[str, str], str] = {}
        self._compatibility_sha256: str | None = None

    @classmethod
    def from_snapshot(
        cls,
        packages: Sequence[ExpertPackage],
        *,
        registry_format_version: int,
        registry_sha256: str,
        compatibility_sha256: str | None,
    ) -> ExpertRegistry:
        """Reconstruct the exact active state of a persisted registry snapshot.

        CORE-006 checkpoints retain active package order and the permanent
        compatibility binding. Historical inactive slots are deliberately not
        retained because their package content is absent from the checkpoint.

        Every package is strictly revalidated through ordinary installation
        before the reconstructed snapshot is compared with the persisted one.
        """

        if isinstance(packages, (str, bytes)) or not isinstance(
            packages,
            Sequence,
        ):
            raise TypeError("packages must be a sequence of ExpertPackage.")
        normalized = tuple(packages)
        if not all(isinstance(package, ExpertPackage) for package in normalized):
            raise TypeError("packages must contain only ExpertPackage values.")
        expected = RegistrySnapshot(
            registry_format_version=registry_format_version,
            registry_sha256=registry_sha256,
            compatibility_sha256=compatibility_sha256,
            active_packages=tuple(package.identity for package in normalized),
        )

        registry = cls()
        for package in normalized:
            registry.install(package)
        if not normalized and compatibility_sha256 is not None:
            # An empty active registry can still remain permanently bound after
            # unloading its final package. No tensor or package identity is
            # synthesized; only the recorded compatibility constraint returns.
            with registry._lock:
                registry._compatibility_sha256 = compatibility_sha256
        if registry.snapshot() != expected:
            raise RegistryLifecycleError(
                "Packages do not reconstruct the persisted registry snapshot."
            )
        return registry

    @property
    def active_packages(self) -> tuple[ExpertPackage, ...]:
        """Return active descriptors in stable historical slot order."""

        with self._lock:
            return tuple(
                self._active[key] for key in self._slots if key in self._active
            )

    @property
    def active_identities(self) -> tuple[PackageIdentity, ...]:
        """Return portable active identities in stable historical slot order."""

        return tuple(package.identity for package in self.active_packages)

    @property
    def active_registry_sha256(self) -> str:
        """Return the deterministic hash of ordered active package identities."""

        return self.snapshot().registry_sha256

    @property
    def compatibility_sha256(self) -> str | None:
        """Return the registry's permanent anchor/feature compatibility identity."""

        with self._lock:
            return self._compatibility_sha256

    def mixture_compatibility(
        self,
        *,
        layer_index: int,
    ) -> MixtureCompatibilityIndex:
        """Return PKG-003 metadata for one layer of the active registry.

        The active package snapshot is checked before and after construction
        so callers never receive compatibility metadata assembled across two
        registry states. Missing package group declarations remain represented
        as non-mixable ``None`` entries.
        """

        from anchored_moe.registry.compatibility import (
            MixtureCompatibilityIndex,
        )

        snapshot = self.snapshot()
        packages = self.active_packages
        index = MixtureCompatibilityIndex.from_packages(
            packages,
            layer_index=layer_index,
        )
        if self.snapshot() != snapshot:
            raise RegistryLifecycleError(
                "Registry changed while mixture compatibility was assembled."
            )
        return index

    def snapshot(self) -> RegistrySnapshot:
        """Return an immutable, self-contained active-registry snapshot."""

        with self._lock:
            active_identities = tuple(
                self._active[key].identity
                for key in self._slots
                if key in self._active
            )
            registry_sha256 = _registry_sha256(active_identities)
            return RegistrySnapshot(
                registry_format_version=REGISTRY_FORMAT_VERSION,
                registry_sha256=registry_sha256,
                compatibility_sha256=self._compatibility_sha256,
                active_packages=active_identities,
            )

    def install(self, package: ExpertPackage) -> RegistryChange:
        """Validate and atomically activate one immutable package version.

        Args:
            package: Strictly validated local descriptor to activate.

        Returns:
            Before/after snapshots and the installed package identity.

        Raises:
            TypeError: If ``package`` is not an :class:`ExpertPackage`.
            RegistryLifecycleError:
                If validation was non-strict, the version is active, known
                content changed, or anchor/router features are incompatible.
            PackageValidationError:
                If live package files changed or fail payload validation.
        """

        if not isinstance(package, ExpertPackage):
            raise TypeError("package must be an ExpertPackage.")
        if not package.strict_validation:
            raise RegistryLifecycleError(
                "Registry installation requires strict package validation."
            )
        # Potentially expensive I/O happens before locking. No registry state is
        # exposed or changed until every payload and live manifest is verified.
        package.validate_live_files()
        identity = package.identity
        version_key = identity.version_key
        compatibility_sha256 = _package_compatibility_sha256(package)

        with self._lock:
            before = self.snapshot()
            if version_key in self._active:
                raise RegistryLifecycleError(
                    "Package version is already active: "
                    f"{identity.specialization_name}@"
                    f"{identity.specialization_version}."
                )
            known_manifest = self._known_manifest_hashes.get(version_key)
            if (
                known_manifest is not None
                and known_manifest != identity.manifest_sha256
            ):
                raise RegistryLifecycleError(
                    "Package content changed for an existing immutable "
                    "specialization version."
                )
            if (
                self._compatibility_sha256 is not None
                and self._compatibility_sha256 != compatibility_sha256
            ):
                raise RegistryLifecycleError(
                    "Package anchor or router feature space is incompatible "
                    "with this registry."
                )

            # Every check has completed. The following assignments are the
            # complete atomic in-memory transition.
            if version_key not in self._known_manifest_hashes:
                self._slots.append(version_key)
                self._known_manifest_hashes[version_key] = (
                    identity.manifest_sha256
                )
            elif version_key not in self._slots:
                # A successful upgrade transfers the superseded slot to the new
                # version. A later ordinary installation of that inactive old
                # version is a new activation, not an implicit rollback.
                self._slots.append(version_key)
            if self._compatibility_sha256 is None:
                self._compatibility_sha256 = compatibility_sha256
            self._active[version_key] = package
            after = self.snapshot()
            return RegistryChange(
                operation="install",
                package=identity,
                before=before,
                after=after,
            )

    def unload(self, package: PackageIdentity) -> RegistryChange:
        """Atomically deactivate one exact package identity.

        The historical slot and manifest digest remain in the registry ledger
        so reinsertion of identical content restores order. The active
        descriptor, including its filesystem path, is released.
        """

        if not isinstance(package, PackageIdentity):
            raise TypeError("package must be a PackageIdentity.")
        version_key = package.version_key
        with self._lock:
            active = self._active.get(version_key)
            if active is None:
                raise RegistryLifecycleError(
                    "Package version is not active: "
                    f"{package.specialization_name}@"
                    f"{package.specialization_version}."
                )
            if active.identity != package:
                raise RegistryLifecycleError(
                    "Active package identity does not match the requested "
                    "manifest digest."
                )

            before = self.snapshot()
            del self._active[version_key]
            after = self.snapshot()
            return RegistryChange(
                operation="unload",
                package=package,
                before=before,
                after=after,
            )

    def upgrade(
        self,
        current: PackageIdentity,
        replacement: ExpertPackage,
    ) -> RegistryUpgradeChange:
        """Atomically replace one active package with a new immutable version.

        The replacement must belong to the same specialization, use a
        different version, pass strict live-file validation, and match the
        registry's permanent anchor/router-feature compatibility identity. It
        occupies the current package's exact slot, so unrelated package order
        remains stable.

        The operation changes registry state only. Callers must explicitly
        synchronize any live model after a successful upgrade.

        Args:
            current: Exact active package identity being superseded.
            replacement: Strictly validated replacement package descriptor.

        Returns:
            An immutable before/after audit record suitable for exact rollback.

        Raises:
            TypeError: If an argument has the wrong public type.
            RegistryLifecycleError:
                If the current package is stale, the replacement is invalid for
                an upgrade, or the transition would violate registry invariants.
            PackageValidationError:
                If replacement files changed or fail payload validation.
        """

        self._validate_upgrade_arguments(
            current=current,
            replacement=replacement,
        )
        # Hashing every live payload occurs before the lock and before any state
        # mutation, matching ordinary installation's validation boundary.
        replacement.validate_live_files()
        with self._lock:
            return self._replace_active_version(
                operation="upgrade",
                current=current,
                replacement=replacement,
            )

    def rollback(
        self,
        upgrade: RegistryUpgradeChange,
        previous_package: ExpertPackage,
    ) -> RegistryUpgradeChange:
        """Reverse one exact upgrade while its resulting snapshot is current.

        Rollback is intentionally stale-safe: any intervening registry change,
        including an unrelated installation, makes the supplied audit record
        ineligible. The caller retains the old immutable package on disk and
        supplies its strictly validated descriptor here; the registry does not
        retain hidden filesystem references to inactive packages.

        Args:
            upgrade: Successful ``operation="upgrade"`` record to reverse.
            previous_package: Exact package removed by that upgrade.

        Returns:
            A rollback audit record whose replacement direction is reversed.

        Raises:
            TypeError: If an argument has the wrong public type.
            RegistryLifecycleError:
                If the record is not an upgrade, is stale, does not match the
                previous package, or cannot be reversed exactly.
            PackageValidationError:
                If previous package files changed or fail payload validation.
        """

        if not isinstance(upgrade, RegistryUpgradeChange):
            raise TypeError("upgrade must be a RegistryUpgradeChange.")
        if not isinstance(previous_package, ExpertPackage):
            raise TypeError("previous_package must be an ExpertPackage.")
        if upgrade.operation != "upgrade":
            raise RegistryLifecycleError(
                "Only an upgrade record can be rolled back."
            )
        if not previous_package.strict_validation:
            raise RegistryLifecycleError(
                "Registry rollback requires strict package validation."
            )
        if previous_package.identity != upgrade.removed_package:
            raise RegistryLifecycleError(
                "previous_package does not match the package removed by upgrade."
            )
        previous_package.validate_live_files()

        with self._lock:
            if self.snapshot() != upgrade.after:
                raise RegistryLifecycleError(
                    "Upgrade rollback is stale because registry state changed."
                )
            return self._replace_active_version(
                operation="rollback",
                current=upgrade.activated_package,
                replacement=previous_package,
            )

    def _validate_upgrade_arguments(
        self,
        *,
        current: PackageIdentity,
        replacement: ExpertPackage,
    ) -> None:
        """Validate public upgrade arguments before live package I/O."""

        if not isinstance(current, PackageIdentity):
            raise TypeError("current must be a PackageIdentity.")
        if not isinstance(replacement, ExpertPackage):
            raise TypeError("replacement must be an ExpertPackage.")
        if not replacement.strict_validation:
            raise RegistryLifecycleError(
                "Registry upgrade requires strict package validation."
            )
        replacement_identity = replacement.identity
        if (
            current.specialization_name
            != replacement_identity.specialization_name
        ):
            raise RegistryLifecycleError(
                "Replacement package must use the same specialization_name."
            )
        if (
            current.specialization_version
            == replacement_identity.specialization_version
        ):
            raise RegistryLifecycleError(
                "Replacement package must use a different specialization_version."
            )

    def _replace_active_version(
        self,
        *,
        operation: str,
        current: PackageIdentity,
        replacement: ExpertPackage,
    ) -> RegistryUpgradeChange:
        """Validate and apply one slot-preserving replacement under the lock."""

        current_key = current.version_key
        replacement_identity = replacement.identity
        replacement_key = replacement_identity.version_key
        active = self._active.get(current_key)
        if active is None:
            raise RegistryLifecycleError(
                "Current package version is not active: "
                f"{current.specialization_name}@"
                f"{current.specialization_version}."
            )
        if active.identity != current:
            raise RegistryLifecycleError(
                "Active package identity does not match the requested "
                "manifest digest."
            )
        if replacement_key in self._active:
            raise RegistryLifecycleError(
                "Replacement package version is already active: "
                f"{replacement_identity.specialization_name}@"
                f"{replacement_identity.specialization_version}."
            )
        same_specialization = tuple(
            package.identity
            for key, package in self._active.items()
            if key != current_key
            and package.identity.specialization_name
            == current.specialization_name
        )
        if same_specialization:
            raise RegistryLifecycleError(
                "Atomic upgrade requires exactly one active version of the "
                f"specialization; also active: {same_specialization!r}."
            )

        known_manifest = self._known_manifest_hashes.get(replacement_key)
        if (
            known_manifest is not None
            and known_manifest != replacement_identity.manifest_sha256
        ):
            raise RegistryLifecycleError(
                "Replacement content changed for an existing immutable "
                "specialization version."
            )
        if replacement_key in self._slots:
            raise RegistryLifecycleError(
                "Replacement version already owns a historical registry slot."
            )
        compatibility_sha256 = _package_compatibility_sha256(replacement)
        if self._compatibility_sha256 != compatibility_sha256:
            raise RegistryLifecycleError(
                "Replacement package anchor or router feature space is "
                "incompatible with this registry."
            )

        before = self.snapshot()
        slot_index = self._slots.index(current_key)
        candidate_identities = list(before.active_packages)
        active_index = candidate_identities.index(current)
        candidate_identities[active_index] = replacement_identity
        after = RegistrySnapshot(
            registry_format_version=REGISTRY_FORMAT_VERSION,
            registry_sha256=_registry_sha256(tuple(candidate_identities)),
            compatibility_sha256=self._compatibility_sha256,
            active_packages=tuple(candidate_identities),
        )
        change = RegistryUpgradeChange(
            operation=operation,
            removed_package=current,
            activated_package=replacement_identity,
            before=before,
            after=after,
        )

        # Every fallible validation and audit-record construction has completed.
        # Readers cannot observe these assignments independently while the lock
        # is held, making this the complete atomic in-memory transition.
        del self._active[current_key]
        self._slots[slot_index] = replacement_key
        if known_manifest is None:
            self._known_manifest_hashes[replacement_key] = (
                replacement_identity.manifest_sha256
            )
        self._active[replacement_key] = replacement
        return change


def _registry_sha256(
    active_packages: tuple[PackageIdentity, ...],
) -> str:
    """Hash only portable ordered active-package identities."""

    return sha256_json(
        {
            "registry_format_version": REGISTRY_FORMAT_VERSION,
            "active_packages": [
                identity.to_dict() for identity in active_packages
            ],
        }
    )


def _package_compatibility_sha256(package: ExpertPackage) -> str:
    """Hash exact anchor, converted-layer, and router-feature compatibility."""

    manifest = package.manifest
    return sha256_json(
        {
            "base_model_repository": manifest.base_model_repository,
            "base_model_revision": manifest.base_model_revision,
            "base_config_sha256": manifest.base_config_sha256,
            "base_weight_sha256": dict(manifest.base_weight_sha256),
            "base_model_dtype": manifest.base_model_dtype,
            "tokenizer_repository": manifest.tokenizer_repository,
            "tokenizer_revision": manifest.tokenizer_revision,
            "converted_layers": list(manifest.converted_layers),
            "router_feature_specification": manifest.to_dict()[
                "router_feature_specification"
            ],
            "router_projection_hash": manifest.router_projection_hash,
        }
    )
