"""Atomic in-memory lifecycle for validated expert packages."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.utils.hashing import sha256_json


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


class ExpertRegistry:
    """Manage validated active packages without hidden model mutation.

    Installation revalidates live local files, binds the registry permanently
    to the first package's anchor and router-feature compatibility identity,
    and activates the package in a stable historical slot. Unloading removes
    the active descriptor but retains only its identity-to-slot ledger, so
    reinserting identical content restores the original ordering and registry
    hash.

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
