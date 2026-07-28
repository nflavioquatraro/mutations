"""Strictly validated local expert-package descriptors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anchored_moe.registry.manifest import (
    DEFAULT_MANIFEST_FILENAME,
    ExpertPackageManifest,
    PackageValidationError,
    PackageValidationReport,
)
from anchored_moe.registry.bindings import (
    ExpertPayloadBinding,
    parse_expert_payload_bindings,
)

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HUB_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, order=True)
class PackageIdentity:
    """Portable immutable identity of one package version.

    Attributes:
        specialization_name: Human-readable specialization namespace.
        specialization_version: Immutable specialization release version.
        manifest_sha256: Canonical manifest content digest.

    Filesystem paths are deliberately excluded so copying an identical package
    does not change its identity.
    """

    specialization_name: str
    specialization_version: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        """Validate public construction independently of a manifest."""

        for field_name in ("specialization_name", "specialization_version"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty string without "
                    "surrounding whitespace."
                )
        if (
            not isinstance(self.manifest_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError(
                "manifest_sha256 must be a lowercase 64-character digest."
            )

    @property
    def version_key(self) -> tuple[str, str]:
        """Return the stable specialization/version lookup key."""

        return self.specialization_name, self.specialization_version

    def to_dict(self) -> dict[str, str]:
        """Return the portable JSON representation used by registry hashes."""

        return {
            "specialization_name": self.specialization_name,
            "specialization_version": self.specialization_version,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class ExpertPackage:
    """Validated local package ready for registry installation.

    Args:
        manifest: Immutable version-1 package manifest.
        package_directory: Absolute resolved package root.
        validation_report: Result of strict payload validation.
        strict_validation: Whether untracked files were rejected.
        hub_repository: Optional source Hub repository for a downloaded package.
        hub_revision: Resolved immutable Hub commit for a downloaded package.

    Construct descriptors with :meth:`from_directory`; direct construction is
    validated but cannot prove that the report came from the supplied path.
    :class:`~anchored_moe.registry.ExpertRegistry` therefore revalidates the
    live directory immediately before every installation.

    The descriptor does not load tensors, execute package code, or attach
    experts to a model. Hub transport occurs only through the explicit class
    and instance methods documented below.
    """

    manifest: ExpertPackageManifest
    package_directory: Path
    validation_report: PackageValidationReport
    strict_validation: bool
    hub_repository: str | None = None
    hub_revision: str | None = None

    def __post_init__(self) -> None:
        """Validate descriptor types, paths, and manifest/report identity."""

        if not isinstance(self.manifest, ExpertPackageManifest):
            raise TypeError("manifest must be an ExpertPackageManifest.")
        if not isinstance(self.package_directory, Path):
            raise TypeError("package_directory must be a pathlib.Path.")
        if not self.package_directory.is_absolute():
            raise ValueError("package_directory must be absolute.")
        if self.package_directory.is_symlink():
            raise ValueError("package_directory must not be a symbolic link.")
        if not self.package_directory.is_dir():
            raise ValueError("package_directory must be an existing directory.")
        if self.package_directory.resolve() != self.package_directory:
            raise ValueError("package_directory must be a resolved path.")
        if not isinstance(self.validation_report, PackageValidationReport):
            raise TypeError(
                "validation_report must be a PackageValidationReport."
            )
        if not isinstance(self.strict_validation, bool):
            raise TypeError("strict_validation must be a boolean.")
        if (self.hub_repository is None) != (self.hub_revision is None):
            raise ValueError(
                "hub_repository and hub_revision must be set together."
            )
        if self.hub_repository is not None and (
            not isinstance(self.hub_repository, str)
            or not self.hub_repository
            or self.hub_repository != self.hub_repository.strip()
        ):
            raise ValueError(
                "hub_repository must be a non-empty string without "
                "surrounding whitespace."
            )
        if self.hub_revision is not None and (
            not isinstance(self.hub_revision, str)
            or _HUB_COMMIT_PATTERN.fullmatch(self.hub_revision) is None
        ):
            raise ValueError(
                "hub_revision must be a lowercase 40-character commit hash."
            )
        if (
            self.validation_report.manifest_sha256
            != self.manifest.manifest_sha256
        ):
            raise ValueError(
                "validation_report does not match the package manifest."
            )
        if self.validation_report.verified_files != tuple(
            self.manifest.package_hashes
        ):
            raise ValueError(
                "validation_report does not cover every manifest payload."
            )

    @classmethod
    def from_directory(
        cls,
        package_directory: str | Path,
        *,
        strict: bool = True,
    ) -> ExpertPackage:
        """Load and validate a non-symlink local package directory.

        Args:
            package_directory: Existing package root containing a JSON manifest.
            strict: Whether every non-manifest file must be hash-tracked.
        Returns:
            An immutable descriptor with resolved path and validation report.

        Raises:
            PackageValidationError:
                If paths, manifest data, payload hashes, formats, or strict
                directory closure are invalid.
        """

        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")
        package_root = Path(package_directory)
        if package_root.is_symlink():
            raise PackageValidationError(
                f"Package directory must not be a symbolic link: {package_root}"
            )
        if not package_root.is_dir():
            raise PackageValidationError(
                f"Package directory does not exist: {package_root}"
            )
        package_root = package_root.resolve()
        manifest_path = _resolve_manifest_path(
            package_root,
            manifest_filename=DEFAULT_MANIFEST_FILENAME,
        )
        manifest = ExpertPackageManifest.from_json(manifest_path)
        report = manifest.validate_package_files(
            package_root,
            strict=strict,
            manifest_filename=DEFAULT_MANIFEST_FILENAME,
        )
        return cls(
            manifest=manifest,
            package_directory=package_root,
            validation_report=report,
            strict_validation=strict,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        token: str | bool | None = None,
        local_files_only: bool = False,
        strict: bool = True,
        endpoint: str | None = None,
    ) -> ExpertPackage:
        """Resolve a strict local or Hugging Face Hub expert package.

        Local directories delegate to :meth:`from_directory` without network
        access. Hub repositories are resolved to one immutable commit, copied
        from the Hub's symlink-based transport cache into a package-only asset
        directory, and strictly revalidated before this method returns.

        Args:
            pretrained_model_name_or_path:
                Existing local package directory or Hub model repository ID.
            revision:
                Optional Hub branch, tag, pull-request ref, or commit hash.
            cache_dir:
                Optional Hugging Face cache root.
            force_download:
                Whether Hub transport should refresh cached files.
            token:
                Hugging Face authentication token or token-selection flag.
            local_files_only:
                Whether Hub resolution must use already cached files.
            strict:
                Whether local directory closure is required. Remote packages
                always require ``True``.
            endpoint:
                Optional Hugging Face-compatible Hub endpoint.

        Returns:
            A validated package whose ``hub_revision`` records the resolved
            immutable commit for remote sources.

        Raises:
            PackageValidationError:
                If local or downloaded package content is unsafe or invalid.
            PackageHubError:
                If Hub arguments or transport snapshot semantics are invalid.
            huggingface_hub.errors.HfHubError:
                If the Hub request or cache lookup fails.
        """

        # Import lazily so package validation remains usable in minimal local
        # environments and to avoid a module cycle with the Hub helper.
        from anchored_moe.registry.hub import resolve_pretrained_package

        return resolve_pretrained_package(
            cls,
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            strict=strict,
            endpoint=endpoint,
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | bool | None = None,
        private: bool | None = None,
        commit_message: str | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        exist_ok: bool = False,
        endpoint: str | None = None,
    ) -> CommitInfo:
        """Upload exactly this package's validated files to a Hub model repo.

        The live package is revalidated, copied into a private package-only
        staging directory, and revalidated again before any remote mutation.
        Only ``manifest.json`` and manifest-tracked payloads are offered to the
        Hub client. Authentication remains owned by ``huggingface_hub``.

        Args:
            repo_id: Destination model repository, such as ``"org/code-moe"``.
            token: Hugging Face authentication token or token-selection flag.
            private: Optional visibility used when creating the repository.
            commit_message: Optional non-empty Hub commit title.
            revision: Optional destination branch.
            create_pr: Whether to upload through a new Hub pull request.
            exist_ok: Whether an already existing repository may be used.
            endpoint: Optional Hugging Face-compatible Hub endpoint.

        Returns:
            The ``huggingface_hub.CommitInfo`` returned by ``upload_folder``.

        Raises:
            PackageValidationError:
                If package content changed or no longer validates strictly.
            PackageHubError:
                If upload arguments or descriptor provenance are unsafe.
            huggingface_hub.errors.HfHubError:
                If repository creation or upload fails.

        Remote repository creation and upload are explicit external side
        effects. A failed upload may leave a newly created empty repository;
        package and registry state remain unchanged.
        """

        from anchored_moe.registry.hub import upload_pretrained_package

        return upload_pretrained_package(
            self,
            repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
            revision=revision,
            create_pr=create_pr,
            exist_ok=exist_ok,
            endpoint=endpoint,
        )

    @property
    def identity(self) -> PackageIdentity:
        """Return the portable immutable identity of this package."""

        return PackageIdentity(
            specialization_name=self.manifest.specialization_name,
            specialization_version=self.manifest.specialization_version,
            manifest_sha256=self.manifest.manifest_sha256,
        )

    @property
    def payload_bindings(self) -> tuple[ExpertPayloadBinding, ...]:
        """Return validated runtime bindings, requiring an attachable package."""

        return parse_expert_payload_bindings(self.manifest, require=True)

    def validate_live_files(self) -> PackageValidationReport:
        """Revalidate on-disk manifest identity and every tracked payload.

        The registry calls this immediately before installation so tampering
        after descriptor construction cannot activate stale metadata.
        """

        manifest_path = _resolve_manifest_path(
            self.package_directory,
            manifest_filename=DEFAULT_MANIFEST_FILENAME,
        )
        live_manifest = ExpertPackageManifest.from_json(manifest_path)
        if live_manifest.manifest_sha256 != self.identity.manifest_sha256:
            raise PackageValidationError(
                "Package manifest changed after descriptor validation."
            )
        return live_manifest.validate_package_files(
            self.package_directory,
            strict=self.strict_validation,
            manifest_filename=DEFAULT_MANIFEST_FILENAME,
        )


def _resolve_manifest_path(
    package_root: Path,
    *,
    manifest_filename: str,
) -> Path:
    """Resolve the one regular package-root manifest accepted by PKG-002."""

    if (
        not isinstance(manifest_filename, str)
        or not manifest_filename
        or manifest_filename != Path(manifest_filename).name
    ):
        raise PackageValidationError(
            "manifest_filename must be one package-root filename."
        )
    manifest_path = package_root / manifest_filename
    if manifest_path.is_symlink():
        raise PackageValidationError(
            f"Package manifest must not be a symbolic link: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise PackageValidationError(
            f"Package manifest is not a regular file: {manifest_path}"
        )
    return manifest_path
