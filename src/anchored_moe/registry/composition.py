"""Immutable local composition checkpoints for anchored models.

CORE-006 persists model composition rather than duplicating the immutable
anchor. A checkpoint records the exact anchor reference and configuration
digest, copies active expert packages byte-for-byte, and binds their stable
order to the active registry hash. All files are validated before loading or
publishing the directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from transformers import GenerationConfig

from anchored_moe.configuration import AnchoredMoEConfig
from anchored_moe.registry.manifest import PackageValidationError
from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.registry.registry import (
    REGISTRY_FORMAT_VERSION,
    ExpertRegistry,
    RegistrySnapshot,
)
from anchored_moe.utils.hashing import canonical_json_bytes, sha256_file, sha256_json


COMPOSITION_FORMAT_VERSION = 1
COMPOSITION_MANIFEST_FILENAME = "anchored_moe_composition.json"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CompositionCheckpointError(RuntimeError):
    """Raised when a local composition checkpoint is invalid or unsafe."""


@dataclass(frozen=True)
class CompositionPackageEntry:
    """Bind one active package identity to its checkpoint-relative directory."""

    relative_path: str
    identity: PackageIdentity

    def __post_init__(self) -> None:
        """Validate a canonical package directory and immutable identity."""

        if not isinstance(self.relative_path, str):
            raise TypeError("relative_path must be a string.")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or "\\" in self.relative_path
            or path.is_absolute()
            or self.relative_path != path.as_posix()
            or len(path.parts) != 2
            or path.parts[0] != "packages"
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in self.relative_path)
        ):
            raise ValueError(
                "relative_path must be a canonical packages/<directory> path."
            )
        if not isinstance(self.identity, PackageIdentity):
            raise TypeError("identity must be a PackageIdentity.")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible package entry."""

        return {
            "relative_path": self.relative_path,
            "identity": self.identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CompositionPackageEntry:
        """Parse one strict package entry from JSON-compatible data."""

        if not isinstance(value, dict) or set(value) != {
            "relative_path",
            "identity",
        }:
            raise CompositionCheckpointError(
                "Composition package entries require relative_path and identity."
            )
        identity = value["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "specialization_name",
            "specialization_version",
            "manifest_sha256",
        }:
            raise CompositionCheckpointError(
                "Composition package identity fields are incomplete."
            )
        try:
            return cls(
                relative_path=value["relative_path"],
                identity=PackageIdentity(**identity),
            )
        except (TypeError, ValueError) as error:
            raise CompositionCheckpointError(
                f"Invalid composition package entry: {error}"
            ) from error


@dataclass(frozen=True)
class AnchoredCompositionManifest:
    """Versioned, portable description of one saved anchored composition.

    The manifest deliberately excludes anchor tensors. ``base_model_name_or_path``
    and ``base_model_revision`` identify the external checkpoint, while
    ``base_config_sha256`` detects configuration drift. Package identities and
    the registry digest preserve deterministic activation order.
    """

    composition_format_version: int
    base_model_name_or_path: str
    base_model_revision: str | None
    base_config_sha256: str
    config_sha256: str
    generation_config_sha256: str | None
    registry_format_version: int
    registry_sha256: str
    compatibility_sha256: str | None
    routing_policy: str
    packages: tuple[CompositionPackageEntry, ...]

    def __post_init__(self) -> None:
        """Validate format, pins, hashes, package order, and registry identity."""

        if self.composition_format_version != COMPOSITION_FORMAT_VERSION:
            raise ValueError(
                "Unsupported composition_format_version: "
                f"{self.composition_format_version!r}."
            )
        if (
            not isinstance(self.base_model_name_or_path, str)
            or not self.base_model_name_or_path
            or self.base_model_name_or_path
            != self.base_model_name_or_path.strip()
        ):
            raise ValueError(
                "base_model_name_or_path must be a non-empty string without "
                "surrounding whitespace."
            )
        if self.base_model_revision is not None and (
            not isinstance(self.base_model_revision, str)
            or not self.base_model_revision
            or self.base_model_revision != self.base_model_revision.strip()
        ):
            raise ValueError(
                "base_model_revision must be None or a non-empty string "
                "without surrounding whitespace."
            )
        _validate_sha256(self.base_config_sha256, "base_config_sha256")
        _validate_sha256(self.config_sha256, "config_sha256")
        if self.generation_config_sha256 is not None:
            _validate_sha256(
                self.generation_config_sha256,
                "generation_config_sha256",
            )
        if self.registry_format_version != REGISTRY_FORMAT_VERSION:
            raise ValueError(
                "Unsupported registry_format_version in composition manifest."
            )
        _validate_sha256(self.registry_sha256, "registry_sha256")
        if self.compatibility_sha256 is not None:
            _validate_sha256(self.compatibility_sha256, "compatibility_sha256")
        if self.routing_policy != "anchored_top1":
            raise ValueError(
                "CORE-006 supports only routing_policy='anchored_top1'."
            )
        if not isinstance(self.packages, tuple) or not all(
            isinstance(entry, CompositionPackageEntry) for entry in self.packages
        ):
            raise TypeError(
                "packages must be a tuple of CompositionPackageEntry values."
            )
        relative_paths = tuple(entry.relative_path for entry in self.packages)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("Composition package paths must be unique.")

        identities = tuple(entry.identity for entry in self.packages)
        # RegistrySnapshot owns the canonical ordered-registry hash contract.
        RegistrySnapshot(
            registry_format_version=self.registry_format_version,
            registry_sha256=self.registry_sha256,
            compatibility_sha256=self.compatibility_sha256,
            active_packages=identities,
        )

    @property
    def manifest_sha256(self) -> str:
        """Return the digest of the canonical composition-manifest payload."""

        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible manifest representation."""

        return {
            "composition_format_version": self.composition_format_version,
            "base_model_name_or_path": self.base_model_name_or_path,
            "base_model_revision": self.base_model_revision,
            "base_config_sha256": self.base_config_sha256,
            "config_sha256": self.config_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "registry_format_version": self.registry_format_version,
            "registry_sha256": self.registry_sha256,
            "compatibility_sha256": self.compatibility_sha256,
            "routing_policy": self.routing_policy,
            "packages": [entry.to_dict() for entry in self.packages],
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the manifest as canonical UTF-8 JSON with a final newline."""

        return canonical_json_bytes(self.to_dict()) + b"\n"

    def save_json(self, path: str | Path) -> None:
        """Write the manifest atomically inside an existing staging directory."""

        destination = Path(path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Manifest parent directory does not exist: {destination.parent}"
            )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(self.to_json_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def from_json(cls, path: str | Path) -> AnchoredCompositionManifest:
        """Load one strict manifest without validating its surrounding files."""

        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CompositionCheckpointError(
                f"Cannot read composition manifest: {source}"
            ) from error
        required = {
            "composition_format_version",
            "base_model_name_or_path",
            "base_model_revision",
            "base_config_sha256",
            "config_sha256",
            "generation_config_sha256",
            "registry_format_version",
            "registry_sha256",
            "compatibility_sha256",
            "routing_policy",
            "packages",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise CompositionCheckpointError(
                "Composition manifest fields are incomplete or unsupported."
            )
        packages = value["packages"]
        if not isinstance(packages, list):
            raise CompositionCheckpointError(
                "Composition manifest packages must be a list."
            )
        try:
            return cls(
                composition_format_version=value["composition_format_version"],
                base_model_name_or_path=value["base_model_name_or_path"],
                base_model_revision=value["base_model_revision"],
                base_config_sha256=value["base_config_sha256"],
                config_sha256=value["config_sha256"],
                generation_config_sha256=value[
                    "generation_config_sha256"
                ],
                registry_format_version=value["registry_format_version"],
                registry_sha256=value["registry_sha256"],
                compatibility_sha256=value["compatibility_sha256"],
                routing_policy=value["routing_policy"],
                packages=tuple(
                    CompositionPackageEntry.from_dict(entry)
                    for entry in packages
                ),
            )
        except (TypeError, ValueError) as error:
            raise CompositionCheckpointError(
                f"Invalid composition manifest: {error}"
            ) from error


@dataclass(frozen=True)
class LoadedCompositionCheckpoint:
    """Fully validated local checkpoint metadata and package descriptors."""

    manifest: AnchoredCompositionManifest
    config: AnchoredMoEConfig
    generation_config: GenerationConfig | None
    packages: tuple[ExpertPackage, ...]


def export_composition_checkpoint(
    checkpoint_directory: str | Path,
    *,
    config: AnchoredMoEConfig,
    generation_config: GenerationConfig | None,
    registry: ExpertRegistry,
) -> AnchoredCompositionManifest:
    """Atomically publish one immutable local anchored composition.

    The external anchor is not copied. Every active package is revalidated,
    copied into a private sibling staging directory, strictly reloaded, and
    compared with its source identity before publication. Existing destination
    paths are never replaced.
    """

    if not isinstance(config, AnchoredMoEConfig):
        raise TypeError("config must be an AnchoredMoEConfig.")
    if generation_config is not None and not isinstance(
        generation_config,
        GenerationConfig,
    ):
        raise TypeError("generation_config must be GenerationConfig or None.")
    if not isinstance(registry, ExpertRegistry):
        raise TypeError("registry must be an ExpertRegistry.")
    if not config.base_model_name_or_path:
        raise CompositionCheckpointError(
            "Composition export requires config.base_model_name_or_path."
        )
    if not config.base_config_sha256:
        raise CompositionCheckpointError(
            "Composition export requires config.base_config_sha256."
        )

    destination = _validate_new_checkpoint_directory(checkpoint_directory)
    packages = registry.active_packages
    snapshot = registry.snapshot()
    if tuple(package.identity for package in packages) != snapshot.active_packages:
        raise CompositionCheckpointError(
            "Registry changed while composition state was captured."
        )
    for package in packages:
        try:
            package.validate_live_files()
        except PackageValidationError as error:
            raise CompositionCheckpointError(
                "Active package failed live validation before composition "
                f"export: {package.identity.version_key!r}."
            ) from error
        if package.manifest.base_config_sha256 != config.base_config_sha256:
            raise CompositionCheckpointError(
                "Active package base_config_sha256 does not match the anchored "
                f"configuration: {package.identity.version_key!r}."
            )

    temporary_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    published = False
    try:
        config.save_pretrained(temporary_root)
        config_path = temporary_root / CONFIG_FILENAME
        generation_digest: str | None = None
        if generation_config is not None:
            generation_config.save_pretrained(temporary_root)
            generation_path = temporary_root / GENERATION_CONFIG_FILENAME
            if not generation_path.is_file():
                raise CompositionCheckpointError(
                    "Generation configuration was not safely serialized."
                )
            generation_digest = sha256_file(generation_path)

        entries: list[CompositionPackageEntry] = []
        for slot, package in enumerate(packages):
            relative_path = (
                f"packages/{slot:04d}-{package.identity.manifest_sha256}"
            )
            target = temporary_root.joinpath(
                *PurePosixPath(relative_path).parts
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package.package_directory, target)
            copied_package = ExpertPackage.from_directory(target, strict=True)
            if copied_package.identity != package.identity:
                raise CompositionCheckpointError(
                    "Copied package identity changed during composition export: "
                    f"{package.identity.version_key!r}."
                )
            entries.append(
                CompositionPackageEntry(
                    relative_path=relative_path,
                    identity=package.identity,
                )
            )

        manifest = AnchoredCompositionManifest(
            composition_format_version=COMPOSITION_FORMAT_VERSION,
            base_model_name_or_path=config.base_model_name_or_path,
            base_model_revision=config.base_model_revision,
            base_config_sha256=config.base_config_sha256,
            config_sha256=sha256_file(config_path),
            generation_config_sha256=generation_digest,
            registry_format_version=snapshot.registry_format_version,
            registry_sha256=snapshot.registry_sha256,
            compatibility_sha256=snapshot.compatibility_sha256,
            routing_policy="anchored_top1",
            packages=tuple(entries),
        )
        manifest.save_json(
            temporary_root / COMPOSITION_MANIFEST_FILENAME
        )
        validated = load_composition_checkpoint(temporary_root)
        if validated.manifest != manifest:
            raise CompositionCheckpointError(
                "Published composition metadata changed during strict reload."
            )
        if destination.exists():
            raise FileExistsError(
                f"Composition destination already exists: {destination}"
            )
        os.rename(temporary_root, destination)
        published = True
        return manifest
    except (FileExistsError, CompositionCheckpointError):
        raise
    except Exception as error:
        raise CompositionCheckpointError(
            f"Cannot export composition checkpoint at {destination}: {error}"
        ) from error
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


def load_composition_checkpoint(
    checkpoint_directory: str | Path,
) -> LoadedCompositionCheckpoint:
    """Strictly validate and load local composition metadata and packages."""

    root = _resolve_checkpoint_directory(checkpoint_directory)
    manifest_path = root / COMPOSITION_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CompositionCheckpointError(
            f"Composition manifest is missing or unsafe: {manifest_path}"
        )
    manifest = AnchoredCompositionManifest.from_json(manifest_path)

    config_path = root / CONFIG_FILENAME
    _validate_tracked_file(
        config_path,
        expected_sha256=manifest.config_sha256,
        field_name=CONFIG_FILENAME,
    )
    try:
        config = AnchoredMoEConfig.from_json_file(config_path)
    except Exception as error:
        raise CompositionCheckpointError(
            f"Cannot load anchored configuration: {config_path}"
        ) from error
    if (
        config.base_model_name_or_path
        != manifest.base_model_name_or_path
        or config.base_model_revision != manifest.base_model_revision
        or config.base_config_sha256 != manifest.base_config_sha256
    ):
        raise CompositionCheckpointError(
            "Anchored configuration pins do not match the composition manifest."
        )

    generation_config: GenerationConfig | None = None
    generation_path = root / GENERATION_CONFIG_FILENAME
    if manifest.generation_config_sha256 is None:
        if generation_path.exists():
            raise CompositionCheckpointError(
                "Untracked generation_config.json is not allowed."
            )
    else:
        _validate_tracked_file(
            generation_path,
            expected_sha256=manifest.generation_config_sha256,
            field_name=GENERATION_CONFIG_FILENAME,
        )
        try:
            generation_config = GenerationConfig.from_pretrained(
                root,
                local_files_only=True,
            )
        except Exception as error:
            raise CompositionCheckpointError(
                f"Cannot load generation configuration: {generation_path}"
            ) from error

    packages: list[ExpertPackage] = []
    expected_files = {
        COMPOSITION_MANIFEST_FILENAME,
        CONFIG_FILENAME,
    }
    if manifest.generation_config_sha256 is not None:
        expected_files.add(GENERATION_CONFIG_FILENAME)
    for entry in manifest.packages:
        package_root = root.joinpath(*PurePosixPath(entry.relative_path).parts)
        try:
            package = ExpertPackage.from_directory(package_root, strict=True)
        except Exception as error:
            raise CompositionCheckpointError(
                "Cannot load bundled package "
                f"{entry.identity.version_key!r}: {error}"
            ) from error
        if package.identity != entry.identity:
            raise CompositionCheckpointError(
                "Bundled package identity does not match the composition "
                f"manifest: {entry.identity.version_key!r}."
            )
        if (
            package.manifest.base_config_sha256
            != manifest.base_config_sha256
        ):
            raise CompositionCheckpointError(
                "Bundled package base configuration does not match the "
                f"composition: {entry.identity.version_key!r}."
            )
        packages.append(package)
        expected_files.update(
            candidate.relative_to(root).as_posix()
            for candidate in package_root.rglob("*")
            if candidate.is_file()
        )

    actual_files: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise CompositionCheckpointError(
                "Symbolic links are not allowed in composition checkpoints: "
                f"{candidate.relative_to(root).as_posix()}"
            )
        if candidate.is_file():
            actual_files.add(candidate.relative_to(root).as_posix())
    untracked = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if untracked or missing:
        raise CompositionCheckpointError(
            "Composition directory closure failed: "
            f"untracked={untracked}, missing={missing}."
        )

    try:
        ExpertRegistry.from_snapshot(
            packages,
            registry_format_version=manifest.registry_format_version,
            registry_sha256=manifest.registry_sha256,
            compatibility_sha256=manifest.compatibility_sha256,
        )
    except Exception as error:
        raise CompositionCheckpointError(
            "Bundled packages do not reconstruct the recorded registry."
        ) from error
    return LoadedCompositionCheckpoint(
        manifest=manifest,
        config=config,
        generation_config=generation_config,
        packages=tuple(packages),
    )


def _validate_new_checkpoint_directory(value: str | Path) -> Path:
    """Return an absolute new destination with an existing resolved parent."""

    destination = Path(value)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Composition destination already exists: {destination}"
        )
    parent = destination.parent.resolve()
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Composition destination parent does not exist: {parent}"
        )
    return parent / destination.name


def _resolve_checkpoint_directory(value: str | Path) -> Path:
    """Resolve one existing non-symlink local checkpoint directory."""

    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise CompositionCheckpointError(
            f"Composition checkpoint must be a local non-symlink directory: {root}"
        )
    return root.resolve()


def _validate_tracked_file(
    path: Path,
    *,
    expected_sha256: str,
    field_name: str,
) -> None:
    """Validate one regular non-symlink file against its recorded digest."""

    if path.is_symlink() or not path.is_file():
        raise CompositionCheckpointError(
            f"Tracked composition file is missing or unsafe: {field_name}."
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise CompositionCheckpointError(
            f"SHA-256 mismatch for composition file: {field_name}."
        )


def _validate_sha256(value: Any, field_name: str) -> None:
    """Validate one lowercase SHA-256 string."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
