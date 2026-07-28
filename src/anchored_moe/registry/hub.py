"""Strict Hugging Face Hub transport for immutable expert packages.

PKG-006 treats Hub operations as transport around the existing PKG-001 local
format. A remote branch or tag is first resolved to one immutable commit. The
manifest is parsed before payload transfer, so only manifest-declared files
are fetched. Hub cache symlinks are copied into a package-only asset directory
and the ordinary strict local validator remains the final trust boundary.

This module implements framework infrastructure rather than a paper-derived
algorithm.

References:
    Hugging Face Hub download API.
    https://huggingface.co/docs/huggingface_hub/package_reference/file_download

    Hugging Face Hub upload API.
    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api

Differences from the referenced APIs:
    The generic Hub client permits partial folders and unrelated repository
    files. PKG-006 resolves a manifest first, transfers only its declared
    package closure, stages regular files outside the symlink cache, and
    requires the existing strict package validator before use or upload.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, cached_assets_path, snapshot_download
from huggingface_hub.utils import validate_repo_id

from anchored_moe.registry.manifest import (
    DEFAULT_MANIFEST_FILENAME,
    ExpertPackageManifest,
    PackageValidationError,
)

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

    from anchored_moe.registry.package import ExpertPackage


_COMMIT_HASH_LENGTH = 40
_LIBRARY_NAME = "anchored-moe"


class PackageHubError(RuntimeError):
    """Report unsafe arguments or inconsistent Hub package snapshots."""


def resolve_pretrained_package(
    package_class: type[ExpertPackage],
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
    strict: bool,
    endpoint: str | None,
) -> ExpertPackage:
    """Resolve one local directory or immutable Hub package snapshot."""

    _validate_common_arguments(
        force_download=force_download,
        local_files_only=local_files_only,
        strict=strict,
        revision=revision,
        token=token,
        endpoint=endpoint,
    )
    source = _normalize_source(pretrained_model_name_or_path)
    local_path = Path(source)
    if local_path.is_dir():
        _reject_local_hub_options(
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            endpoint=endpoint,
        )
        return package_class.from_directory(local_path, strict=strict)
    if isinstance(pretrained_model_name_or_path, Path) or _looks_like_local_path(
        source
    ):
        raise PackageValidationError(
            f"Local package directory does not exist: {local_path}"
        )
    if not strict:
        raise PackageHubError("Hub expert packages require strict=True.")

    validate_repo_id(source)
    cache_path = None if cache_dir is None else Path(cache_dir)
    manifest_snapshot = _download_snapshot(
        repo_id=source,
        revision=revision,
        cache_dir=cache_path,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=(DEFAULT_MANIFEST_FILENAME,),
        endpoint=endpoint,
    )
    resolved_revision = _snapshot_commit_hash(manifest_snapshot)
    manifest_source = _resolve_declared_source(
        manifest_snapshot,
        DEFAULT_MANIFEST_FILENAME,
        permit_transport_symlinks=True,
    )
    manifest = ExpertPackageManifest.from_json(manifest_source)
    transfer_paths = (
        DEFAULT_MANIFEST_FILENAME,
        *tuple(manifest.package_hashes),
    )

    # Pin the second request to the immutable commit discovered by the first
    # request. A moving branch therefore cannot combine a manifest from one
    # revision with payloads from another.
    complete_snapshot = _download_snapshot(
        repo_id=source,
        revision=resolved_revision,
        cache_dir=cache_path,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=transfer_paths,
        endpoint=endpoint,
    )
    if _snapshot_commit_hash(complete_snapshot) != resolved_revision:
        raise PackageHubError(
            "Hub snapshot changed after its immutable revision was resolved."
        )

    destination = _package_asset_destination(
        repo_id=source,
        endpoint=endpoint,
        manifest_sha256=manifest.manifest_sha256,
        cache_dir=cache_path,
    )
    if destination.exists():
        return _load_hub_package(
            package_class,
            destination,
            repo_id=source,
            revision=resolved_revision,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        _copy_declared_files(
            complete_snapshot,
            staging_root,
            relative_paths=transfer_paths,
        )
        staged = package_class.from_directory(staging_root, strict=True)
        if staged.identity.manifest_sha256 != manifest.manifest_sha256:
            raise PackageHubError(
                "Staged Hub package manifest identity changed during transfer."
            )
        try:
            staging_root.rename(destination)
        except FileExistsError:
            # Another process may have published the same immutable package
            # after our initial cache check. Validate rather than overwrite it.
            pass
        return _load_hub_package(
            package_class,
            destination,
            repo_id=source,
            revision=resolved_revision,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def upload_pretrained_package(
    package: ExpertPackage,
    repo_id: str,
    *,
    token: str | bool | None,
    private: bool | None,
    commit_message: str | None,
    revision: str | None,
    create_pr: bool,
    exist_ok: bool,
    endpoint: str | None,
) -> CommitInfo:
    """Stage and upload exactly one strict package through ``HfApi``."""

    from anchored_moe.registry.package import ExpertPackage

    if not isinstance(package, ExpertPackage):
        raise TypeError("package must be an ExpertPackage.")
    _validate_upload_arguments(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
        revision=revision,
        create_pr=create_pr,
        exist_ok=exist_ok,
        endpoint=endpoint,
    )
    if not package.strict_validation:
        raise PackageHubError(
            "Hub upload requires an ExpertPackage created with strict validation."
        )
    package.validate_live_files()
    transfer_paths = (
        DEFAULT_MANIFEST_FILENAME,
        *tuple(package.manifest.package_hashes),
    )
    resolved_message = commit_message or (
        "Upload anchored expert package "
        f"{package.identity.specialization_name}@"
        f"{package.identity.specialization_version}"
    )

    # Upload from a second validated copy. This prevents a concurrent edit to
    # the caller's directory from changing bytes after the live-file check.
    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-hub-upload-"
    ) as temporary:
        staging_root = Path(temporary)
        _copy_declared_files(
            package.package_directory,
            staging_root,
            relative_paths=transfer_paths,
            permit_transport_symlinks=False,
        )
        staged = ExpertPackage.from_directory(staging_root, strict=True)
        if staged.identity != package.identity:
            raise PackageHubError(
                "Package identity changed while constructing the upload snapshot."
            )

        api = HfApi(
            endpoint=endpoint,
            token=token,
            library_name=_LIBRARY_NAME,
        )
        api.create_repo(
            repo_id,
            private=private,
            repo_type="model",
            exist_ok=exist_ok,
        )
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=staging_root,
            commit_message=resolved_message,
            repo_type="model",
            revision=revision,
            create_pr=create_pr,
            allow_patterns=list(transfer_paths),
        )


def _download_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: Path | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
    allow_patterns: tuple[str, ...],
    endpoint: str | None,
) -> Path:
    """Call ``snapshot_download`` and require one ordinary snapshot path."""

    result = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        library_name=_LIBRARY_NAME,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=list(allow_patterns),
        endpoint=endpoint,
    )
    if not isinstance(result, (str, os.PathLike)):
        raise PackageHubError(
            "snapshot_download did not return a local snapshot directory."
        )
    snapshot = Path(result)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise PackageHubError(
            f"Hub snapshot is not a regular directory: {snapshot}"
        )
    return snapshot.resolve()


def _snapshot_commit_hash(snapshot: Path) -> str:
    """Extract the immutable revision from the documented Hub cache layout."""

    commit_hash = snapshot.name
    if (
        snapshot.parent.name != "snapshots"
        or len(commit_hash) != _COMMIT_HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in commit_hash)
    ):
        raise PackageHubError(
            "Hub snapshot path does not expose a lowercase 40-character "
            "resolved commit."
        )
    return commit_hash


def _package_asset_destination(
    *,
    repo_id: str,
    endpoint: str | None,
    manifest_sha256: str,
    cache_dir: Path | None,
) -> Path:
    """Return a content-addressed regular-file cache outside Hub snapshots."""

    source_key = hashlib.sha256(
        f"{endpoint or 'https://huggingface.co'}\0{repo_id}".encode("utf-8")
    ).hexdigest()
    assets_dir = None if cache_dir is None else cache_dir / "assets"
    source_root = cached_assets_path(
        library_name="anchored_moe",
        namespace="expert_packages",
        subfolder=source_key,
        assets_dir=assets_dir,
    )
    if source_root.is_symlink():
        raise PackageHubError(
            f"Package asset cache root must not be a symbolic link: {source_root}"
        )
    return source_root / manifest_sha256


def _copy_declared_files(
    source_root: Path,
    destination_root: Path,
    *,
    relative_paths: tuple[str, ...],
    permit_transport_symlinks: bool = True,
) -> None:
    """Copy only canonical package paths into one regular-file directory."""

    for relative_path in relative_paths:
        parts = PurePosixPath(relative_path).parts
        resolved_source = _resolve_declared_source(
            source_root,
            relative_path,
            permit_transport_symlinks=permit_transport_symlinks,
        )

        destination = destination_root.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved_source, destination)


def _resolve_declared_source(
    source_root: Path,
    relative_path: str,
    *,
    permit_transport_symlinks: bool,
) -> Path:
    """Resolve one package file without following an escaping symlink."""

    source = source_root.joinpath(*PurePosixPath(relative_path).parts)
    if source.is_symlink() and not permit_transport_symlinks:
        raise PackageValidationError(
            f"Symbolic links are not allowed in packages: {relative_path}"
        )
    try:
        resolved_source = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise PackageValidationError(
            f"Package payload is missing: {relative_path}"
        ) from error
    if permit_transport_symlinks and source.is_symlink():
        # Hub snapshots intentionally symlink into the repository's blob
        # cache. A tampered link that escapes that cache is never followed.
        repository_cache_root = source_root.parent.parent.resolve()
        if not resolved_source.is_relative_to(repository_cache_root):
            raise PackageHubError(
                "Hub transport symlink escapes its repository cache: "
                f"{relative_path}"
            )
    elif not resolved_source.is_relative_to(source_root.resolve()):
        raise PackageValidationError(
            f"Package payload escapes its root: {relative_path}"
        )
    if not resolved_source.is_file():
        raise PackageValidationError(
            f"Package payload is not a regular file: {relative_path}"
        )
    return resolved_source


def _load_hub_package(
    package_class: type[ExpertPackage],
    destination: Path,
    *,
    repo_id: str,
    revision: str,
) -> ExpertPackage:
    """Strictly load cached bytes and attach non-identity Hub provenance."""

    loaded = package_class.from_directory(destination, strict=True)
    return package_class(
        manifest=loaded.manifest,
        package_directory=loaded.package_directory,
        validation_report=loaded.validation_report,
        strict_validation=True,
        hub_repository=repo_id,
        hub_revision=revision,
    )


def _normalize_source(value: str | Path) -> str:
    """Validate and normalize the shared local-path/Hub-ID input."""

    if isinstance(value, Path):
        source = str(value)
    elif isinstance(value, str):
        source = value
    else:
        raise TypeError(
            "pretrained_model_name_or_path must be a string or pathlib.Path."
        )
    if not source or source != source.strip():
        raise ValueError(
            "pretrained_model_name_or_path must be non-empty without "
            "surrounding whitespace."
        )
    return source


def _looks_like_local_path(value: str) -> bool:
    """Distinguish explicit missing paths from valid Hub repository IDs."""

    path = Path(value)
    return path.is_absolute() or value.startswith((".", os.sep))


def _reject_local_hub_options(
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
    endpoint: str | None,
) -> None:
    """Reject silently ignored Hub-only options for an existing local path."""

    supplied = {
        "cache_dir": cache_dir is not None,
        "endpoint": endpoint is not None,
        "force_download": force_download,
        "local_files_only": local_files_only,
        "revision": revision is not None,
        "token": token is not None,
    }
    active = sorted(name for name, present in supplied.items() if present)
    if active:
        raise PackageHubError(
            "Hub-only options are invalid for a local package directory: "
            f"{active}."
        )


def _validate_common_arguments(
    *,
    force_download: bool,
    local_files_only: bool,
    strict: bool,
    revision: str | None,
    token: str | bool | None,
    endpoint: str | None,
) -> None:
    """Validate shared download flags before local or remote I/O."""

    for name, value in (
        ("force_download", force_download),
        ("local_files_only", local_files_only),
        ("strict", strict),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    _validate_optional_nonempty_string(revision, field_name="revision")
    _validate_token(token)
    _validate_optional_nonempty_string(endpoint, field_name="endpoint")


def _validate_upload_arguments(
    *,
    repo_id: str,
    token: str | bool | None,
    private: bool | None,
    commit_message: str | None,
    revision: str | None,
    create_pr: bool,
    exist_ok: bool,
    endpoint: str | None,
) -> None:
    """Validate upload options before package staging or remote mutation."""

    if not isinstance(repo_id, str):
        raise TypeError("repo_id must be a string.")
    validate_repo_id(repo_id)
    _validate_token(token)
    if private is not None and not isinstance(private, bool):
        raise TypeError("private must be a boolean or None.")
    _validate_optional_nonempty_string(
        commit_message,
        field_name="commit_message",
    )
    _validate_optional_nonempty_string(revision, field_name="revision")
    _validate_optional_nonempty_string(endpoint, field_name="endpoint")
    for name, value in (("create_pr", create_pr), ("exist_ok", exist_ok)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if create_pr and revision is not None:
        raise PackageHubError(
            "create_pr=True cannot be combined with an explicit revision."
        )


def _validate_optional_nonempty_string(
    value: str | None,
    *,
    field_name: str,
) -> None:
    """Validate one optional non-empty trimmed string."""

    if value is not None and (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace or None."
        )


def _validate_token(token: str | bool | None) -> None:
    """Validate the token shape without owning authentication semantics."""

    if token is not None and not isinstance(token, (str, bool)):
        raise TypeError("token must be a string, boolean, or None.")
    if isinstance(token, str) and (not token or token != token.strip()):
        raise ValueError(
            "token strings must be non-empty without surrounding whitespace."
        )
