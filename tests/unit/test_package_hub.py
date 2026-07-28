"""Acceptance tests for strict, offline PKG-006 Hub transport."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anchored_moe import (
    ExpertPackage,
    PackageHubError,
    PackageValidationError,
)
from anchored_moe.registry import hub


_COMMIT = "a" * 40


def _hub_snapshot(
    tmp_path: Path,
    package: ExpertPackage,
    *,
    commit: str = _COMMIT,
) -> Path:
    """Copy a package into the documented Hub snapshot layout."""

    snapshot = (
        tmp_path
        / "hub"
        / "models--organization--code-experts"
        / "snapshots"
        / commit
    )
    shutil.copytree(package.package_directory, snapshot)
    # Unrelated repository content is not part of the manifest-declared
    # transfer closure and therefore never enters the strict package cache.
    (snapshot / "README.md").write_text("Hub model card", encoding="utf-8")
    return snapshot


def test_from_pretrained_local_directory_is_network_free(
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """An existing path must preserve the ordinary strict local contract."""

    package = expert_package_factory("code")

    def fail_download(**_: object) -> str:
        raise AssertionError("local package resolution must not use the Hub")

    monkeypatch.setattr(hub, "snapshot_download", fail_download)
    loaded = ExpertPackage.from_pretrained(package.package_directory)

    assert loaded.identity == package.identity
    assert loaded.package_directory == package.package_directory
    assert loaded.hub_repository is None
    assert loaded.hub_revision is None

    with pytest.raises(PackageHubError, match="Hub-only options"):
        ExpertPackage.from_pretrained(
            package.package_directory,
            revision="main",
        )


def test_from_pretrained_resolves_one_commit_and_strictly_caches_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Manifest-first transfer must pin payloads and exclude unrelated files."""

    package = expert_package_factory("code")
    snapshot = _hub_snapshot(tmp_path, package)
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(snapshot)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    cache_dir = tmp_path / "cache"
    loaded = ExpertPackage.from_pretrained(
        "organization/code-experts",
        revision="release-1",
        cache_dir=cache_dir,
        token=False,
        local_files_only=True,
    )

    expected_files = {
        "manifest.json",
        *package.manifest.package_hashes,
    }
    actual_files = {
        path.relative_to(loaded.package_directory).as_posix()
        for path in loaded.package_directory.rglob("*")
        if path.is_file()
    }
    assert loaded.identity == package.identity
    assert loaded.hub_repository == "organization/code-experts"
    assert loaded.hub_revision == _COMMIT
    assert loaded.package_directory != snapshot
    assert actual_files == expected_files
    assert not any(path.is_symlink() for path in loaded.package_directory.rglob("*"))

    assert len(calls) == 2
    assert calls[0]["revision"] == "release-1"
    assert calls[0]["allow_patterns"] == ["manifest.json"]
    assert calls[1]["revision"] == _COMMIT
    assert set(calls[1]["allow_patterns"]) == expected_files
    assert all(call["repo_type"] == "model" for call in calls)
    assert all(call["local_files_only"] is True for call in calls)
    assert all(call["token"] is False for call in calls)


def test_from_pretrained_rejects_unpinned_or_tampered_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Unsafe cache layout and payload bytes must fail before publication."""

    package = expert_package_factory("code")
    unpinned = tmp_path / "hub" / "working-tree"
    shutil.copytree(package.package_directory, unpinned)
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_: str(unpinned),
    )
    with pytest.raises(PackageHubError, match="resolved commit"):
        ExpertPackage.from_pretrained(
            "organization/code-experts",
            cache_dir=tmp_path / "unpinned-cache",
        )

    snapshot = _hub_snapshot(tmp_path, package, commit="b" * 40)
    payload = snapshot / package.manifest.expert_weight_files[0]
    payload.write_bytes(payload.read_bytes() + b"tampered")
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_: str(snapshot),
    )
    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        ExpertPackage.from_pretrained(
            "organization/code-experts",
            cache_dir=tmp_path / "tampered-cache",
        )


def test_from_pretrained_rejects_transport_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """The expected Hub symlink mechanism must remain inside its repo cache."""

    package = expert_package_factory("code")
    snapshot = _hub_snapshot(tmp_path, package)
    external = tmp_path / "external-manifest.json"
    shutil.copyfile(snapshot / "manifest.json", external)
    (snapshot / "manifest.json").unlink()
    (snapshot / "manifest.json").symlink_to(external)
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_: str(snapshot),
    )

    with pytest.raises(PackageHubError, match="escapes"):
        ExpertPackage.from_pretrained(
            "organization/code-experts",
            cache_dir=tmp_path / "cache",
        )


def test_from_pretrained_materializes_legitimate_hub_cache_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Documented snapshot-to-blob links must become regular package files."""

    package = expert_package_factory("code")
    repository_cache = (
        tmp_path / "hub" / "models--organization--code-experts"
    )
    snapshot = repository_cache / "snapshots" / _COMMIT
    blobs = repository_cache / "blobs"
    blobs.mkdir(parents=True)
    for index, source in enumerate(
        (
            package.package_directory / "manifest.json",
            *(
                package.package_directory / relative
                for relative in package.manifest.package_hashes
            ),
        )
    ):
        relative = (
            "manifest.json"
            if source.name == "manifest.json"
            else package.manifest.expert_weight_files[0]
        )
        blob = blobs / f"blob-{index}"
        shutil.copyfile(source, blob)
        link = snapshot / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)

    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_: str(snapshot),
    )
    loaded = ExpertPackage.from_pretrained(
        "organization/code-experts",
        cache_dir=tmp_path / "cache",
    )

    assert loaded.identity == package.identity
    assert not any(path.is_symlink() for path in loaded.package_directory.rglob("*"))


class _FakeHubApi:
    """Capture upload calls while the private staging directory still exists."""

    instances: list[_FakeHubApi] = []

    def __init__(self, **kwargs: object) -> None:
        self.client_kwargs = dict(kwargs)
        self.create_calls: list[tuple[str, dict[str, object]]] = []
        self.upload_calls: list[dict[str, object]] = []
        self.uploaded_bytes: dict[str, bytes] = {}
        type(self).instances.append(self)

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        self.create_calls.append((repo_id, dict(kwargs)))
        return object()

    def upload_folder(self, **kwargs: object) -> object:
        self.upload_calls.append(dict(kwargs))
        folder = Path(kwargs["folder_path"])
        self.uploaded_bytes = {
            path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*")
            if path.is_file()
        }
        return {"commit_url": "https://huggingface.co/commit/example"}


def test_push_to_hub_uploads_only_a_second_strict_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Upload must use exact tracked bytes and explicit repository options."""

    package = expert_package_factory("code")
    _FakeHubApi.instances.clear()
    monkeypatch.setattr(hub, "HfApi", _FakeHubApi)

    result = package.push_to_hub(
        "organization/code-experts",
        token="secret-token",
        private=True,
        commit_message="Publish code experts",
        exist_ok=True,
        endpoint="https://hub.example.test",
    )

    client = _FakeHubApi.instances[-1]
    expected_paths = {
        "manifest.json",
        *package.manifest.package_hashes,
    }
    assert result == {"commit_url": "https://huggingface.co/commit/example"}
    assert client.client_kwargs == {
        "endpoint": "https://hub.example.test",
        "token": "secret-token",
        "library_name": "anchored-moe",
    }
    assert client.create_calls == [
        (
            "organization/code-experts",
            {
                "private": True,
                "repo_type": "model",
                "exist_ok": True,
            },
        )
    ]
    assert len(client.upload_calls) == 1
    assert set(client.upload_calls[0]["allow_patterns"]) == expected_paths
    assert client.upload_calls[0]["repo_type"] == "model"
    assert client.upload_calls[0]["commit_message"] == "Publish code experts"
    assert set(client.uploaded_bytes) == expected_paths
    for relative_path, uploaded in client.uploaded_bytes.items():
        assert uploaded == (package.package_directory / relative_path).read_bytes()


def test_push_to_hub_rejects_unsafe_input_before_remote_mutation(
    monkeypatch: pytest.MonkeyPatch,
    expert_package_factory: Callable[..., ExpertPackage],
) -> None:
    """Strict provenance, current hashes, and coherent PR options are required."""

    _FakeHubApi.instances.clear()
    monkeypatch.setattr(hub, "HfApi", _FakeHubApi)
    non_strict = expert_package_factory(
        "code",
        strict=False,
        add_untracked_file=True,
    )
    with pytest.raises(PackageHubError, match="strict validation"):
        non_strict.push_to_hub("organization/code-experts")

    package = expert_package_factory("math")
    payload = package.package_directory / package.manifest.expert_weight_files[0]
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(PackageValidationError, match="SHA-256 mismatch"):
        package.push_to_hub("organization/math-experts")

    clean = expert_package_factory("biology")
    with pytest.raises(PackageHubError, match="cannot be combined"):
        clean.push_to_hub(
            "organization/biology-experts",
            revision="release",
            create_pr=True,
        )
    assert _FakeHubApi.instances == []


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"force_download": 1}, TypeError, "force_download"),
        ({"strict": False}, PackageHubError, "strict=True"),
        ({"token": ""}, ValueError, "token strings"),
        ({"endpoint": " "}, ValueError, "endpoint"),
    ],
)
def test_remote_from_pretrained_validates_arguments_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    error_type: type[Exception],
    match: str,
) -> None:
    """Malformed options must not start a Hub request."""

    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_: pytest.fail("transport should not run"),
    )
    with pytest.raises(error_type, match=match):
        ExpertPackage.from_pretrained(
            "organization/code-experts",
            **kwargs,
        )
