"""Exercise PKG-006 download and upload semantics without network access."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from anchored_moe import ExpertPackage, ExpertPackageManifest
from anchored_moe.registry import PACKAGE_FORMAT_VERSION
from anchored_moe.utils import sha256_file, sha256_json


_COMMIT = "a" * 40


class _OfflineHubApi:
    """Record the exact staged upload closure for this offline example."""

    uploaded_files: tuple[str, ...] = ()

    def __init__(self, **_: object) -> None:
        pass

    def create_repo(self, repo_id: str, **_: object) -> str:
        return repo_id

    def upload_folder(self, **kwargs: object) -> dict[str, str]:
        folder = Path(kwargs["folder_path"])
        type(self).uploaded_files = tuple(
            sorted(
                path.relative_to(folder).as_posix()
                for path in folder.rglob("*")
                if path.is_file()
            )
        )
        return {"commit_url": "offline://organization/code-experts/commit"}


def _create_package(root: Path) -> ExpertPackage:
    """Create one minimal immutable package with a safe tensor payload."""

    package_root = root / "local-package"
    relative_weight = "experts/model.safetensors"
    weight_path = package_root / relative_weight
    weight_path.parent.mkdir(parents=True)
    save_file({"expert.weight": torch.zeros(2, 2)}, weight_path)
    manifest = ExpertPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION,
        specialization_name="code",
        specialization_version="1.0.0",
        package_structure="single_expert",
        base_model_repository="Qwen/Qwen3-4B",
        base_model_revision="pinned-base-revision",
        base_config_sha256=sha256_json({"model_type": "qwen3"}),
        base_weight_sha256={"model.safetensors": "1" * 64},
        tokenizer_repository="Qwen/Qwen3-4B",
        tokenizer_revision="pinned-tokenizer-revision",
        base_model_dtype="float32",
        converted_layers=[2, 3],
        expert_parameterization="full_mlp",
        expert_parameterization_config={"initialization": "anchor_copy"},
        expert_count=1,
        weight_dtype="float32",
        expert_weight_files=[relative_weight],
        runtime_versions={
            "accelerate": "installed",
            "kernels": "torch-default-cpu",
            "peft": "installed",
            "python": "3.12",
            "safetensors": "installed",
            "torch": torch.__version__,
            "transformers": "installed",
        },
        training_dataset={
            "dataset_id": "example/code",
            "revision": "pinned-data-revision",
            "split": "train",
            "filters": [],
        },
        package_hashes={relative_weight: sha256_file(weight_path)},
    )
    manifest.save_json(package_root / "manifest.json")
    return ExpertPackage.from_directory(package_root)


def main() -> None:
    """Resolve a pinned fake snapshot and stage an exact fake Hub upload."""

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-package-hub-"
    ) as temporary:
        root = Path(temporary)
        package = _create_package(root)
        snapshot = (
            root
            / "hub"
            / "models--organization--code-experts"
            / "snapshots"
            / _COMMIT
        )
        shutil.copytree(package.package_directory, snapshot)
        (snapshot / "README.md").write_text(
            "Unrelated Hub repository file",
            encoding="utf-8",
        )

        with patch(
            "anchored_moe.registry.hub.snapshot_download",
            return_value=str(snapshot),
        ):
            downloaded = ExpertPackage.from_pretrained(
                "organization/code-experts",
                revision="release-1",
                cache_dir=root / "cache",
                local_files_only=True,
            )

        with patch("anchored_moe.registry.hub.HfApi", _OfflineHubApi):
            upload = downloaded.push_to_hub(
                "organization/code-experts-copy",
                commit_message="Offline PKG-006 verification",
            )

        downloaded_files = tuple(
            sorted(
                path.relative_to(downloaded.package_directory).as_posix()
                for path in downloaded.package_directory.rglob("*")
                if path.is_file()
            )
        )
        print(
            json.dumps(
                {
                    "downloaded_files": downloaded_files,
                    "hub_repository": downloaded.hub_repository,
                    "hub_revision": downloaded.hub_revision,
                    "identity_preserved": downloaded.identity == package.identity,
                    "upload": upload,
                    "uploaded_files": _OfflineHubApi.uploaded_files,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
