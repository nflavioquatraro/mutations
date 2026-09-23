"""Versioned branch-only safetensors composition packages with strict reload."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from anchored_moe.utils.hashing import canonical_json_bytes, sha256_file

from .configuration import BranchedModelConfig
from .model import BranchedModel
from .parameters import checked_parameters
from .routing import (
    ArgmaxSelection,
    FixedBranchSelection,
    LinearRequestScorer,
    ThresholdSelection,
)

BRANCH_PACKAGE_VERSION = 1


class BranchPackageError(ValueError):
    """Raised for incompatible, incomplete, stale or corrupted branch packages."""


def _selection_dict(selection):
    if isinstance(selection, ThresholdSelection):
        return {
            "type": "threshold",
            "thresholds": selection.thresholds,
            "kind": selection.kind,
            "composition_key": selection.composition_key,
            "calibration_data_id": selection.calibration_data_id,
        }
    if isinstance(selection, FixedBranchSelection):
        return {"type": "fixed", "branch_id": selection.branch_id}
    if isinstance(selection, ArgmaxSelection):
        return {"type": "argmax"}
    raise BranchPackageError(
        "Custom selection strategies need a separate explicit serializer."
    )


def _selection_from_dict(raw):
    raw = dict(raw)
    kind = raw.pop("type")
    if kind == "threshold":
        raw["thresholds"] = tuple(tuple(pair) for pair in raw["thresholds"])
        return ThresholdSelection(**raw)
    if kind == "fixed":
        return FixedBranchSelection(**raw)
    if kind == "argmax" and not raw:
        return ArgmaxSelection()
    raise BranchPackageError("Unknown or malformed selection strategy.")


def save_branch_package(model: BranchedModel, path: str | Path) -> Path:
    """Atomically export owned suffixes, affine scorer, policy and provenance.

    The original base is referenced by identity and never bundled. Existing
    destinations and stale calibration are rejected. Only builtin scorer and
    selection formats are serialized; no pickle or executable configuration.
    """
    destination = Path(path).absolute()
    with model.operation():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if model.calibration_stale:
            raise BranchPackageError("Cannot export stale router calibration.")
        if not isinstance(model.scorer, LinearRequestScorer):
            raise BranchPackageError(
                "Only affine request scorers have a package format."
            )
        selection = _selection_dict(model.selection)
        scorer = model.scorer
        tensors = {
            f"branch/{branch_id}/{key}": value
            for branch_id, parameters in model._parameters.items()
            for key, value in parameters.items()
        }
        tensors.update(
            {f"scorer/{key}": value for key, value in scorer.parameters.items()}
        )
        manifest = {
            "format_version": BRANCH_PACKAGE_VERSION,
            "backend": model.backend.identity,
            "config": model.config.to_dict(),
            "selection": selection,
            "scorer": {
                "branch_ids": scorer.branch_ids,
                "kind": scorer.kind,
                "temperature": scorer.temperature,
            },
            "composition_key": model.composition_key,
            "training_data_ids": sorted(model._training_data_ids),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
        )
        try:
            tensor_path = stage / "branches.safetensors"
            save_file(tensors, str(tensor_path))
            manifest["tensor_sha256"] = sha256_file(tensor_path)
            (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            # Validate the actual staged artifact, including every tensor, before
            # making the directory visible at its final name.
            load_branch_package(stage, model.backend)
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(stage, destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return destination


def load_branch_package(path: str | Path, backend) -> BranchedModel:
    """Reload on a caller-supplied identical base/backend without network access.

    Checks version, exact runtime/base identity, payload digest, parameter keys,
    shapes, dtypes, finite values, composition hash and calibration provenance.
    Hashes establish integrity, not package author authenticity.
    """
    directory = Path(path)
    try:
        manifest_path, tensor_path = (
            directory / "manifest.json",
            directory / "branches.safetensors",
        )
        if directory.is_symlink() or any(
            p.is_symlink() or not p.is_file() for p in (manifest_path, tensor_path)
        ):
            raise BranchPackageError("Expected regular local package files.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_fields = {
            "format_version",
            "backend",
            "config",
            "selection",
            "scorer",
            "composition_key",
            "training_data_ids",
            "tensor_sha256",
        }
        if (
            set(manifest) != expected_fields
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != BRANCH_PACKAGE_VERSION
        ):
            raise BranchPackageError("Unsupported package version or manifest fields.")
        if manifest["backend"] != backend.identity:
            raise BranchPackageError(
                "Package base, split or numerical runtime does not match."
            )
        if sha256_file(tensor_path) != manifest["tensor_sha256"]:
            raise BranchPackageError("Branch tensor digest mismatch.")
        tensors = load_file(str(tensor_path))
        if any(
            tensors[key].dtype != np.float32 for key in ("scorer/weight", "scorer/bias")
        ):
            raise BranchPackageError(
                "Scorer tensors must preserve their float32 dtype."
            )
        score = manifest["scorer"]
        if set(score) != {"branch_ids", "kind", "temperature"}:
            raise BranchPackageError("Invalid scorer declaration.")
        scorer = LinearRequestScorer(
            score["branch_ids"],
            tensors["scorer/weight"],
            tensors["scorer/bias"],
            kind=score["kind"],
            temperature=score["temperature"],
        )
        model = BranchedModel(
            backend,
            BranchedModelConfig.from_dict(manifest["config"]),
            scorer,
            _selection_from_dict(manifest["selection"]),
        )
        expected = {"scorer/weight", "scorer/bias"}
        for branch_id, template in model._parameters.items():
            prefix = f"branch/{branch_id}/"
            expected.update(prefix + key for key in template)
            candidate = {
                key[len(prefix) :]: value
                for key, value in tensors.items()
                if key.startswith(prefix)
            }
            model._parameters[branch_id] = checked_parameters(candidate, template)
        if set(tensors) != expected:
            raise BranchPackageError("Unexpected or missing serialized tensor names.")
        data_ids = manifest["training_data_ids"]
        if (
            not isinstance(data_ids, list)
            or any(not isinstance(i, str) or not i for i in data_ids)
            or len(set(data_ids)) != len(data_ids)
        ):
            raise BranchPackageError("Invalid training provenance.")
        model._training_data_ids = set(data_ids)
        if (
            model.composition_key != manifest["composition_key"]
            or model.calibration_stale
        ):
            raise BranchPackageError(
                "Composition identity or calibration binding mismatch."
            )
        model.set_selection(model.selection)
        if (
            isinstance(model.selection, FixedBranchSelection)
            and model.selection.branch_id not in model.response_ids
        ):
            raise BranchPackageError("Unknown forced response branch.")
        return model
    except BranchPackageError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise BranchPackageError(f"Invalid branch package: {exc}") from exc
