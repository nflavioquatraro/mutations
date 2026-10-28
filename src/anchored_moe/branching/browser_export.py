"""Base-plus-patch delivery of ordinary, independently executable .cact models.

References:
    https://www.cactuscompute.com/blog/cact-format

Differences:
    Patches are a transport format, never a new Needle architecture or runtime
    format. COPY/INSERT instructions reconstruct an exact validated export; no
    floating-point merging or requantization occurs in the browser.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from anchored_moe.utils.hashing import canonical_json_bytes, sha256_file

from .generative import GenerativeRouterConfig
from .native_export import CactExportConfig, _tokenizer_bytes, _write_response
from .needle import NeedleBranchBackend

PATCH_MAGIC = b"NDPATCH1"
MAX_MODEL_BYTES = 512 * 1024 * 1024


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def create_cact_patch(base: bytes, target: bytes, *, chunk_size: int = 4096) -> bytes:
    """Encode an exact target as base byte ranges plus literal replacement bytes.

    Both inputs are complete exports from the same pinned exporter/settings.
    Fixed-size chunk matching also deduplicates repeated blocks. Consecutive
    instructions are coalesced. File-format interpretation remains upstream's job.
    """
    if not isinstance(base, bytes) or not isinstance(target, bytes):
        raise TypeError("Patch inputs must be immutable bytes.")
    if not base or not target or max(len(base), len(target)) > MAX_MODEL_BYTES:
        raise ValueError("Empty or oversized model archive.")
    if type(chunk_size) is not int or not 64 <= chunk_size <= 1024 * 1024:
        raise ValueError("chunk_size must be between 64 bytes and 1 MiB.")
    index = {}
    for offset in range(0, len(base), chunk_size):
        block = base[offset : offset + chunk_size]
        index.setdefault(_digest(block), offset)
    literals = bytearray()
    operations = []
    for start in range(0, len(target), chunk_size):
        block = target[start : start + chunk_size]
        offset = index.get(_digest(block))
        if offset is not None and base[offset : offset + len(block)] == block:
            source = "base"
        else:
            source, offset = "patch", len(literals)
            literals.extend(block)
        if (
            operations
            and operations[-1][0] == source
            and operations[-1][1] + operations[-1][2] == offset
        ):
            operations[-1][2] += len(block)
        else:
            operations.append([source, offset, len(block)])
    header = canonical_json_bytes(
        {
            "format": "needle-byte-patch-v1",
            "base": {"sha256": _digest(base), "bytes": len(base)},
            "target": {"sha256": _digest(target), "bytes": len(target)},
            "payload_sha256": _digest(literals),
            "operations": operations,
        }
    )
    return PATCH_MAGIC + struct.pack("<I", len(header)) + header + literals


def apply_cact_patch(base: bytes, patch: bytes) -> bytes:
    """Validate all ranges/digests and reconstruct the exact full model bytes."""
    if len(patch) < 12 or patch[:8] != PATCH_MAGIC:
        raise ValueError("Invalid Needle patch magic.")
    header_size = struct.unpack_from("<I", patch, 8)[0]
    if header_size > 16 * 1024 * 1024 or 12 + header_size > len(patch):
        raise ValueError("Invalid patch header size.")
    try:
        header = json.loads(patch[12 : 12 + header_size])
        payload = patch[12 + header_size :]
        size = header["target"]["bytes"]
        if (
            header["format"] != "needle-byte-patch-v1"
            or header["base"] != {"sha256": _digest(base), "bytes": len(base)}
            or type(size) is not int
            or not 0 < size <= MAX_MODEL_BYTES
            or header["payload_sha256"] != _digest(payload)
        ):
            raise ValueError("Patch identity, size or integrity mismatch.")
        operations = header["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("Missing patch operations.")
        total = 0
        for source, offset, length in operations:
            if (
                source not in ("base", "patch")
                or type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length <= 0
                or offset + length > len(base if source == "base" else payload)
            ):
                raise ValueError("Invalid patch range.")
            total += length
        if total != size:
            raise ValueError("Patch output size mismatch.")
        output = b"".join(
            (base if source == "base" else payload)[offset : offset + length]
            for source, offset, length in operations
        )
        if _digest(output) != header["target"]["sha256"]:
            raise ValueError("Reconstructed model hash mismatch.")
        return output
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed Needle patch.") from exc


@dataclass(frozen=True)
class BrowserBranchBundleReport:
    """Transport sizes include the base and patches, excluding JS/engine assets."""

    path: Path
    manifest_sha256: str
    full_model_bytes: int
    download_bytes: int


def export_needle_browser_bundle(
    model,
    path: str | Path,
    *,
    router: GenerativeRouterConfig,
    tokenizer,
    config: CactExportConfig | None = None,
) -> BrowserBranchBundleReport:
    """Atomically publish a frozen base, generative router and response patches.

    The base and all variants use identical CQ4 settings. LoRA is merged offline.
    The generative contract replaces the affine routing strategy for this bundle;
    affine calibration is neither consumed nor exported. Training is explicit.
    Official needle.js/needle.wasm are supplied separately by the application.
    """
    config = config or CactExportConfig()
    if not isinstance(model.backend, NeedleBranchBackend):
        raise TypeError("Browser native export requires a NeedleBranchBackend.")
    if not isinstance(config, CactExportConfig) or not isinstance(
        router, GenerativeRouterConfig
    ):
        raise TypeError("Expected native export and generative router configurations.")
    if (
        router.branch_ids != model.response_ids
        or router.fallback_branch != model.config.fallback_branch
        or model.config.router_branch is None
    ):
        raise ValueError(
            "Router candidates/fallback must match a model with a router suffix."
        )
    destination = Path(path).absolute()
    with model.operation():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = _tokenizer_bytes(tokenizer, model.backend._cfg)
        stage = Path(
            tempfile.mkdtemp(prefix=".needle-browser-", dir=destination.parent)
        )
        try:
            base_id = router.fallback_branch
            base_report = _write_response(
                model, base_id, stage / "base.cact", blob, config
            )
            base = base_report.path.read_bytes()
            if len(base) > MAX_MODEL_BYTES:
                raise ValueError("Base exceeds browser reconstruction size limit.")
            base_asset = {
                "file": "base.cact",
                "bytes": len(base),
                "sha256": _digest(base),
            }
            models = {base_id: {"kind": "base", "target": base_asset}}
            full_size, download_size = len(base), len(base)
            router_id = model.config.router_branch.branch_id
            for branch in (*[i for i in model.response_ids if i != base_id], router_id):
                full_path = stage / "candidate.cact"
                _write_response(
                    model, branch, full_path, blob, config, allow_router=True
                )
                target = full_path.read_bytes()
                patch = create_cact_patch(base, target)
                if apply_cact_patch(base, patch) != target:
                    raise ValueError("Patch roundtrip failed.")
                patch_path = stage / f"{branch}.ndpatch"
                patch_path.write_bytes(patch)
                full_path.unlink()
                models[branch] = {
                    "kind": "patch",
                    "target": {"bytes": len(target), "sha256": _digest(target)},
                    "patch": {
                        "file": patch_path.name,
                        "bytes": len(patch),
                        "sha256": _digest(patch),
                    },
                }
                full_size += len(target)
                download_size += len(patch)
            manifest = {
                "format": "needle-browser-branches-v1",
                "exporter": {
                    "cactus-needle": version("cactus-needle"),
                    **asdict(config),
                },
                "source_backend": model.backend.identity,
                "topology": model.config.to_dict(),
                "base": base_asset,
                "models": models,
                "routing": {
                    "strategy": "generative-tool-v1",
                    "model_id": router_id,
                    **router.to_dict(),
                },
                "native_prefix_sharing": False,
                "quality_validation": "requires_held_out_evaluation",
                "full_model_bytes": full_size,
                "download_bytes": download_size,
            }
            (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            shutil.copytree(
                Path(__file__).with_name("browser"),
                stage / "browser",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            manifest_hash = sha256_file(stage / "manifest.json")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(stage, destination)
            return BrowserBranchBundleReport(
                destination, manifest_hash, full_size, download_size
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage)
