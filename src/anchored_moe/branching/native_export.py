"""Standard Needle 3 .cact response models and explicit native branch bundles.

References:
    https://github.com/cactus-compute/needle/blob/main/needle/model/export.py
    https://github.com/cactus-compute/needle/blob/main/needle/model/finetune.py

Differences:
    Materializes one frozen-prefix plus selected-suffix model before calling the
    pinned upstream CQ4 exporter. A bundle contains separate standard models;
    the stock engine has no split/resume or branch-container ABI. Original probe
    heads are omitted, and source routing calibration is never labeled valid for
    quantized deployment. Export never downloads weights, tokenizers or engines.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from anchored_moe.utils.hashing import canonical_json_bytes, sha256_file

from .needle import NeedleBranchBackend
from .package import _selection_dict
from .routing import LinearRequestScorer


@dataclass(frozen=True)
class CactExportConfig:
    """Pinned upstream deployment numerics, independent of training strategy.

    weight_bits=4 and group_size=128 match the supported local Needle exporter.
    KV uses the checkpoint's declared deployment width/window. Other CQ widths
    require a matching exporter and quantization validation, and are rejected.
    """

    weight_bits: int = 4
    group_size: int = 128

    def __post_init__(self):
        """Reject unsupported settings before creating deployment artifacts."""
        if type(self.weight_bits) is not int or self.weight_bits != 4:
            raise ValueError("The pinned native exporter supports CQ4 weights only.")
        if type(self.group_size) is not int or self.group_size != 128:
            raise ValueError("The supported native quantization group size is 128.")


@dataclass(frozen=True)
class CactExportReport:
    """A standard native response artifact; its source base is merged and included.

    Quantization changes numerics. This report verifies structure and integrity,
    not downstream quality, native confidence, or routing calibration.
    """

    path: Path
    branch_id: str
    sha256: str
    size_bytes: int
    tensor_count: int
    dropped_probe_heads: tuple[str, ...]


@dataclass(frozen=True)
class NativeBranchBundleReport:
    """Published directory, standard .cact responses and manifest integrity hash.

    Routing remains external to the stock engine, with its source parameters in
    routing.safetensors. Every .cact repeats its full prefix. Neither the native
    engine nor these files implement shared-prefix routing between models.
    """

    path: Path
    branches: tuple[CactExportReport, ...]
    manifest_sha256: str


def _tokenizer_bytes(tokenizer, cfg):
    from needle.model.export import read_tokenizer_blob

    if isinstance(tokenizer, (str, Path)):
        source = Path(tokenizer)
        if source.suffix != ".cact" or not source.is_file():
            raise ValueError(
                "Tokenizer source must be an existing Needle 3 .cact file."
            )
        blob = read_tokenizer_blob(source)
    elif isinstance(tokenizer, (bytes, bytearray)):
        blob = bytes(tokenizer)
    else:
        raise TypeError("Supply packaged tokenizer bytes or a local .cact source.")
    # Parse before upstream export: its writer checks only the piece count.
    try:
        header = struct.unpack_from("<IIIIIBBH", blob)
        count, pad, eos, bos, unk, dummy, fallback, reserved = header
        if not 0 < count <= cfg.vocab_size or pad != cfg.pad_token_id:
            raise ValueError(
                "Tokenizer vocabulary or padding ID differs from the model."
            )
        if len({pad, eos, bos, unk}) != 4 or max(pad, eos, bos, unk) >= count:
            raise ValueError("Invalid tokenizer special token IDs.")
        if cfg.out_vocab and count > cfg.out_vocab:
            raise ValueError("Tokenizer pieces exceed the output vocabulary.")
        if dummy not in (0, 1) or fallback not in (0, 1) or reserved:
            raise ValueError("Invalid packaged tokenizer flags.")
        offset, pieces, byte_pieces = struct.calcsize("<IIIIIBBH"), set(), set()
        for _ in range(count):
            score, kind, length = struct.unpack_from("<fBH", blob, offset)
            offset += struct.calcsize("<fBH")
            if (
                not np.isfinite(score)
                or kind not in range(5)
                or length <= 0
                or offset + length > len(blob)
            ):
                raise ValueError("Malformed tokenizer piece record.")
            surface = blob[offset : offset + length].decode("utf-8")
            if surface in pieces:
                raise ValueError("Duplicate tokenizer piece.")
            pieces.add(surface)
            if kind == 4:
                byte_pieces.add(surface)
            offset += length
        if offset != len(blob):
            raise ValueError("Unexpected trailing tokenizer bytes.")
        if fallback and byte_pieces != {f"<0x{i:02X}>" for i in range(256)}:
            raise ValueError("Byte fallback requires all 256 byte pieces.")
    except (struct.error, UnicodeError) as exc:
        raise ValueError("Malformed packaged tokenizer.") from exc
    return blob


def _require_exportable(model):
    if not isinstance(model.backend, NeedleBranchBackend):
        raise TypeError("Native Needle export requires a NeedleBranchBackend.")
    if model.calibration_stale:
        raise ValueError("Recalibrate or select a manual policy before native export.")


def _merged_parameters(model, branch_id):
    from flax.core import unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict
    from needle.model.architecture import HEAD_KEYS

    if branch_id not in model.response_ids:
        raise ValueError(
            "Native response export requires an installed response branch."
        )
    backend = model.backend
    suffix = model.effective_suffix(branch_id)
    native = unfreeze(backend._params)
    flat = flatten_dict(native)
    # Copy only changed stacked leaves; all other arrays remain immutable inputs
    # to the exporter. mHC leaves share a leading layer dimension like blocks.
    replacements = {}
    for key, value in suffix.items():
        _, layer, family, *tail = key.split("/")
        path = (
            ("stack", "layers", "block", *tail)
            if family == "block"
            else ("stack", "mhc_" + "/".join(tail))
        )
        if path not in replacements:
            replacements[path] = np.array(flat[path], copy=True)
        replacements[path][int(layer)] = np.asarray(value)
    flat.update(replacements)
    dropped = tuple(key for key in HEAD_KEYS if key in native)
    # Probe outputs/calibration are not trained with these branch modifications
    # or validated after this new quantization; omit them rather than reuse them.
    return unflatten_dict(
        {path: value for path, value in flat.items() if path[0] not in HEAD_KEYS}
    ), dropped


def _write_response(model, branch_id, path, blob, config):
    from needle.model.architecture import (
        effective_kv_window,
        engram_geometry,
        head_dims,
    )
    from needle.model.export import read_export, read_tokenizer_blob, write_export

    parameters, dropped = _merged_parameters(model, branch_id)
    info = write_export(
        parameters,
        model.backend._cfg,
        str(path),
        bits=config.weight_bits,
        group=config.group_size,
        tokenizer=blob,
        kv_window=effective_kv_window(model.backend._cfg),
    )
    geometry, tensors = read_export(path)
    expected = model.backend._cfg
    qk_dim, value_dim = head_dims(expected)
    orders, heads, sub_dim = engram_geometry(expected)
    if (
        any(
            geometry[key] != getattr(expected, key)
            for key in (
                "vocab_size",
                "d_model",
                "num_layers",
                "num_heads",
                "num_kv_heads",
                "mhc_lanes",
                "max_seq_len",
                "sliding_window",
                "qkv_conv_taps",
                "engram_slots",
                "out_vocab",
            )
        )
        or geometry["engram_orders"] != tuple(orders)
        or geometry["engram_layers"] != expected.engram_layers
        or set(geometry["global_layers"]) != set(expected.global_layers)
        or geometry["qk_head_dim"] != qk_dim
        or geometry["v_head_dim"] != value_dim
        or geometry["num_engram_tables"] != len(orders) * heads
        or geometry["engram_sub_dim"] != sub_dim
        or not np.isclose(geometry["rope_theta"], expected.rope_theta, rtol=1e-6)
        or read_tokenizer_blob(path) != blob
    ):
        raise ValueError(
            "Exported native geometry or tokenizer failed roundtrip validation."
        )
    if any(not np.isfinite(v).all() for v in tensors if isinstance(v, np.ndarray)):
        raise ValueError("Quantization produced nonfinite deployment tensors.")
    return CactExportReport(
        path,
        branch_id,
        sha256_file(path),
        path.stat().st_size,
        info["tensors"],
        dropped,
    )


def export_needle_branch(
    model,
    branch_id: str,
    path: str | Path,
    *,
    tokenizer,
    config: CactExportConfig | None = None,
) -> CactExportReport:
    """Atomically export one response as a standard, fully merged CQ4 .cact model.

    tokenizer is a local source .cact path or its packaged tokenizer bytes. It
    must match the model's vocabulary and token IDs; no downloads occur. LoRA is
    merged before quantization; copied suffixes replace their exact layer rows.
    Existing destinations are rejected. Original probe heads are not exported.
    Use the returned path with Needle's native --model/weights argument.
    """
    config = config or CactExportConfig()
    if not isinstance(config, CactExportConfig):
        raise TypeError("Expected a CactExportConfig.")
    destination = Path(path).absolute()
    if destination.suffix != ".cact":
        raise ValueError("A native response artifact must use the .cact extension.")
    with model.operation():
        _require_exportable(model)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        blob = _tokenizer_bytes(tokenizer, model.backend._cfg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".needle-export-", dir=destination.parent))
        try:
            staged = _write_response(
                model, branch_id, stage / destination.name, blob, config
            )
            # Exclusive creation also protects against a concurrent writer. Link
            # publication is atomic and does not require symlink privileges.
            os.link(staged.path, destination)
            return CactExportReport(
                destination,
                branch_id,
                staged.sha256,
                staged.size_bytes,
                staged.tensor_count,
                staged.dropped_probe_heads,
            )
        finally:
            shutil.rmtree(stage)


def export_needle_bundle(
    model, path: str | Path, *, tokenizer, config: CactExportConfig | None = None
) -> NativeBranchBundleReport:
    """Publish one native response per branch plus explicit external-routing data.

    manifest.json records pinned source identity, quantization, tokenizer digest,
    per-file hashes, response mapping and source routing policy. routing.safetensors
    holds the affine scorer and optional private router-suffix parameters. That
    source router still needs its original backend/base; the stock native engine
    does not execute it. Source thresholds need native-outcome reevaluation.
    No custom scorer/selector serialization or native shared-prefix ABI is implied.
    """
    from safetensors.numpy import save_file

    config = config or CactExportConfig()
    if not isinstance(config, CactExportConfig):
        raise TypeError("Expected a CactExportConfig.")
    destination = Path(path).absolute()
    with model.operation():
        _require_exportable(model)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if not isinstance(model.scorer, LinearRequestScorer):
            raise TypeError(
                "Native bundle routing metadata supports affine scorers only."
            )
        policy = _selection_dict(model.selection)
        blob = _tokenizer_bytes(tokenizer, model.backend._cfg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".needle-bundle-", dir=destination.parent))
        try:
            responses = tuple(
                _write_response(model, branch, stage / f"{branch}.cact", blob, config)
                for branch in model.response_ids
            )
            tensors = {"scorer/" + k: v for k, v in model.scorer.parameters.items()}
            if model.config.router_branch:
                tensors.update(
                    {
                        "router/" + k: v
                        for k, v in model.branch_parameters(
                            model.config.router_branch.branch_id
                        ).items()
                    }
                )
            save_file(tensors, str(stage / "routing.safetensors"))
            manifest = {
                "format": "needle-native-branches-v1",
                "exporter": {
                    "cactus-needle": version("cactus-needle"),
                    **asdict(config),
                },
                "source_backend": model.backend.identity,
                "source_composition_key": model.composition_key,
                "topology": model.config.to_dict(),
                "tokenizer_sha256": hashlib.sha256(blob).hexdigest(),
                "responses": {
                    r.branch_id: {
                        "file": r.path.name,
                        "sha256": r.sha256,
                        "bytes": r.size_bytes,
                        "dropped_probe_heads": r.dropped_probe_heads,
                    }
                    for r in responses
                },
                "routing": {
                    "execution": "external-source-backend",
                    "native_prefix_sharing": False,
                    "source_base_required": True,
                    "native_calibration": "requires_evaluation",
                    "scorer": {
                        "branch_ids": model.scorer.branch_ids,
                        "kind": model.scorer.kind,
                        "temperature": model.scorer.temperature,
                    },
                    "source_selection": policy,
                    "file": "routing.safetensors",
                    "sha256": sha256_file(stage / "routing.safetensors"),
                },
            }
            (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            manifest_hash = sha256_file(stage / "manifest.json")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(stage, destination)
            return NativeBranchBundleReport(
                destination,
                tuple(
                    CactExportReport(
                        destination / r.path.name,
                        r.branch_id,
                        r.sha256,
                        r.size_bytes,
                        r.tensor_count,
                        r.dropped_probe_heads,
                    )
                    for r in responses
                ),
                manifest_hash,
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage)
