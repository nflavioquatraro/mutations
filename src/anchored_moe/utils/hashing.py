"""Deterministic SHA-256 helpers for package manifests and payload files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value into deterministic UTF-8 bytes.

    Object keys are sorted, insignificant whitespace is removed, Unicode is
    retained, and non-finite floats are rejected. Those rules make the output
    suitable for hashing across supported Python runtimes.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Hash a regular file without loading the complete payload into memory.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read per iteration. It must be positive.

    Raises:
        ValueError: If ``chunk_size`` is not positive.
        FileNotFoundError: If ``path`` does not identify a regular file.
    """

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash non-file path: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
