"""Tests for deterministic, bounded-memory hashing utilities."""

from __future__ import annotations

import hashlib

import pytest

from anchored_moe.utils import canonical_json_bytes, sha256_file, sha256_json


def test_canonical_json_is_independent_of_mapping_insertion_order() -> None:
    first = {"z": 1, "message": "caffè", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "message": "caffè", "z": 1}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_json(first) == sha256_json(second)


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"invalid": float("nan")})


def test_sha256_file_streams_multiple_chunks(tmp_path) -> None:
    payload = b"anchored-moe" * 10
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=7) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_sha256_file_requires_positive_chunk_size(tmp_path, chunk_size: int) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=chunk_size)
