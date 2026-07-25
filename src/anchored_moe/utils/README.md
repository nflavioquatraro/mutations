# Utilities

## Scope

Utilities provide small deterministic behavioral primitives shared across
feature subpackages. They do not own package schemas, model behavior, or
registry lifecycle.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| PKG-001 | `canonical_json_bytes`, `sha256_json`, `sha256_file` | `hashing.py` | Validated |

## Behavioral contracts

Canonical JSON sorts object keys, removes insignificant whitespace, preserves
Unicode, encodes with UTF-8, and rejects `NaN` and infinities.

File hashing reads bounded chunks, so memory use is:

\[
O(\text{chunk size})
\]

rather than \(O(\text{file size})\).

## Algorithms

1. Canonicalize JSON with fixed serialization options.
2. Hash canonical bytes or stream file chunks through SHA-256.
3. Return lowercase hexadecimal digests.

## Assumptions and invariants

- SHA-256 is an integrity and content-identity mechanism, not a signature.
- Chunk size must be a positive integer.
- File hashing accepts regular files only.
- Hashing never mutates its inputs.

## Hugging Face integration

The helpers are format-neutral and can hash local files produced by
Transformers, safetensors, or Hub downloads.

## Configuration and examples

```python
from anchored_moe.utils import sha256_file

digest = sha256_file("experts/model.safetensors")
```

## References

- [FIPS 180-4: Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4)
  defines SHA-256.
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
  motivates deterministic JSON, although this implementation defines a smaller
  project-specific contract and does not claim full RFC 8785 conformance.

## Tests and known limitations

Tests cover chunked file hashing, mapping-order independence, Unicode, and
non-finite value rejection. Cryptographic signing and remote streaming are not
implemented.
