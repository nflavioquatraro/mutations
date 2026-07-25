# Registry and package manifests

## Scope

The registry subpackage defines immutable package metadata and validates package
payload integrity. PKG-001 does not install, unload, activate, upgrade, or fetch
packages; those lifecycle operations belong to later registry features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| PKG-001 | `package_format_version: 1` | `manifest.py` | Validated |

## Mathematical formulation

For every package payload \(f_i\), the manifest records:

\[
d_i=\operatorname{SHA256}(\operatorname{bytes}(f_i)).
\]

Validation succeeds only when the recomputed digest \(\hat d_i\) satisfies
\(\hat d_i=d_i\). The manifest identity is computed separately:

\[
d_{\mathrm{manifest}}
=\operatorname{SHA256}(\operatorname{CanonicalJSON}(M)).
\]

`manifest.json` is excluded from `package_hashes` to avoid recursive
self-hashing.

## Algorithms

Manifest construction:

1. Validate format version and required checkpoint/tokenizer pins.
2. Normalize layer indices, file paths, policy names, and mapping order.
3. Reject unsafe paths and pickle-backed tensor formats.
4. Freeze nested metadata into immutable mappings and tuples.
5. Serialize with canonical JSON ordering and no non-finite values.

Package validation:

1. Resolve every tracked path beneath the package root.
2. Reject symbolic links, missing files, and root escapes.
3. Stream each file through SHA-256 and compare its digest.
4. In strict mode, reject untracked package payloads.
5. Parse safetensors headers and router JSON without executing content.
6. Return an auditable validation report.

## Assumptions and invariants

- Package format version 1 is the only accepted version.
- Base repository, revision, configuration digest, weight digests, tokenizer
  repository, tokenizer revision, and base numerical precision are mandatory.
- Converted layers are sorted, unique, and non-negative.
- Expert tensors use safetensors; router data uses JSON or safetensors.
- Payload paths are normalized package-relative POSIX paths.
- Python, PyTorch, Transformers, PEFT, Accelerate, safetensors, and kernel
  versions or explicit absence markers are recorded.
- Training dataset identifier, revision, filters, and split are recorded.
- Loading a manifest never imports or executes package code.

For future PEFT experts, `expert_parameterization_config` must record target
modules, rank, scaling, and PEFT version. PKG-001 preserves that metadata but
does not yet define an adapter-specific validator.

## Hugging Face integration

Repository and revision fields use Hugging Face Hub naming conventions. Package
payloads are independently uploadable because the manifest refers only to
relative files and does not duplicate anchor weights.

`from_pretrained()` and Hub download behavior are intentionally deferred to
PKG-006.

## Configuration and examples

```python
from anchored_moe import ExpertPackageManifest

manifest = ExpertPackageManifest.from_json("manifest.json")
report = manifest.validate_package_files(".")
print(report.manifest_sha256)
```

See `examples/inspect_expert_package.py` for an executable package construction
and validation example.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines the package fields, independent package unit, and compatibility pins.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor payload requirement.
- [Hugging Face Hub repository revisions](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-from-the-hub)
  defines repository revision semantics.

No paper-derived algorithm is implemented by PKG-001.

## Tests and known limitations

Acceptance tests cover deterministic JSON round trips, missing and unknown
fields, checkpoint pins, unsafe paths, strict directory closure, malformed
safetensors, and payload tampering. Installation, registry hashes, atomic
upgrades, rollback, and Hub transfer are not implemented.
