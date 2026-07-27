# Registry and package manifests

## Scope

The registry subpackage owns immutable package metadata, local payload
validation, active package ordering, lifecycle operations, compatibility
binding, and deterministic registry identity.

PKG-001 defines the versioned manifest and safe payload contract. PKG-002 adds
strict local `ExpertPackage` descriptors and an atomic in-memory
`ExpertRegistry`. It does not materialize payload tensors, infer layer/expert
ownership from filenames, attach modules to a model, recalibrate routers,
perform upgrades or rollback, persist registry state, or transfer Hub content.

## Implemented strategies

| Feature ID | Public configuration or API | Module | Status |
|---|---|---|---|
| PKG-001 | `package_format_version: 1` | `manifest.py` | Validated |
| PKG-002 | `ExpertPackage.from_directory`, `ExpertRegistry` | `package.py`, `registry.py` | Validated |

`PackageIdentity` is the portable specialization/version/content identity.
`RegistrySnapshot` and `RegistryChange` provide immutable audit records for
active state and successful transitions.

## Mathematical formulation

For every package payload \(f_i\), the manifest records

\[
d_i=\operatorname{SHA256}(\operatorname{bytes}(f_i)).
\]

Validation succeeds only when the recomputed digest \(\hat d_i=d_i\). The
manifest identity is

\[
d_{\mathrm{manifest}}
=\operatorname{SHA256}(\operatorname{CanonicalJSON}(M)).
\]

A package identity is the portable tuple

\[
p=(\text{specialization},\text{version},d_{\mathrm{manifest}}).
\]

For ordered active package identities \(P=[p_1,\ldots,p_n]\), registry format
version \(v\), and no local filesystem paths, the active registry identity is

\[
d_{\mathrm{registry}}
=
\operatorname{SHA256}\left(
\operatorname{CanonicalJSON}
\left(
\{\text{registry format}:v,\text{active packages}:P\}
\right)
\right).
\]

The first successful installation permanently binds a registry instance to a
compatibility digest over:

- base repository, revision, configuration hash, and weight hashes;
- base numerical precision;
- tokenizer repository and revision;
- converted layer indices;
- router feature specification and projection hash.

This compatibility identity remains bound when the active registry becomes
empty.

## Algorithms

Package descriptor construction:

1. Reject missing or symbolic-link package roots and manifests.
2. Parse the JSON-only manifest without executing package content.
3. Validate every manifest invariant.
4. Verify tracked payload hashes and safe formats.
5. In strict mode, reject every untracked file and symbolic link.
6. Store the resolved local path, immutable manifest, and validation report.

Installation:

1. Require a descriptor produced with strict validation.
2. Re-read the live manifest and require the same canonical digest.
3. Revalidate every live payload before acquiring the registry lock.
4. Compute package identity and compatibility identity.
5. Under the lock, reject an already-active version, changed content for a
   known immutable version, or incompatible anchor/router feature space.
6. Assign a new historical slot only for a previously unseen version.
7. Activate the descriptor and return before/after snapshots.

Unload and reinsertion:

1. Require the exact active `PackageIdentity`, including manifest digest.
2. Remove the active descriptor atomically.
3. Retain only the version's historical slot and known manifest digest.
4. On reinsertion, revalidate identical content and reactivate the original
   slot.
5. Recompute the same ordered registry hash when the same active package set is
   restored.

Every failure condition is checked before active state changes.

## Assumptions and invariants

- Package format version 1 and registry format version 1 are the only accepted
  versions.
- Expert tensors use safetensors; router data uses JSON or safetensors.
- Payload paths are normalized package-relative POSIX paths.
- Loading a descriptor never imports or executes package code.
- Installation always requires strict directory closure.
- Live files are revalidated immediately before activation.
- Specialization versions are immutable: a known `(name, version)` cannot
  later refer to another manifest digest.
- Active package order uses stable historical slots.
- Unload releases the active descriptor and its local path from registry state.
- Reinsertion restores the original slot.
- Registry hashes exclude local paths and include active order.
- The compatibility binding persists across an empty active registry.
- Registry transitions are in-memory, lock-protected, and atomic with respect
  to registry readers.
- Installation never triggers router fitting, recalibration, or model mutation.

The registry validates compatibility between packages. Validation against a
concrete live model and exact runtime environment remains part of future model
composition.

## Hugging Face integration

Repository and revision fields use Hugging Face Hub naming conventions.
Packages are independently uploadable because manifests contain package-local
paths and never duplicate an anchor implicitly.

`ExpertPackage.from_directory()` currently supports local directories only.
`from_pretrained()` and Hub download behavior are deferred to PKG-006. A future
runtime package format must explicitly map payloads to `(layer, expert_id)`
slots before package-driven `RoutedMLP` attachment can be implemented safely.

## Configuration and examples

```python
from anchored_moe import ExpertPackage, ExpertRegistry

package = ExpertPackage.from_directory("./code-experts")
registry = ExpertRegistry()

installed = registry.install(package)
print(installed.after.registry_sha256)

registry.unload(package.identity)
registry.install(ExpertPackage.from_directory("./code-experts"))
assert registry.active_registry_sha256 == installed.after.registry_sha256
```

See `examples/verify_registry_lifecycle.py` for an executable local lifecycle
example.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines exact package compatibility, stable ordering, auditable
  installation, pure unloading, and immutable upgrade semantics.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates non-executable tensor payloads.
- [Hugging Face Hub repository revisions](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-from-the-hub)
  defines repository revision semantics.
- [FIPS PUB 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
  defines SHA-256.
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
  motivates stable JSON hashing. The project uses its documented local
  canonical JSON rules rather than claiming full RFC 8785 conformance.
- [Python `threading.RLock` documentation](https://docs.python.org/3/library/threading.html#rlock-objects)
  defines the synchronization primitive protecting registry state.

PKG-001 and PKG-002 are framework infrastructure and do not derive a training
or optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover deterministic manifest serialization, payload tampering,
strict directory closure, malformed formats, portable identities, deterministic
empty and active registry hashes, stable installation order, live-file
revalidation, duplicate activation, exact unloading, reinsertion restoration,
compatibility rejection, immutable-version conflicts, non-strict descriptor
rejection, atomic failure behavior, documentation, examples, and wheel content.

Payload materialization, model attachment, persistent registry files,
cross-process locking, runtime-version enforcement, atomic upgrades, rollback,
registry-specific routers, Hub transfer, and distributed lifecycle operations
remain unsupported.
