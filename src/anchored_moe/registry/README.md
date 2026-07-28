# Registry and package manifests

## Scope

The registry subpackage owns immutable package metadata, local payload
validation, active package ordering, lifecycle operations, compatibility
binding, and deterministic registry identity.

PKG-001 defines the versioned manifest and safe payload contract. PKG-002 adds
strict local `ExpertPackage` descriptors and an atomic in-memory
`ExpertRegistry`. PKG-003 derives immutable expert-to-mixture-group metadata
from active package manifests. PKG-004 adds atomic same-specialization version
replacement and stale-safe rollback. PKG-007 defines explicit full-MLP payload
ownership metadata used by the inference composition boundary. NB-003 adds
staged expert-only package export from independently owned `FullMLPExpert`
objects. NB-004 derives a new immutable router-bearing version from a calibrated
frozen router while preserving source expert bytes. The registry never infers
layer/expert ownership from filenames, and lifecycle operations never attach
modules or recalibrate routing. CORE-006 adds an immutable local composition
checkpoint that references the external anchor, bundles active packages, and
persists the exact active registry identity without serializing anchor weights.
PKG-005 stores an RTR-004 result as a standalone safetensors artifact bound to
the complete active `RegistrySnapshot`, feature specification, and
recalibration report. Loading or activation never changes registry state.

## Implemented strategies

| Feature ID | Public configuration or API | Module | Status |
|---|---|---|---|
| PKG-001 | `package_format_version: 1` | `manifest.py` | Validated |
| PKG-002 | `ExpertPackage.from_directory`, `ExpertRegistry` | `package.py`, `registry.py` | Validated |
| PKG-003 | `mixture_compatibility_group`, `MixtureCompatibilityIndex` | `compatibility.py`, `registry.py` | Validated |
| PKG-004 | `ExpertRegistry.upgrade`, `ExpertRegistry.rollback`, `RegistryUpgradeChange` | `registry.py` | Validated |
| PKG-005 | `RegistryRouterVersion`, `registry_router_version_format_version: 1` | `router_versions.py` | Validated |
| PKG-007 | `payload_binding_format_version: 1`, `ExpertPayloadBinding` | `bindings.py` | Validated |
| CORE-006 | `AnchoredMoEForCausalLM.save_pretrained`, local composition `from_pretrained` | `composition.py`, `modeling.py` | Validated |
| NB-003 | `ExpertPackageExportConfig`, `ExpertPackageExportBinding`, `export_full_mlp_expert_package` | `export.py` | Validated |
| NB-004 | `CalibratedRouterPackageConfig`, `derive_calibrated_router_package` | `export.py` | Validated |

`PackageIdentity` is the portable specialization/version/content identity.
`RegistrySnapshot` and `RegistryChange` provide immutable audit records for
active state and successful transitions.
`RegistryRouterVersion` owns a defensive frozen router copy and returns a new
logical router only after exact snapshot equality. `RegistryRouterVersionError`
reports unsafe payloads, digest failures, or stale activation attempts.

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

For an upgrade from version \(p_i\) to \(p_i'\), the active registry changes
only one ordered position:

\[
P'=[p_1,\ldots,p_{i-1},p_i',p_{i+1},\ldots,p_n],
\qquad |P'|=|P|.
\]

The two identities share a specialization name and differ in immutable
specialization version. All other identities, slot index \(i\), and the
registry compatibility digest remain unchanged. Rollback is the inverse
replacement and is accepted only when the live registry snapshot still equals
the recorded upgraded snapshot.

The first successful installation permanently binds a registry instance to a
compatibility digest over:

- base repository, revision, configuration hash, and weight hashes;
- base numerical precision;
- tokenizer repository and revision;
- converted layer indices;
- router feature specification and projection hash.

This compatibility identity remains bound when the active registry becomes
empty.

For converted layers \(L\), stable package expert identifiers \(K\), and
expert tensor payloads \(F\), PKG-007 records the explicit relation

\[
B=\{(\ell,k,f):\ell\in L,\;k\in K,\;f\in F\}.
\]

For the current full-MLP format, \(B\) must be a bijection between runtime
slots \(L\times K\) and `expert_weight_files`. Every converted layer has the
same \(K\), and \(|K|=\text{expert_count}\). File paths contribute no implied
ownership.

For stable expert ID \(k\), PKG-003 defines

\[
c(k)\in\mathcal G\cup\{\bot\},
\]

where \(\mathcal G\) is the set of explicit manifest group identifiers and
\(\bot\) represents a missing declaration. Two experts are mix-compatible
exactly when

\[
\operatorname{compatible}(i,j)
=
\mathbb{1}[i\ne j\land c(i)\ne\bot\land c(i)=c(j)].
\]

There is no implicit global group: two missing declarations are not
compatible. The index is aligned by stable expert ID, so router candidate
order may be realigned without relying on filenames or package paths.

For expert-only package \(P_E\), router payload \(r\), and calibrated metadata
\(C\), NB-004 creates

\[
P_R=\operatorname{Publish}(P_E,r,C)
\]

under a new specialization version. Every expert payload digest is preserved,
while the new manifest additionally binds \(d_r=\operatorname{SHA256}(r)\),
the source manifest digest, calibration-bank identities, and the complete
CAL-005 report.

For PKG-005 router tensor state \(R\), exact registry snapshot \(S_R\), frozen
feature specification \(\Phi\), and RTR-004 report \(Q_R\), the standalone
version identity is

\[
d_V
=
\operatorname{SHA256}
\left(
\operatorname{CanonicalJSON}
\left[
S_R,\Phi,Q_R,\operatorname{SHA256Tensor}(R)
\right]
\right).
\]

The safetensors metadata records every layer/expert ownership key in global
append order plus exact tensor digests. Explicit activation succeeds only when
the supplied live snapshot equals \(S_R\), including ordered package
identities, registry hash, compatibility hash, and format version.

For anchored configuration bytes \(C\), optional generation configuration
bytes \(G\), ordered active package identities \(P\), and registry digest
\(d_{\mathrm{registry}}\), CORE-006 records

\[
S=
\left(
\operatorname{SHA256}(C),
\operatorname{SHA256}(G),
P,
d_{\mathrm{registry}},
d_{\mathrm{base\ config}}
\right).
\]

The anchor tensors are deliberately outside \(S\). Reloading resolves the
recorded anchor repository or path and revision, then requires its live
configuration digest to equal \(d_{\mathrm{base\ config}}\). Each copied
package retains its original manifest identity and internal payload hashes.

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

Composition checkpoint export and reload:

1. Require a pinned external anchor reference and configuration digest.
2. Require the active registry hash to equal the runtime hash most recently
   applied by explicit model synchronization.
3. Revalidate every active package before copying.
4. Save anchored and generation configuration into a private sibling staging
   directory.
5. Copy packages in active-registry order under content-addressed directories.
6. Strictly reload every copied package and require unchanged identity.
7. Write a canonical version-1 composition manifest containing file hashes,
   anchor pins, package order, compatibility binding, and registry hash.
8. Validate complete directory closure, then atomically rename the new
   directory into place without replacing an existing path.
9. On reload, validate the complete local artifact before loading the external
   anchor, reconstruct the exact active registry snapshot, and explicitly
   materialize anchored top-1 routing.

Atomic upgrade and rollback:

1. Require an exact active identity and a strict replacement package from the
   same specialization under a different version.
2. Revalidate every replacement payload before acquiring the registry lock.
3. Under the lock, reject a stale current identity, another active version of
   the specialization, an already slotted replacement, changed immutable
   content, or incompatible anchor/router features.
4. Construct and validate the complete before/after audit record.
5. Replace the active descriptor and its slot key as one lock-protected
   transition without changing any unrelated slot.
6. For rollback, require the original upgrade record, exact old package, and
   exact upgraded registry snapshot; revalidate the old package before
   reversing the replacement.
7. Keep model synchronization separate and explicit.

Payload-binding validation:

1. Legacy package-format-v1 manifests may omit bindings and remain usable for
   registry lifecycle operations.
2. If either binding key is declared, require both
   `payload_binding_format_version` and `payload_bindings`.
3. Parse only the exact fields `layer_index`, `expert_id`, and `weight_file`.
4. Reject duplicate slots, duplicate files, unconverted layers, and unsafe or
   undeclared paths.
5. Require exact file coverage, `expert_count` experts per layer, and the same
   expert-ID set at every converted layer.
6. Runtime materialization requires the block and fails explicitly when a
   legacy package does not provide it.

Mixture-compatibility indexing:

1. Capture active packages in stable registry order.
2. Require the requested converted layer in every active package.
3. Parse explicit payload bindings and collect the layer's stable expert IDs.
4. Assign each expert its package-level `mixture_compatibility_group`, retaining
   a missing declaration as `None`.
5. Reject duplicate expert IDs across active packages.
6. Recheck the registry snapshot before returning the immutable index.

Expert-only package export:

1. Validate a complete rectangular set of `(layer, expert_id, expert)`
   bindings.
2. Reject repeated expert objects and shared tensor storage across slots.
3. Derive converted layers, expert count, package structure, and one common
   floating weight dtype from the live experts.
4. Write deterministic safetensors filenames in a private sibling staging
   directory.
5. Compute every payload digest and construct the explicit PKG-007 bijection.
6. Create a router-free manifest from frozen checkpoint, runtime, dataset, and
   training metadata.
7. Strictly reload the staged package, publish it without replacing an
   existing path, and strictly reload the published result.
8. Remove staged files after every failure.

Calibrated router-package derivation:

1. Require a strict, binding-complete, router-free NB-003 package containing
   only its declared expert payloads.
2. Require a new specialization version and revalidate live source files.
3. Require fitting and calibration banks with disjoint global row IDs and
   exact source feature/layer/expert compatibility.
4. Require frozen finite calibrated heads whose keys, dtype, temperature, and
   threshold exactly match both payload bindings and the CAL-005 report.
5. Copy source expert files into staging and verify their original digests.
6. Serialize the calibrated router and record its digest.
7. Construct a manifest containing source identity, bank identities,
   provenance, ridge configuration, calibration report, and thresholds.
8. Strictly validate and publish the new package without modifying or
   overwriting either source or destination content.

Registry-specific router versions:

1. Require a non-empty frozen CPU logical router, an active registry snapshot,
   exact feature specification, and matching RTR-004 report.
2. Convert the logical buffers to storage-independent affine heads while
   preserving global ownership order, temperatures, and thresholds.
3. Record the full registry snapshot, feature space, recalibration report,
   tensor names, dtypes, and exact tensor digests in canonical JSON metadata.
4. Atomically write only finite floating-point tensors to safetensors without
   executable Python content.
5. On load, require the exact schema and tensor-key closure, recheck every
   digest, reconstruct frozen heads, and require regenerated canonical metadata
   to equal the stored envelope.
6. On explicit activation, compare the complete live snapshot before returning
   a fresh storage-independent logical router.
7. Never install, unload, upgrade, roll back, synchronize, or recalibrate a
   model as a side effect.

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
- Upgrade preserves active count and transfers the superseded package's slot
  to a different immutable version of the same specialization.
- Rollback requires the exact old package and refuses to overwrite any
  intervening registry change.
- Inactive package paths are not retained secretly; callers keep rollback
  artifacts available and immutable.
- Registry hashes exclude local paths and include active order.
- The compatibility binding persists across an empty active registry.
- Registry transitions are in-memory, lock-protected, and atomic with respect
  to registry readers.
- Installation never triggers router fitting, recalibration, or model mutation.
- Package format v1 remains backward compatible: binding-free manifests keep
  their existing serialization and identity behavior.
- Declared bindings are validated during manifest construction and again
  before runtime materialization.
- Full-MLP payloads own exactly one `(layer_index, expert_id)` slot.
- Expert IDs, not filenames, are matched against serialized router-head keys.
- Compatibility groups never alter package installation eligibility.
- Missing compatibility metadata means non-mixable, not a shared default.
- A compatibility index requires explicit payload bindings and exact active
  layer coverage.
- Active packages may not repeat an expert ID within one layer index.
- Candidate expert order may be realigned by ID, but the candidate set must
  exactly match the compatibility index.
- NB-003 export never serializes the anchor, optimizer, scheduler, or router.
- Export destinations must be new; immutable package versions are never
  overwritten.
- Exported bindings form a complete layer-by-expert rectangle with independent
  expert objects and storage.
- All floating tensors in one exported full-MLP package use one declared
  weight dtype.
- NB-003 packages are expert-only and cannot configure anchored top-1 routing
  until NB-004 supplies calibrated router heads through a later artifact.
- NB-004 never mutates or reuses the source specialization version.
- Derived expert payload bytes and hashes exactly match the NB-003 source.
- Fitting and calibration row IDs are globally disjoint.
- Calibrated heads form an exact bijection with source payload bindings and
  remain frozen after serialization and reload.
- Router feature size and dtype match the package feature/hidden contract.
- The manifest records source identity, bank identities, calibration report,
  thresholds, runtime, and router payload hash without embedding prompts.
- PKG-005 artifacts require an active snapshot and exact agreement between its
  registry/compatibility hashes and the RTR-004 report.
- Registry-router tensors are finite CPU values with embedded dtype-aware
  SHA-256 digests and exact tensor-key closure.
- Every activation returns new frozen storage and rejects stale, reordered,
  removed, upgraded, or otherwise different registry snapshots.
- Saving, loading, and activation never mutate `ExpertRegistry` or a model.
- CORE-006 composition destinations are new and immutable.
- Composition checkpoints contain no anchor tensors and require an external
  anchor reference plus live configuration-hash agreement.
- Active package bytes, identities, ordering, compatibility binding, and
  registry hash survive a round trip.
- Saving rejects stale registry/runtime state rather than synchronizing
  implicitly.
- Loading a saved composition restores its runtime explicitly; ordinary
  install, unload, upgrade, and rollback operations remain registry-only.
- Symbolic links, untracked files, unsafe package paths, modified
  configurations, and modified package payloads are rejected.

The registry validates compatibility between packages. CORE-006 also checks
the live anchor configuration digest; exact anchor-weight and runtime-version
enforcement remain future compatibility work.

## Hugging Face integration

Repository and revision fields use Hugging Face Hub naming conventions.
Packages are independently uploadable because manifests contain package-local
paths and never duplicate an anchor implicitly.

Registry router versions are standalone safetensors artifacts rather than
`PreTrainedModel` checkpoints or package payloads. They can be transferred as
ordinary Hub files, but PKG-005 performs no network access and requires the
caller to validate and activate the exact local registry snapshot explicitly.

`ExpertPackage.from_directory()` currently supports local directories only.
`from_pretrained()` and Hub download behavior are deferred to PKG-006.
`ExpertPackage.payload_bindings` exposes the validated materialization
contract. The inference subpackage consumes it only during an explicit
registry-to-model synchronization operation.

`export_full_mlp_expert_package()` is the explicit post-training boundary. Its
configuration records Hugging Face repository and revision identities,
tokenizer identity, runtime versions, and dataset provenance while package
payloads contain only safe expert tensors. Production callers provide actual
checkpoint shard hashes; deterministic in-memory examples use a named
state-dict fingerprint.

`derive_calibrated_router_package()` is the explicit post-calibration boundary.
It consumes CAL-001/CAL-002 fitting and held-out banks plus the CAL-005 frozen
router/report pair, then publishes an installable anchored-top-1 package. Bank
tensors remain separate reproducibility artifacts; the manifest records their
content identities and dataset provenance.

`AnchoredMoEForCausalLM.install()`, `.unload()`, `.upgrade()`, and `.rollback()`
are thin public delegates to the owned `ExpertRegistry`; none changes model
routes. Calling `.synchronize_registry()` separately applies the active
registry through PKG-007. This preserves the same lifecycle semantics behind
the higher-level Hugging Face facade.

`AnchoredMoEForCausalLM.save_pretrained()` publishes the CORE-006 local
composition format. Unlike a conventional full-weight Hugging Face checkpoint,
it does not write the anchor state dict. A matching
`AnchoredMoEForCausalLM.from_pretrained()` call detects the local composition,
validates it, loads the recorded anchor through `AutoModelForCausalLM`, rebuilds
the registry, and restores the saved anchored-top-1 runtime. The optional
`anchor_model_name_or_path` argument relocates an equivalent local anchor;
configuration hashes still have to match. Hub transfer remains PKG-006.

`ExpertRegistry.mixture_compatibility(layer_index=...)` exposes PKG-003 without
loading expert or router tensors. The resulting index is consumed by POL-005
and can be passed through INF-002 for standalone compatibility-aware sparse
execution; model-level top-2 attachment remains a separate inference feature.

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

replacement = ExpertPackage.from_directory("./code-experts-2")
upgrade = registry.upgrade(package.identity, replacement)
assert upgrade.after.active_packages[0] == replacement.identity

rollback = registry.rollback(upgrade, package)
assert rollback.after == upgrade.before
```

A synchronized model composition round-trips through familiar Hugging Face
method names while keeping anchor weights separate:

```python
model.install(ExpertPackage.from_directory("./code-experts"))
model.synchronize_registry()
model.save_pretrained("./code-composition")

reloaded = AnchoredMoEForCausalLM.from_pretrained("./code-composition")
assert (
    reloaded.synchronized_registry_sha256
    == model.registry.active_registry_sha256
)
```

Compatibility metadata is explicit and layer-specific:

```python
compatibility = registry.mixture_compatibility(layer_index=18)
assert compatibility.are_compatible("code:0", "code:1")
assert not compatibility.are_compatible("code:0", "math:0")
```

An RTR-004 result is persisted independently of package lifecycle:

```python
from anchored_moe import RegistryRouterVersion

version = RegistryRouterVersion(
    recalibrated_router,
    registry_snapshot=registry.snapshot(),
    feature_specification=router_feature_specification,
    recalibration_report=recalibration_report,
)
version.save_safetensors("./registry-router.safetensors")

loaded = RegistryRouterVersion.from_safetensors(
    "./registry-router.safetensors"
)
router = loaded.activate(registry.snapshot())
```

Expert-only export uses one immutable provenance object and explicit live
expert bindings:

```python
from anchored_moe import (
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    export_full_mlp_expert_package,
)

package = export_full_mlp_expert_package(
    "./code-experts-1.0.0",
    bindings=[
        ExpertPackageExportBinding(18, "code:0", layer_18_expert),
        ExpertPackageExportBinding(19, "code:0", layer_19_expert),
    ],
    config=ExpertPackageExportConfig(
        specialization_name="code",
        specialization_version="1.0.0",
        base_model_repository="Qwen/Qwen3-4B",
        base_model_revision="pinned-revision",
        base_config_sha256=config_digest,
        base_weight_sha256=checkpoint_shard_digests,
        tokenizer_repository="Qwen/Qwen3-4B",
        tokenizer_revision="pinned-tokenizer-revision",
        base_model_dtype="bfloat16",
        runtime_versions=runtime_versions,
        training_dataset=dataset_provenance,
        router_feature_specification=router_feature_specification,
        expert_parameterization_config={
            "initialization": "anchor_copy",
            "training_strategy": "forced_path",
        },
    ),
)
```

The exporter owns `payload_binding_format_version`, `payload_bindings`,
converted layers, expert count, package structure, weight dtype, filenames,
and hashes. Callers cannot override those derived fields.

Router derivation uses a new immutable version:

```python
from anchored_moe import (
    CalibratedRouterPackageConfig,
    derive_calibrated_router_package,
)

routed_package = derive_calibrated_router_package(
    package,
    "./code-experts-1.0.0-router-1",
    calibrated_router=calibrated_router,
    calibration_report=calibration_report,
    fitting_feature_bank=fitting_features,
    fitting_utility_labels=fitting_utilities,
    calibration_feature_bank=held_out_features,
    calibration_utility_labels=held_out_utilities,
    config=CalibratedRouterPackageConfig(
        specialization_version="1.0.0+router.1",
        calibration_dataset_version="router-data-v1",
        ridge_regularization=1e-3,
        runtime_versions=runtime_versions,
    ),
)
```

An attachable full-MLP manifest includes:

```json
{
  "expert_parameterization_config": {
    "initialization": "anchor_copy",
    "payload_binding_format_version": 1,
    "payload_bindings": [
      {
        "layer_index": 18,
        "expert_id": "code:0",
        "weight_file": "experts/layer-18-code-0.safetensors"
      }
    ]
  }
}
```

See `examples/verify_registry_lifecycle.py` for an executable registry-only
lifecycle example, `examples/verify_compatibility_top2.py` for PKG-003/POL-005
selection and INF-002 sparse execution, and
`examples/verify_package_runtime.py` for the complete
model-facade lifecycle plus CORE-006 save/reload round trip.
`examples/verify_registry_router_versions.py` exercises deterministic RTR-004
fitting, PKG-005 safe round trip, and exact explicit activation.
`examples/export_reload_lifecycle.py` and
`notebooks/06_export_and_reload.ipynb` exercise the complete NB-007 workflow:
local composition export/reload, explicit unload and reinsertion
synchronization, slot-preserving upgrade, and exact rollback.
`examples/train_expert_package.py` and
`notebooks/02_train_expert_package.ipynb` exercise the NB-003 training/export
boundary. `examples/fit_router.py` and
`notebooks/03_fit_and_calibrate_router.ipynb` exercise NB-004 derivation and
materialization.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines exact package compatibility, stable ordering, auditable
  installation, pure unloading, immutable upgrades, and explicit rollback
  semantics.
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

PKG-001, PKG-002, PKG-003, PKG-004, PKG-005, PKG-007, CORE-006, NB-003 export,
NB-004 derivation, and the NB-007 lifecycle workflow are framework
infrastructure and do not derive a training or optimization algorithm from a
paper. NB-004
packages the already documented CAL-004 ridge and CAL-005
temperature/threshold results without changing their algorithms.

## Tests and known limitations

Acceptance tests cover deterministic manifest serialization, payload tampering,
strict directory closure, malformed formats, portable identities, deterministic
empty and active registry hashes, stable installation order, live-file
revalidation, duplicate activation, exact unloading, reinsertion restoration,
compatibility rejection, immutable-version conflicts, non-strict descriptor
rejection, atomic failure behavior, slot-preserving upgrades, exact rollback,
stale rollback rejection, upgrade and rollback payload revalidation, explicit
synchronization, legacy binding-free manifests, explicit binding coverage and
uniqueness, missing and shared mixture groups, candidate realignment, stale
candidate sets, duplicate cross-package expert IDs, layer-coverage failures,
deterministic staged export, expert-only payload closure, dtype derivation,
topology and alias rejection, staging cleanup, immutable routed derivation,
exact expert-copy digests, data/head/report compatibility, router hashing,
composition file and package tampering, stale-runtime rejection, exact routed
logit and registry restoration, documentation, examples, notebooks, and wheel
content. NB-007 additionally executes composition reload followed by
unsynchronized and synchronized removal, reinsertion, upgrade, and rollback;
it checks exact logits, registry hashes, slot restoration, and complete
parameter freezing.

PKG-005 tests additionally cover exact registry/compatibility/report binding,
canonical metadata round trips, tensor digest tampering, tensor-key closure,
feature-size rejection, source-storage independence, fresh frozen activation,
stale removal rejection, exact reinsertion restoration, public documentation,
and the executable offline example.

PKG-007 materializes the supported binding contract in the inference
subpackage. NB-003 does not fit or serialize a router, save optimizer state, or
generate replay teacher labels. NB-004 supports one package-local independent
affine router and anchored top-1 policy. PKG-005 versions are standalone and
are not yet embedded in CORE-006 composition checkpoints. Persistent registry
files, cross-process locking, exact
live-model checkpoint hashing, runtime-version enforcement, rollback across an
intervening registry change, historical inactive-slot persistence, standalone
registry lifecycle persistence, optional anchor bundling, Hub transfer, and
distributed lifecycle operations remain unsupported. CORE-006 restores only the active
registry snapshot; it does not retain inactive package content or rollback
records. Consequently, NB-007 supplies the prior immutable package explicitly
when rolling back after reload instead of implying that the composition
checkpoint persisted inactive history.
