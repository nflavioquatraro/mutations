# Routers

## Scope

The routers subpackage owns frozen feature transformations, independently
owned score heads, and registry-level logical score banks. RTR-001 defines the
initial pre-MLP feature space. RTR-002 maps those features to calibrated scalar
evidence for an expert at one layer. RTR-003 compiles those package-owned heads
into frozen layer-local matrices without changing the package wire format.
RTR-004 creates a new frozen logical router by jointly recalibrating all
installed affine columns against aligned anchor-relative utility replay data.

This subpackage does not choose between the anchor and experts, execute an MLP,
combine expert outputs, fit calibration parameters, cache features, enforce
capacity, or install and remove packages. Those responsibilities remain with
selection, modeling, combination, calibration, capacity, and registry
features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| RTR-001 | `feature_type: identity`, `input: pre_mlp_hidden_state` | `base.py`, `features.py` | Validated |
| RTR-002 | `router_architecture: independent_affine_heads` | `base.py`, `heads.py` | Validated |
| RTR-003 | `router_architecture: logical_append_only_columns` | `base.py`, `logical.py` | Validated |
| RTR-004 | `recalibration: router_only_global` | `calibration/global_recalibration.py` | Validated |

`RouterFeatureExtractor`, `RouterHead`, and `RouterScoreBank` are
model-family-neutral extension interfaces. `IdentityRouterFeatureExtractor`,
`AffineRouterHead`, `AppendOnlyRouterHeads`, and `LogicalAppendOnlyRouter`
provide the concrete implementations. `compile_logical_router()` makes the
package-head-to-registry-runtime boundary explicit.

## Mathematical formulation

At decoder layer \(\ell\), let the pre-MLP hidden states be

\[
h_\ell\in\mathbb{R}^{B\times S\times d}.
\]

The initial frozen feature transformation is the identity:

\[
\phi_\ell(h_\ell)=h_\ell,
\qquad
\phi_\ell(h_\ell)\in\mathbb{R}^{B\times S\times d}.
\]

For expert \(k\), one independently owned affine head computes

\[
s_{\ell,k}(h)
=
w_{\ell,k}^{\top}\phi_\ell(h)+b_{\ell,k},
\]

\[
g_{\ell,k}(h)
=
\frac{s_{\ell,k}(h)}{T_{\ell,k}}-\tau_{\ell,k},
\qquad T_{\ell,k}>0.
\]

The result has shape

\[
g_{\ell,k}(h)\in\mathbb{R}^{B\times S}.
\]

When a layer has \(K_\ell\) registered heads, the append-only bank stacks
scores in deterministic append order:

\[
G_\ell(h)
=
\left[g_{\ell,1}(h),\ldots,g_{\ell,K_\ell}(h)\right]
\in\mathbb{R}^{B\times S\times K_\ell}.
\]

RTR-003 stores the same layer-local coefficients as

\[
W_\ell =
\begin{bmatrix}
w_{\ell,1}^{\top}\\
\vdots\\
w_{\ell,K_\ell}^{\top}
\end{bmatrix}
\in\mathbb{R}^{K_\ell\times d},
\]

\[
b_\ell,T_\ell,\tau_\ell\in\mathbb{R}^{K_\ell},
\qquad
G_\ell(h)
=
\frac{\phi_\ell(h)W_\ell^\top+b_\ell}{T_\ell}-\tau_\ell.
\]

The division and subtraction are elementwise along the final expert
dimension. The logical router registers these tensors as frozen buffers rather
than trainable parameters. Its result has the same
\(B\times S\times K_\ell\) shape, expert IDs, and calibrated values as the
independent source heads.

The anchor score remains defined by the architecture as
\(g_{\ell,0}(h)=0\), but this subpackage does not yet compare expert scores
against it.

RTR-004 uses the fixed-zero anchor during offline fitting. For replay utility
matrix \(A_\ell\), it selects the anchor-first best-route class from
\([0,A_\ell]\) and minimizes cross-entropy over deployed logits
\([0,G_\ell]\), plus normalized proximal drift from the source affine matrix.
This changes all specialist score functions for the bound registry and
therefore deliberately does not preserve RTR-003 append invariance.

## Algorithms

Identity feature extraction:

1. Require a floating-point tensor with the configured final dimension.
2. Return the same tensor object without allocation, projection, or detachment.
3. Expose an immutable JSON-compatible feature specification for package
   compatibility checks.

Affine scoring:

1. Validate the feature tensor and final dimension.
2. compute one affine scalar per feature vector;
3. divide by the positive immutable temperature;
4. subtract the immutable activation threshold.

Append-only registration:

1. Require an `AffineRouterHead` with the bank's feature size.
2. Reject an existing `(layer_index, expert_id)` ownership key.
3. append the module without replacing or rewriting any prior module;
4. filter by layer and score in original append order.

Safetensors export:

1. Reject empty banks, unsafe paths, symbolic links, and accidental
   overwrites.
2. Require finite floating-point affine tensors.
3. detach, sort, make contiguous, and move tensors to CPU;
4. encode format, feature size, ownership, append order, temperatures, and
   thresholds in one canonical JSON metadata envelope;
5. write beside the destination and replace it atomically.

Safetensors loading:

1. Reject non-files, symbolic links, and incorrect suffixes.
2. Parse and strictly validate the fixed metadata schema.
3. reconstruct only known affine modules in recorded append order;
4. validate tensor keys, shapes, dtypes, and finite values;
5. load tensors strictly and apply the requested trainability.

Logical compilation:

1. Require an RTR-002 `AppendOnlyRouterHeads` source.
2. Visit heads in global append order and validate feature width, ownership,
   dtype, device, and finite tensor values.
3. Clone each affine row and calibration scalar so compiled buffers never
   alias package-owned parameters.
4. Group copied rows by decoder layer while retaining globally filtered expert
   order.
5. Evaluate every row with a stable `[1, d]` affine reduction and stack the
   results. This deliberately prevents a wider GEMM selected after append from
   changing incumbent floating-point results.

Logical append:

1. Validate the complete candidate before changing registered state.
2. Concatenate its copied row and calibration scalars to the target layer.
3. Append its ownership key only after tensor registration succeeds.
4. Leave every incumbent value, source head, and source storage unchanged.

Global recalibration:

1. Convert a frozen logical router to storage-independent affine heads.
2. Validate exact replay layer and expert-column alignment.
3. Jointly optimize every layer's weights and biases while holding temperature
   and threshold fixed.
4. Retain the best deterministic full-batch objective, including the source.
5. Compile the accepted copied heads into a new buffer-only logical router.
6. Bind the report to one registry and compatibility hash without activating
   it.

## Assumptions and invariants

- Router input is the pre-MLP hidden state.
- The initial feature space is exactly the model hidden space.
- Identity extraction has no parameters and preserves tensor storage, dtype,
  device, and caller-controlled gradient flow.
- Feature extraction is separate from caching; cached features should be
  detached by CAL-001 rather than by the runtime transform.
- Every head is owned by exactly one `(layer_index, expert_id)` pair.
- New affine heads begin with zero weight and bias, producing neutral raw
  evidence relative to the fixed-zero anchor score.
- Temperature is finite and positive; threshold is finite.
- Features and heads must share a dtype and device; all heads in one bank must
  also share them.
- Appending cannot replace or remove an existing head and does not mutate its
  parameters.
- Layer-local score columns follow stable global append order.
- Both RTR-002 and RTR-003 implement `RouterScoreBank`; routed inference does
  not depend on their storage representation.
- Logical-router buffers never alias their source heads and contain no
  trainable parameters.
- Gradients through logical-router input features are preserved.
- Appending a logical column preserves incumbent scores bit-for-bit for
  identical inputs, dtype, device, and backend.
- Global recalibration is explicit, source-immutable, CPU deterministic, and
  returns no trainable parameters.
- A recalibrated router is meaningful only for the registry hash and exact
  expert order recorded by its report.
- Logical compilation rejects non-finite state even before serialization.
- Router payloads contain no executable Python content.
- Router tensor payloads are finite, floating-point, and schema-validated.
- Existing payloads are not overwritten unless `overwrite=True`.

The RTR-002 append-only guarantee covers the act of appending. A caller that
retains a mutable head reference can still train that head explicitly. RTR-003
instead copies and freezes all compiled state, so later source-head training
does not change deployed scores.

## Hugging Face integration

The implementations are ordinary `torch.nn.Module` objects and accept hidden
states produced by Transformers decoder layers. `RoutedMLP` depends on the
shared `RouterScoreBank` contract and therefore accepts either RTR-002 heads or
an RTR-003 logical router without changing selection or dispatch behavior.
Explicit registry synchronization validates package-owned RTR-002 payloads,
then compiles each complete layer into RTR-003 before replacing routed runtime
references. The forced-anchor path remains exact.

Router safetensors files can be listed under `router_files` in an
`ExpertPackageManifest`. The matching manifest values are:

```yaml
router_architecture: independent_affine_heads
router_feature_specification:
  feature_type: identity
  input: pre_mlp_hidden_state
  output_size: 4096
  version: 1
```

`logical_append_only_columns` describes the in-memory registry runtime, not a
package manifest. PKG-005 serializes a recalibrated logical router as a
standalone registry-specific version; it does not change the package format.

## Configuration and examples

```python
from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    compile_logical_router,
    recalibrate_global_router,
)

features = IdentityRouterFeatureExtractor(feature_size=4096)(hidden_states)

heads = AppendOnlyRouterHeads(feature_size=4096)
heads.append(
    AffineRouterHead(
        4096,
        layer_index=18,
        expert_id="code:0",
        temperature=1.0,
        threshold=0.0,
    )
)
logical_router = compile_logical_router(heads)
scores = logical_router(features, layer_index=18)

recalibrated, report = recalibrate_global_router(
    logical_router,
    replay_features,
    replay_utilities,
    registry_sha256=registry.snapshot().registry_sha256,
    compatibility_sha256=registry.snapshot().compatibility_sha256,
)
```

See `examples/verify_affine_router.py` for RTR-002 deterministic scoring,
append invariance, and safe serialization. See
`examples/verify_logical_router.py` for RTR-003 compilation, exact incumbent
score preservation, frozen state, and routed execution. Neither example
downloads model weights.
See `examples/verify_registry_router_versions.py` for RTR-004 fitting and
PKG-005 persistence without model downloads.

`AnchoredMoEConfig` continues to reject non-anchor selection policies. INF-001
allows router objects to be attached explicitly to one `RoutedMLP`; package
installation and complete model composition are still required before a
declarative model configuration can be executable.

## References

- [PyTorch `nn.Linear` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
  defines the affine transformation and tensor-shape behavior.
- [PyTorch module documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  defines parameter registration, freezing, and module state.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor format and metadata envelope.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines frozen router features, independent affine heads, calibrated scores,
  fixed-zero anchor evidence, and append-only installation.

RTR-001, RTR-002, RTR-003, and RTR-004 are framework infrastructure and do not
derive an optimization algorithm from a paper. RTR-004 implements the
blueprint's explicit router-only mode using a documented project-specific
anchored classification objective.

## Tests and known limitations

Acceptance tests cover identity and gradient preservation, feature metadata,
the calibrated score equation, tensor shapes, neutral initialization,
trainability, append invariance, ownership uniqueness, deterministic layer
ordering, exact safetensors round trips, overwrite protection, malformed
metadata and tensor rejection, finite-value checks, expert-package manifest
compatibility, logical matrix shapes, source-storage independence, frozen
compiled state, dtype migration, exact RTR-002 score parity, exact incumbent
score preservation, and `RoutedMLP` plus package-synchronization integration.

The current router implementation has no learned projection or head removal.
RTR-003 uses shape-stable per-column affine evaluation to guarantee exact
append invariance; it does not yet fuse all rows into one GEMM. It has no
independent on-disk format and cannot be trained in place. RTR-004 instead
optimizes a copied affine representation and returns a new logical router;
PKG-005 owns registry-specific persistence.
CAL-001 provides feature caches and
CAL-004 fits utility-target affine heads in the calibration subpackage.
CAL-005 calibrates their temperatures and thresholds without mutating affine
parameters. CAL-006 independently refines newly calibrated affine parameters
against scores from an immutable incumbent `RouterScoreBank`; it returns
package-owned RTR-002 heads and does not mutate the RTR-003 logical router.
Standalone anchor comparison and top-1 selection are implemented by POL-001;
INF-001 explicitly attaches them for routed-layer dispatch,
PKG-007 validates serialized router ownership against expert payload bindings,
and BAL-001 handles supplied availability masks. Learned projections, capacity
estimation, top-2 routing, and the other limitations remain tracked as later
calibration, runtime-composition, and capacity features.
