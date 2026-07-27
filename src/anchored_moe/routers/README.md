# Routers

## Scope

The routers subpackage owns frozen feature transformations and independently
owned score heads. RTR-001 defines the initial pre-MLP feature space. RTR-002
maps those features to calibrated scalar evidence for an expert at one layer.

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

`RouterFeatureExtractor` and `RouterHead` are model-family-neutral extension
interfaces. `IdentityRouterFeatureExtractor`, `AffineRouterHead`, and
`AppendOnlyRouterHeads` provide the first concrete implementations.

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

The anchor score remains defined by the architecture as
\(g_{\ell,0}(h)=0\), but this subpackage does not yet compare expert scores
against it.

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
- Router payloads contain no executable Python content.
- Router tensor payloads are finite, floating-point, and schema-validated.
- Existing payloads are not overwritten unless `overwrite=True`.

The append-only guarantee covers the act of appending. A caller that retains a
mutable head reference can still train that head explicitly. Package
installation will later freeze old heads mechanically.

## Hugging Face integration

The implementations are ordinary `torch.nn.Module` objects and accept hidden
states produced by Transformers decoder layers. They do not patch a model
family or alter generation behavior. `RoutedMLP` integration is intentionally
deferred until a selection policy exists, so the current forced-anchor path
remains exact.

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

## Configuration and examples

```python
from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
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
scores = heads(features, layer_index=18)
```

See `examples/verify_affine_router.py` for deterministic scoring, append
invariance, and safe serialization without model downloads.

`AnchoredMoEConfig` continues to reject non-anchor selection policies. The
router objects are configured directly until POL-001 integrates their scores
with model execution; this prevents a declared router from silently affecting
the exact anchor fallback.

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

RTR-001 and RTR-002 are framework infrastructure and do not derive an
optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover identity and gradient preservation, feature metadata,
the calibrated score equation, tensor shapes, neutral initialization,
trainability, append invariance, ownership uniqueness, deterministic layer
ordering, exact safetensors round trips, overwrite protection, malformed
metadata and tensor rejection, finite-value checks, and expert-package
manifest compatibility.

The current implementation has no learned projection, feature cache, fitting
algorithm, head removal, model attachment, or capacity behavior. Standalone
anchor comparison and top-1 selection are implemented by POL-001; model
dispatch remains a follow-up. The other limitations remain tracked as
CAL-001/CAL-004, PKG-002, and BAL-001.
