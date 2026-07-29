# Routers

## Scope

The routers subpackage owns frozen feature transformations, independently
owned score heads, and registry-level logical score banks. RTR-001 defines the
initial pre-MLP feature space. RTR-002 maps those features to calibrated scalar
evidence for an expert at one layer. RTR-003 compiles those package-owned heads
into frozen layer-local matrices without changing the package wire format.
RTR-004 creates a new frozen logical router by jointly recalibrating all
installed affine columns against aligned anchor-relative utility replay data.
RTR-005 composes one package-level logical router with one logical local router
per package and evaluates local scores only for token rows already assigned to
that package.
RTR-007 distills the anchored hard assignments of any immutable teacher score
bank into a deterministic affine student and freezes it for deployment.
RTR-008 applies an explicit switching margin to adjacent-layer anchored top-1
decisions without retaining request state inside the router or model.

Except for RTR-008's explicit stabilization of a caller-supplied top-1
decision, this subpackage does not choose between the anchor and experts. It
does not execute an MLP, combine expert outputs, fit calibration parameters,
cache features, enforce capacity, or install and remove packages. Those
responsibilities remain with selection, modeling, combination, calibration,
capacity, and registry features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| RTR-001 | `feature_type: identity`, `input: pre_mlp_hidden_state` | `base.py`, `features.py` | Validated |
| RTR-002 | `router_architecture: independent_affine_heads` | `base.py`, `heads.py` | Validated |
| RTR-003 | `router_architecture: logical_append_only_columns` | `base.py`, `logical.py` | Validated |
| RTR-004 | `recalibration: router_only_global` | `calibration/global_recalibration.py` | Validated |
| RTR-005 | `router_architecture: hierarchical_package_local` | `hierarchical.py` | Validated |
| RTR-007 | `strategy: hard_route_distillation` | `stabilized.py` | Validated |
| RTR-008 | `strategy: cross_layer_top1_stickiness`, `switch_margin` | `stickiness.py` | Validated |

`RouterFeatureExtractor`, `RouterHead`, and `RouterScoreBank` are
model-family-neutral extension interfaces. `IdentityRouterFeatureExtractor`,
`AffineRouterHead`, `AppendOnlyRouterHeads`, and `LogicalAppendOnlyRouter`
provide the concrete implementations. `compile_logical_router()` makes the
package-head-to-registry-runtime boundary explicit.
`HierarchicalRouterScoreBank` composes copied `LogicalAppendOnlyRouter`
instances and returns `HierarchicalLocalScores` groups whose flattened row
indices make selected-package-only scoring auditable.

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

For RTR-005, let \(P\) be the number of installed packages and let package
\(p\) own \(K_p\ge2\) local experts. The package bank returns

\[
Q_\ell(h)
=
\left[q_{\ell,1}(h),\ldots,q_{\ell,P}(h)\right].
\]

POL-004 supplies a selected package index \(c_t\in\{0,\ldots,P\}\) for each
token, where zero is the anchor. RTR-005 then evaluates local bank \(p\) only
on the flattened selected row set

\[
I_{\ell,p}=\{t:c_t=p\},
\qquad
L_{\ell,p}
=
\left[
g_{\ell,p,1}(h_t),\ldots,g_{\ell,p,K_p}(h_t)
\right]_{t\in I_{\ell,p}}.
\]

Thus package scoring costs \(P\) affine columns per token, while local scoring
costs only the \(K_{c_t}\) columns belonging to the selected package. Anchor
rows belong to no \(I_{\ell,p}\) and execute no local router. RTR-005 returns
strictly increasing flattened row indices with every \(L_{\ell,p}\), so the
policy can reconstruct the original token-leading shape exactly.

RTR-004 uses the fixed-zero anchor during offline fitting. For replay utility
matrix \(A_\ell\), it selects the anchor-first best-route class from
\([0,A_\ell]\) and minimizes cross-entropy over deployed logits
\([0,G_\ell]\), plus normalized proximal drift from the source affine matrix.
This changes all specialist score functions for the bound registry and
therefore deliberately does not preserve RTR-003 append invariance.

RTR-007 treats an evaluation-mode `RouterScoreBank` as a teacher. For
specialist scores

\[
G_\ell^{\mathrm{teacher}}\in\mathbb{R}^{N\times K_\ell},
\]

the fixed-zero anchor is prepended and hard assignments are

\[
z_\ell
=
\operatorname*{argmax}
\left[
\mathbf{0},
G_\ell^{\mathrm{teacher}}
\right].
\]

The anchor is class zero, so exact ties choose the anchor before experts; ties
between experts follow stable layer-local expert order. The affine student
uses cached features \(X_\ell\in\mathbb{R}^{N\times d}\):

\[
G_\ell^{\mathrm{student}}=X_\ell W_\ell^\top+b_\ell,
\]

\[
\mathcal{L}_{\mathrm{distill}}
=
\operatorname{CE}
\left(
\left[\mathbf{0},G_\ell^{\mathrm{student}}\right],
z_\ell
\right)
+
\lambda
\frac{\lVert W_\ell\rVert_2^2+\lVert b_\ell\rVert_2^2}
{K_\ell d+K_\ell}.
\]

This initial strategy distills assignments, not teacher score magnitudes.
Its acceptance metric is hard-route agreement on the exact reference bank
rather than score equality.

RTR-008 consumes ordinary anchored top-1 baseline route \(b_\ell\), the
previous route \(p\) aligned by expert ID to the current layer, current
candidate scores \(s_\ell(c)\), and a non-negative switch margin \(m\). The
anchor remains candidate zero with \(s_\ell(0)=0\). For \(m>0\),

\[
r_\ell =
\begin{cases}
b_\ell, & p\text{ is unavailable},\\
b_\ell, & s_\ell(b_\ell)-s_\ell(p)>m,\\
p, & \text{otherwise}.
\end{cases}
\]

The comparison uses current-layer evidence for both candidates; a stale
previous-layer score is never compared across layer-specific calibration
spaces. The strict inequality creates a closed dead band: an exact margin tie
retains the incumbent. The explicit \(m=0\) mode bypasses stabilization and is
exactly POL-001, including anchor-first and append-order ties.

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

Hierarchical package/local scoring:

1. Require a non-empty RTR-003 package router and an exact local-router mapping.
2. Validate one package order at every covered layer, exact local layer
   coverage, at least two stable local experts per package, globally unique
   expert IDs, and one feature width, dtype, and device.
3. Copy every source bank through independent affine heads into new buffer-only
   logical routers.
4. Score every package for all token rows.
5. After POL-004 selects package indices, flatten features and gather the rows
   assigned to each non-anchor package in package order.
6. Score only the matching local router and return finite scores plus strictly
   increasing flattened row indices.

Global recalibration:

1. Convert a frozen logical router to storage-independent affine heads.
2. Validate exact replay layer and expert-column alignment.
3. Jointly optimize every layer's weights and biases while holding temperature
   and threshold fixed.
4. Retain the best deterministic full-batch objective, including the source.
5. Compile the accepted copied heads into a new buffer-only logical router.
6. Bind the report to one registry and compatibility hash without activating
   it.

Hard-route distillation:

1. Require a frozen evaluation-mode teacher and an immutable CAL-001 feature
   bank with the same feature width.
2. Evaluate finite CPU teacher scores in each declared layer and prepend the
   permanent zero anchor before `argmax`.
3. Fit zero-initialized affine student columns independently by layer using
   seed-free full-batch CPU float64 Adam.
4. Retain the best finite regularized objective, then re-evaluate it in the
   feature bank's deployed dtype.
5. Fall back to the zero student if dtype conversion worsens the initial total
   objective.
6. Compile copied affine columns into a buffer-only RTR-003 router and record
   score, assignment, feature-bank, configuration, loss, and agreement
   metadata in an immutable report.

Cross-layer top-1 stickiness:

1. Compute the current layer's ordinary POL-001 anchored top-1 baseline.
2. Treat `previous_routes=None` as an explicit sequence boundary and return
   the baseline unchanged.
3. Align previous expert candidates to current score columns by stable expert
   ID; preserve anchor index zero and mark absent experts unavailable.
4. If the margin is zero, return the baseline exactly.
5. Otherwise gather the current score of the aligned incumbent and retain it
   unless the current baseline winner exceeds it by strictly more than the
   configured margin.
6. Return current-score-aligned effective routes plus availability, retention,
   fallback, stabilization, and switch masks without storing the result.

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
- RTR-005 uses a strict rectangular topology: every package appears at every
  covered layer, every local router covers exactly those layers, and local
  expert order is constant across layers.
- RTR-005 requires at least two local experts per package and globally unique
  expert IDs across packages.
- Package and local banks share one feature width, dtype, and device and are
  copied into storage-independent buffer-only RTR-003 deployments.
- Anchor rows never reach a local router; each selected row reaches exactly
  one package-local score bank.
- RTR-005 does not choose packages or experts, compare local scores with the
  anchor, execute experts, or persist a new hierarchy artifact.
- Hard-route distillation never changes the teacher, feature bank, anchor,
  experts, or model.
- The teacher must be frozen, in evaluation mode, on CPU, and produce finite
  scores with the feature bank's dtype and exact row/expert shape.
- The feature bank defines the distilled layers; every declared layer exposes
  at least one uniquely identified teacher expert.
- Student initialization and optimization consume no random state.
- The distilled router has no trainable parameters or teacher-storage aliases
  and preserves teacher expert order within every layer.
- The report binds the result to exact feature-bank metadata and hashed teacher
  scores and assignments, but not to raw prompts.
- Cross-layer stickiness accepts only a previous top-1 decision with the same
  token shape and device as the current scores.
- Previous expert IDs are realigned to current order; an absent expert cannot
  be retained and falls back to ordinary current-layer selection.
- The permanent anchor is always available and retains score zero.
- A zero switch margin is bit-exact ordinary anchored top-1 selection.
- A positive switch margin changes only discrete route choice; selected scores
  remain current-layer values and preserve gradients to selected evidence.
- The stabilizer owns no tensors, parameters, mutable route cache, request
  identity, or generation lifecycle.
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
`hierarchical_package_local` is likewise an in-memory composition of existing
logical banks. Package and local affine heads retain their existing
safetensors formats; callers rebuild the hierarchy explicitly after package
and registry validation.
RTR-007 returns the same `LogicalAppendOnlyRouter` runtime contract, so
`RoutedMLP` needs no distillation-specific dispatch path. This initial slice
does not add a report-bearing persistence format or automatic activation.
RTR-008 operates on `RouteDecision` values outside the Hugging Face model
forward. This keeps batch and generation boundaries explicit and avoids
thread-unsafe hidden state. Model-level propagation of previous routes is not
introduced by this slice.

## Configuration and examples

```python
from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    CrossLayerStickinessConfig,
    CrossLayerTop1Stickiness,
    HardRouteDistillationConfig,
    IdentityRouterFeatureExtractor,
    HierarchicalRouterScoreBank,
    compile_logical_router,
    distill_frozen_router,
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

distilled, distillation_report = distill_frozen_router(
    logical_router.eval(),
    replay_features,
    config=HardRouteDistillationConfig(
        learning_rate=0.05,
        optimization_steps=256,
        l2_weight=1e-4,
    ),
)

stabilizer = CrossLayerTop1Stickiness(
    CrossLayerStickinessConfig(switch_margin=0.1)
)
sticky_result = stabilizer(
    logical_router(next_features, layer_index=20),
    expert_ids=logical_router.expert_ids_for_layer(20),
    previous_routes=previous_layer_routes,
)
previous_layer_routes = sticky_result.routes

hierarchical = HierarchicalRouterScoreBank(
    package_logical_router,
    {
        "code-package": code_local_logical_router,
        "math-package": math_local_logical_router,
    },
)
package_scores = hierarchical.score_packages(features, layer_index=18)
```

See `examples/verify_affine_router.py` for RTR-002 deterministic scoring,
append invariance, and safe serialization. See
`examples/verify_logical_router.py` for RTR-003 compilation, exact incumbent
score preservation, frozen state, and routed execution. Neither example
downloads model weights.
See `examples/verify_registry_router_versions.py` for RTR-004 fitting and
PKG-005 persistence without model downloads.
See `examples/verify_router_distillation.py` for RTR-007 deterministic hard
assignment fitting, exact replay agreement, and frozen deployment state.
See `examples/verify_cross_layer_stickiness.py` for RTR-008 explicit
adjacent-layer sequencing, margin retention, auditable switching, and
buffer-only logical-router immutability.
See `examples/verify_hierarchical_routing.py` for RTR-005 package scoring,
selected-row local scoring, exact anchor omission, and POL-004 reconstruction.

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
- [Dai et al., “StableMoE: Stable Routing Strategy for Mixture of
  Experts”](https://aclanthology.org/2022.acl-long.489/) motivates separating
  route discovery from a lightweight frozen deployment router.

RTR-001, RTR-002, RTR-003, and RTR-004 are framework infrastructure and do not
derive an optimization algorithm from a paper. RTR-004 implements the
blueprint's explicit router-only mode using a documented project-specific
anchored classification objective.
RTR-005 is also framework infrastructure. It composes the blueprint's
two-stage hierarchy from existing affine banks without adopting a separate
paper-derived fitting algorithm.
RTR-007 adopts StableMoE's two-stage distill-then-freeze principle. It differs
by distilling fixed-zero-anchor assignments from an arbitrary immutable
composed teacher over offline CAL-001 features. It does not reproduce the
paper's jointly learned balanced routing objective or its later expert-training
stage.
RTR-008 adopts only StableMoE's goal of reducing route fluctuation. Its
switching-margin algorithm is project-specific: it uses current calibrated
evidence and an explicit previous route, and does not learn, distill, balance,
or persist a router.

## Tests and known limitations

Acceptance tests cover identity and gradient preservation, feature metadata,
the calibrated score equation, tensor shapes, neutral initialization,
trainability, append invariance, ownership uniqueness, deterministic layer
ordering, exact safetensors round trips, overwrite protection, malformed
metadata and tensor rejection, finite-value checks, expert-package manifest
compatibility, logical matrix shapes, source-storage independence, frozen
compiled state, dtype migration, exact RTR-002 score parity, exact incumbent
score preservation, hard-route teacher labeling, anchor-first ties,
deterministic student state, report schema round trips, teacher immutability,
student storage independence, exact replay agreement, invalid teacher
rejection, zero-margin POL-001 equivalence, strict-margin anchor/expert
retention, current-score gradients, cross-layer expert-ID realignment,
absent-expert fallback, explicit sequence boundaries, stateless repeated
calls, audit masks, rectangular hierarchical topology, source-storage
independence, package/local identity validation, selected-row-only local
scoring, anchor-row omission, and `RoutedMLP` plus package-synchronization
integration.

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
RTR-007 currently supports anchored hard top-1 assignments only. Teacher-score
distribution distillation, top-2/adaptive distillation, frozen-route expert
training, production-scale streaming, accelerator fitting, and report-bearing
registry persistence remain separate work. RTR-008 supports adjacent-layer
anchored top-1 decisions only. Top-2/adaptive stickiness, autoregressive
token-to-token hysteresis, automatic model-level route propagation,
capacity-aware incumbent invalidation, and persistent route state are not
implemented.
RTR-005 currently requires rectangular layer coverage and at least two local
experts per package. Ragged package topologies, package-local router
serialization as one artifact, automatic construction from registry
manifests, distributed package routing, and prompt-level shortlisting remain
separate work.
Standalone anchor comparison and top-1 selection are implemented by POL-001;
INF-001 explicitly attaches them for routed-layer dispatch,
PKG-007 validates serialized router ownership against expert payload bindings,
and BAL-001 handles supplied availability masks. Learned projections, capacity
estimation, top-2 routing, and the other limitations remain tracked as later
calibration, runtime-composition, and capacity features.
