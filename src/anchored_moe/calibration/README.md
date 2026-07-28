# Calibration

## Scope

The calibration subpackage owns reusable data contracts for fitting and
calibrating router heads. CAL-001 implements a versioned router feature cache
and reference-bank format. CAL-002 defines categorical domain targets and
individual-expert utility targets aligned to that reference bank. CAL-004
fits independent affine router heads from aligned features and utility targets
using deterministic ridge regression. CAL-005 calibrates those heads with
per-head binary temperature scaling and held-out false-activation-constrained
threshold selection. CAL-007 defines a sparse counterfactual route-loss table
and evaluates router regret against a deterministic anchor-first oracle.
CAL-003 supplies deterministic bounded pair plans plus pair utility and
improvement-over-best-single targets from aligned precomputed losses. CAL-006
refines newly calibrated package heads against an immutable incumbent router
using explicit stability and useful-win margins. RTR-004 uses the same aligned
feature/utility contracts to recalibrate every installed logical-router column
jointly against a fixed-zero-anchor route objective. COM-003 consumes explicit
oracle pair weights aligned to a CAL-003 plan and deterministically fits a
separate frozen affine mixture head.

NB-004 composes the validated CAL-004/CAL-005 outputs with an immutable
NB-003 expert-only package. Its registry-owned artifact boundary verifies the
exact feature space, data identities, expert/head topology, and report before
publishing a new router-bearing package version.

This subpackage does not extract hidden states through model hooks, execute
counterfactual model paths, derive oracle weights from mixture sweeps,
construct a persistent pair-utility bank, mutate an incumbent router in place, persist
registry-specific routers, or install packages. Those responsibilities remain
with model backends, routers, PKG-005, and the registry.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| CAL-001 | `router_feature_bank_format_version: 1` | `features.py` | Validated |
| CAL-002 | `router_label_bank_format_version: 1` | `labels.py` | Validated |
| CAL-003 | `plan_expert_pairs`, `compute_pair_utilities` | `pairs.py` | Validated |
| CAL-004 | `fitting: independent_ridge`, `regularization: <positive float>` | `ridge.py` | Validated |
| CAL-005 | `calibration: temperature_threshold`, `false_activation_rate: <0..1>` | `temperature.py` | Validated |
| CAL-006 | `refinement: stability_margin`, explicit stability/win partitions | `margins.py` | Validated |
| CAL-007 | `CounterfactualRouteLosses`, `evaluate_counterfactual_oracle` | `oracle.py` | Validated |
| COM-003 | `fit_learned_mixture_head`, `strategy: learned_pair_weight_affine` | `mixture.py` | Validated |
| RTR-004 | `recalibrate_global_router`, `strategy: router_only_global` | `global_recalibration.py` | Validated |
| NB-004 | `CalibratedRouterPackageConfig`, `derive_calibrated_router_package` | `registry/export.py` | Validated |

`RouterFeatureBank` is the immutable in-memory and safetensors-backed format.
`FeatureBankSerializationError` reports unsafe or incompatible payloads.
`DomainLabelBank` and `UtilityLabelBank` are explicit target types because
categorical domains and continuous layer-expert advantages have different
shapes and semantics. `LabelBankSerializationError` reports invalid payloads.
`fit_independent_ridge` consumes a feature bank and aligned utility-label bank,
then returns frozen `AppendOnlyRouterHeads`. `RidgeFittingError` reports
non-finite or numerically unusable fitted parameters.
`calibrate_temperature_and_thresholds` consumes uncalibrated CAL-004 heads and
an aligned held-out bank split into two explicit row partitions. It returns a
new frozen router plus a JSON-compatible
`TemperatureThresholdCalibrationReport`. Per-head metrics are represented by
`HeadCalibrationReport`; `TemperatureThresholdCalibrationError` reports
numerically unusable calibration results.
`refine_stability_margins` consumes frozen CAL-005 candidate heads, an
optional frozen incumbent score bank, and aligned candidate utilities. It
returns independently owned frozen heads plus a JSON-compatible
`StabilityMarginRefinementReport`. `HeadStabilityMarginReport` records
deployed-dtype before/after hinge and route metrics;
`StabilityMarginRefinementError` reports non-finite state or scores.
`CounterfactualRouteLosses` owns detached CPU losses for the anchor, every
single expert, and an explicit sparse set of evaluated pairs.
`evaluate_counterfactual_oracle` returns the minimum-loss candidate,
selected-route loss, and regret without executing model paths.
`PairSamplingPlan` records the exact bounded or exhaustive pair set, seed, and
registry order. `PairUtilityResult` exposes pair advantage over the anchor,
improvement over the best single expert, stable best-pair targets, a strict
mixture-margin mask, and direct construction of `CounterfactualRouteLosses`.
`recalibrate_global_router` consumes one frozen RTR-003 logical router and
aligned anchor-relative utilities, then returns a storage-independent frozen
logical router plus `GlobalRouterRecalibrationReport`. Layer-local metrics use
`LayerGlobalRouterRecalibrationReport`; failures that would publish non-finite
state use `GlobalRouterRecalibrationError`.

`derive_calibrated_router_package` does not refit either stage. It binds the
already fitted frozen heads and complete report to immutable feature/utility
bank identities and a new package version.

## Mathematical formulation

At decoder layer \(\ell\), a frozen router feature extractor produces

\[
\phi_\ell(h)\in\mathbb{R}^{D}.
\]

For a deterministic ordered reference set containing \(N\) token-layer
examples, CAL-001 stores

\[
X_\ell
=
\begin{bmatrix}
\phi_\ell(h_1)^\top\\
\vdots\\
\phi_\ell(h_N)^\top
\end{bmatrix}
\in\mathbb{R}^{N\times D}.
\]

Every cached layer has the same row count \(N\), feature size \(D\), dtype, and
row identifier vector

\[
r\in\mathbb{N}_0^N.
\]

Row identifiers are unique and align a future label vector \(y\) with each
layer's design matrix without retaining prompts or token text.

For domain supervision, CAL-002 stores one categorical target per row:

\[
c_t\in\{0,\ldots,C-1\},
\qquad
c\in\mathbb{N}_0^N.
\]

Class names use an explicitly serialized order. Construction from strings
sorts unique class names lexicographically, making the mapping deterministic.

For individual expert \(k\), the utility target at layer \(\ell\) and row \(t\)
is

\[
a_{\ell,t,k}
=
\mathcal L_{\mathrm{incumbent},\ell,t}
-
\mathcal L_{\mathrm{expert}\ k,\ell,t}.
\]

For \(K_\ell\) experts, the target matrix is

\[
Y_\ell\in\mathbb{R}^{N\times K_\ell}.
\]

Positive utility means the forced expert reduced loss. Expert columns follow
the stored `expert_ids` order and are designed to align with append-only router
head columns.

For CAL-004, center the layer design and utility matrices:

\[
X_{\ell,c}=X_\ell-\mathbf{1}\bar{x}_\ell^\top,
\qquad
Y_{\ell,c}=Y_\ell-\mathbf{1}\bar{y}_\ell^\top.
\]

With a strictly positive ridge penalty \(\lambda\), solve all \(K_\ell\)
independent expert targets together:

\[
W_\ell
=
\left(
X_{\ell,c}^\top X_{\ell,c}+\lambda I_D
\right)^{-1}
X_{\ell,c}^\top Y_{\ell,c}
\in\mathbb{R}^{D\times K_\ell}.
\]

The intercept is fitted without regularization:

\[
b_\ell=\bar{y}_\ell-\bar{x}_\ell^\top W_\ell
\in\mathbb{R}^{K_\ell}.
\]

Router head \(k\) receives column \(W_{\ell,:,k}\) and scalar
\(b_{\ell,k}\). Before CAL-005, CAL-004 heads use \(T_{\ell,k}=1\) and
\(\tau_{\ell,k}=0\), so their scores are the raw fitted utilities.

CAL-005 converts utility sign into a binary activation target:

\[
z_{\ell,t,k}
=
\mathbb{1}\!\left[a_{\ell,t,k}>0\right].
\]

On the temperature partition \(\mathcal I_T\), each head independently fits
one positive scalar:

\[
T_{\ell,k}^{*}
=
\underset{T\in[10^{-2},10^2]}{\arg\min}
\frac{1}{|\mathcal I_T|}
\sum_{t\in\mathcal I_T}
\operatorname{BCEWithLogits}
\left(
\frac{s_{\ell,t,k}}{T},
z_{\ell,t,k}
\right).
\]

The implementation searches inverse temperature \(\beta=1/T\), for which the
binary-log-loss objective is convex. A fixed 96-iteration bounded scalar
search is deterministic. If \(\mathcal I_T\) contains only one target class,
the scale is not identifiable; CAL-005 retains \(T=1\) and reports
`single_class_fallback`.

On the disjoint threshold partition \(\mathcal I_\tau\), a specialist is active
when

\[
\frac{s_{\ell,t,k}}{T_{\ell,k}^{*}}-\tau_{\ell,k}>0.
\]

For requested false-activation budget \(\rho\), eligible thresholds satisfy

\[
\operatorname{FAR}_{\ell,k}(\tau)
=
\frac{
\sum_{t\in\mathcal I_\tau}
\mathbb{1}[z_{\ell,t,k}=0]
\mathbb{1}[s_{\ell,t,k}/T_{\ell,k}^{*}>\tau]
}{
\sum_{t\in\mathcal I_\tau}
\mathbb{1}[z_{\ell,t,k}=0]
}
\le \rho.
\]

Among eligible score boundaries, CAL-005 maximizes true-positive count, then
minimizes false-positive count, then selects the higher threshold. With no
negative rows FAR is defined as zero; with no positive rows the tie rules
select a conservative no-activation threshold.

For CAL-006, let the immutable incumbent maximum include the fixed-zero anchor:

\[
m_{\ell,t}^{\mathrm{old}}
=
\max\left(0,\max_{j\in\mathcal E_{\mathrm{old}}}
g_{\ell,t,j}^{\mathrm{old}}\right).
\]

For candidate \(k\), stability rows \(\mathcal I_S\), useful win rows
\(\mathcal I_{W,k}^{+}=\{t\in\mathcal I_W:a_{\ell,t,k}>0\}\), and non-negative
margins \(\gamma_S,\gamma_W\), the hinge terms are

\[
\mathcal L_{\ell,k}^{S}
=
\frac{1}{|\mathcal I_S|}
\sum_{t\in\mathcal I_S}
\max(0,g_{\ell,t,k}^{\mathrm{new}}
-m_{\ell,t}^{\mathrm{old}}+\gamma_S),
\]

\[
\mathcal L_{\ell,k}^{W}
=
\frac{1}{|\mathcal I_{W,k}^{+}|}
\sum_{t\in\mathcal I_{W,k}^{+}}
\max(0,m_{\ell,t}^{\mathrm{old}}+\gamma_W
-g_{\ell,t,k}^{\mathrm{new}}).
\]

The win term is zero when a head has no positive-utility win rows. CAL-006
optimizes each candidate affine state
\(\theta_{\ell,k}=(w_{\ell,k},b_{\ell,k})\) independently:

\[
\mathcal L_{\ell,k}
=
\lambda_S\mathcal L_{\ell,k}^{S}
+\lambda_W\mathcal L_{\ell,k}^{W}
+\lambda_D\frac{\|\theta_{\ell,k}-\theta_{\ell,k}^{(0)}\|_2^2}
{|\theta_{\ell,k}|}.
\]

Temperature and threshold remain fixed at their CAL-005 values. Because the
affine state changes, the CAL-006 report—not the earlier CAL-005 report—is
authoritative for deployment activation metrics of the refined head.

For RTR-004, prepend zero anchor utility and zero anchor score to every row:

\[
z_{\ell,t}
=
\underset{c\in\{0,\ldots,K_\ell\}}{\arg\max}
\left[0,a_{\ell,t,1},\ldots,a_{\ell,t,K_\ell}\right]_c,
\]

\[
q_{\ell,t}
=
\left[0,g_{\ell,t,1},\ldots,g_{\ell,t,K_\ell}\right].
\]

Class zero is the immutable anchor, so exact ties conservatively choose it.
All affine columns at one layer are optimized jointly:

\[
\mathcal L_\ell
=
\operatorname{CE}(q_\ell,z_\ell)
+
\lambda_D
\frac{\|\theta_\ell-\theta_\ell^{(0)}\|_2^2}{|\theta_\ell|}.
\]

Temperatures and thresholds remain fixed. The complete source initialization
is an eligible optimum, and reduced-precision publication falls back to it if
the deployed objective would increase. Utilities must therefore be measured
relative to `incumbent_route: anchor`; an arbitrary incumbent is not
semantically interchangeable with the fixed-zero class.

For \(K\) experts, CAL-003 considers the registry-ordered pair universe

\[
\mathcal P
=
\{(i,j)\mid 0\le i<j<K\},
\qquad
|\mathcal P|=\frac{K(K-1)}{2}.
\]

Bounded mode maps a deterministic SHA-256 stream derived from the seed to
unique flat pair positions, keeps at most \(B\) pairs, and restores
lexicographic registry order for the aligned loss columns. It does not
materialize \(\mathcal P\) when \(B<|\mathcal P|\). Setting `max_pairs=None`
explicitly selects all of \(\mathcal P\).

For measured anchor loss \(L_{t,0}\), individual-expert losses \(L_{t,k}\),
and sampled pair loss \(L_{t,i,j}\), pair utility is

\[
U^{A}_{t,i,j}=L_{t,0}-L_{t,i,j},
\]

while improvement over the best single expert is

\[
U^{S}_{t,i,j}
=
\min_k L_{t,k}-L_{t,i,j}.
\]

A pair passes the strict mixture acceptance gate with margin
\(\delta_{\mathrm{mix}}\ge0\) when

\[
U^{S}_{t,i,j}>\delta_{\mathrm{mix}}.
\]

The sampled best pair is

\[
p_t^\star
=
\underset{(i,j)\in\mathcal S}{\arg\min}\ L_{t,i,j},
\]

where exact ties use lexicographic registry order. Sampling controls which
pairs are measured; it never changes the stable order used by losses, targets,
or tie-breaking.

COM-003 receives an ordered selected-score matrix
\(S\in\mathbb R^{N\times2}\), oracle target weights
\(A\in[0,1]^{N\times2}\) with \(A_{t,1}+A_{t,2}=1\), and one CAL-003 plan
index per row. It fits

\[
z_t=w^\top S_t+b,
\qquad
\widehat A_{t,1}=\sigma(z_t),
\qquad
\widehat A_{t,2}=1-\widehat A_{t,1},
\]

using soft-target binary cross entropy and normalized coefficient decay:

\[
\mathcal L_{\mathrm{mix}}
=
-\frac1N\sum_t
\left[
A_{t,1}\log\sigma(z_t)
+(1-A_{t,1})\log(1-\sigma(z_t))
\right]
+\lambda\frac{\|w\|_2^2}{2}.
\]

The target weights are explicit measured supervision. A single CAL-003 pair
loss \(L_{t,i,j}\) at one unknown or fixed interpolation coefficient cannot
identify the loss-minimizing coefficient and is never silently treated as a
weight label.

For CAL-007, let \(\mathcal C\) be the explicit candidate sequence containing
the anchor, every single expert, and a sampled subset of expert pairs. For
token or example \(t\), the table stores finite counterfactual losses

\[
L_{t,c},\qquad c\in\mathcal C.
\]

The deterministic oracle is

\[
c_t^\star=\underset{c\in\mathcal C}{\arg\min}\ L_{t,c}.
\]

Candidate order breaks exact loss ties: anchor first, then single experts in
registry order, then sampled pairs in lexicographic registry order. If router
policy \(\pi\) selected a table candidate \(c_t^\pi\), its regret is

\[
R_t^\pi=L_{t,c_t^\pi}-L_{t,c_t^\star}\ge 0.
\]

All single routes are required, while pair routes may remain sparse. This
keeps top-1 regret well-defined without silently performing exhaustive
\(O(K^2)\) pair evaluation.

For expert-only source package \(P_E\), calibrated router \(R\), fitting-bank
identity \(d_F\), calibration-bank identity \(d_C\), and report \(Q\), NB-004
publishes a new package

\[
P_R=\operatorname{Derive}(P_E,R,d_F,d_C,Q).
\]

For every expert payload \(f\), its bytes and digest are invariant:

\[
\operatorname{bytes}_{P_R}(f)=\operatorname{bytes}_{P_E}(f),
\qquad
d_{P_R}(f)=d_{P_E}(f).
\]

The new manifest identity changes because it adds a new immutable version,
router payload hash, calibration identities, thresholds, metrics, and runtime
provenance.

## Algorithms

Feature-batch collection:

1. Accept already extracted floating-point feature batches in caller order.
2. validate their final dimension against the frozen feature specification;
3. detach each batch and move its cached copy to contiguous CPU storage;
4. flatten leading batch and sequence dimensions into rows;
5. concatenate batches independently for each layer;
6. require identical row count, feature size, and dtype across layers;
7. attach unique non-negative row identifiers and reproducibility metadata.

Safetensors export:

1. Reject unsafe paths, symbolic links, and accidental overwrites.
2. write row identifiers and ascending layer tensors under canonical names;
3. describe feature space, dataset revision, collection seed, shapes, dtypes,
   and tensor SHA-256 digests in a canonical JSON metadata envelope;
4. write beside the destination and atomically replace it.

Safetensors loading:

1. Parse the non-executable safetensors header and fixed metadata schema.
2. require the supported format and version;
3. validate exact tensor names and reconstruct the known bank type;
4. recheck row alignment, feature compatibility, dtype, and finite values;
5. compare recomputed tensor digests and canonical metadata;
6. return an immutable bank only after every check passes.

Domain-label construction:

1. Require exactly one non-empty domain string per feature-bank row.
2. sort the unique strings lexicographically;
3. encode each row using that stable class mapping;
4. copy the feature bank's row identifiers and metadata SHA-256 identity;
5. record a non-empty label-source identifier.

Individual-utility construction:

1. Require incumbent losses with shape `[row_count]`.
2. require forced expert losses with shape `[row_count, expert_count]`;
3. require all loss and expert-ID mappings to exactly cover bank layers;
4. compute `incumbent_loss - expert_loss` without mutating inputs;
5. detach continuous targets to CPU and preserve expert column order;
6. record the loss name, incumbent route, row identifiers, and feature-bank
   metadata identity.

Label-bank serialization follows the same atomic, canonical, digest-checked
safetensors protocol as CAL-001, using separate format markers for categorical
and utility payloads.

Independent ridge fitting:

1. Validate the feature and utility bank types, exact metadata identity, row
   identifiers, layer sets, and a finite positive regularization value.
2. Read defensive copies of each layer's `[N, D]` features and `[N, K_l]`
   utility targets.
3. Convert the copies to CPU `torch.float64`.
4. Center features and targets so the intercept remains unregularized.
5. Solve the positive-definite regularized feature system for all expert
   columns at once.
6. Recover the intercept and reject non-finite results.
7. Cast each expert column to the feature-bank dtype.
8. Append a frozen affine head in ascending layer and stored expert order.

Temperature and threshold calibration:

1. Require uncalibrated heads with temperature one and threshold zero.
2. Validate feature size, layer set, expert order, feature/utility identity,
   and finite affine parameters.
3. Require two non-empty, disjoint row-ID sets that cover the held-out bank;
   canonicalize both into feature-bank order.
4. Compute raw affine scores with the source head's deployed dtype.
5. On the temperature rows, convert scores to float64 and optimize bounded
   inverse temperature for the binary target `utility > 0`.
6. Retain temperature one and report a fallback if those targets contain only
   one class.
7. On the disjoint threshold rows, sort deployed-dtype scaled scores and scan
   cumulative class counts in \(O(N\log N)\).
8. Choose the recall-maximizing threshold under the false-activation budget,
   using conservative deterministic ties and explicit single-class statuses.
9. Copy unchanged affine tensors into a new frozen router with calibrated
   temperature and threshold metadata.
10. Return canonical row partitions and per-head losses, counts, rates,
    statuses, temperatures, and thresholds in a JSON-compatible report.

Stability-margin refinement:

1. Require frozen finite candidate heads, a frozen finite incumbent router or
   anchor-only `None`, and exact feature/utility layer and expert alignment.
2. Require non-empty, disjoint stability and win row-ID subsets and
   canonicalize them into feature-bank order.
3. Evaluate the incumbent score bank in its deployed dtype and take the
   maximum with the fixed-zero anchor without retaining gradients.
4. For each candidate head, keep only strictly positive-utility win rows and
   optimize its stability hinge, win hinge, and proximal drift independently.
5. Run seed-free full-batch Adam on CPU in `torch.float64` for the configured
   fixed number of steps and retain the best observed objective.
6. Copy the best affine state to a new frozen head in the source dtype/device,
   while preserving temperature, threshold, ownership, and append order.
7. Re-evaluate deployed-dtype scores; retain the exact CAL-005 source state if
   dtype conversion does not improve the complete objective.
8. Return canonical partitions, hyperparameters, before/after violations,
   incumbent-overwrite activations, useful-row counts, and status in a
   JSON-compatible report.

Global router-only recalibration:

1. Require one non-empty CPU RTR-003 router and exactly aligned CAL-001/CAL-002
   layers, feature dtype, expert identifiers, and column order.
2. Require anchor-relative utilities plus explicit active-registry and
   compatibility SHA-256 identities.
3. Derive anchored best-route targets with stable anchor-first ties.
4. Optimize all affine weights and biases at each layer jointly using
   seed-free full-batch float64 Adam and normalized proximal drift.
5. Preserve every temperature, threshold, ownership key, and global append
   position.
6. Re-evaluate after conversion to the deployed dtype and retain the complete
   source layer if the converted total objective is worse.
7. Return a new buffer-only logical router and JSON-compatible registry-bound
   report without mutating or activating any runtime state.

Pair planning and utility construction:

1. Require at least two unique experts in stable registry order.
2. compute \(K(K-1)/2\) without enumerating candidates;
3. in bounded mode, map SHA-256 `(seed, counter)` values without modulo bias
   to unique flat pair positions and restore lexicographic registry order;
4. in explicit exhaustive mode, enumerate every pair;
5. require finite anchor, single, and pair losses with one token shape, dtype,
   and device and exact registry/plan column counts;
6. detach all losses into owned contiguous CPU storage;
7. compute anchor utility and improvement over the best single expert;
8. select best-pair columns with stable first-minimum ties;
9. optionally apply a strict non-negative mixture-improvement margin;
10. concatenate anchor, single, and sampled-pair losses into CAL-007 order
    without recomputing them.

Learned mixture-weight fitting:

1. Require a CAL-003 plan, finite ordered score pairs, finite normalized target
   weights, and plan-local `torch.long` pair indices with exact row alignment.
2. Detach and copy all fitting inputs to contiguous CPU `torch.float64`.
3. Initialize the two affine coefficients and scalar bias at zero, giving
   neutral one-half deployment weights.
4. Optimize soft-target binary cross entropy plus normalized coefficient
   decay with seed-free full-batch Adam for a fixed step count.
5. Retain the best finite iterate, with the neutral initialization always
   eligible and accepted output forbidden from worsening its objective.
6. Cast the selected coefficients to the caller's score dtype, re-evaluate
   deployed arithmetic, and restore the exactly representable neutral head if
   reduced precision violates the acceptance gate.
7. Publish coefficients as storage-independent buffers with no trainable
   parameters.
8. Report input digests, per-plan-pair row counts, configuration, initial and
   final loss, total objective, and mean absolute weight error.

Counterfactual oracle evaluation:

1. Require canonical anchor and complete single-expert candidate columns.
2. validate sparse pair identities in lexicographic registry order;
3. detach finite losses into owned contiguous CPU storage;
4. map the evaluated router route to one exact candidate column;
5. use first-minimum `argmin` over the canonical columns;
6. gather selected and oracle losses and report their non-negative difference.

NB-004 package derivation:

1. Require a strictly validated router-free NB-003 source package.
2. Revalidate both feature/utility pairs and require globally disjoint fitting
   and calibration row IDs.
3. Require both feature specifications, converted layers, and expert-column
   order to exactly match the source payload bindings.
4. Require frozen finite calibrated heads with exact binding and report
   ownership, temperature, threshold, feature size, and dtype agreement.
5. Copy every expert payload byte-for-byte into private staging and verify its
   original SHA-256 digest.
6. Serialize calibrated heads with the existing RTR-002 safetensors format.
7. Record source identity, bank identities and dataset provenance, ridge
   regularization, the complete CAL-005 report, and calibration runtime.
8. Strictly validate and publish a new immutable package version without
   changing the source package.

## Assumptions and invariants

- Input features have already been produced by a frozen
  `RouterFeatureExtractor`.
- Flat bank tensors have shape `[row_count, feature_size]`.
- Batched inputs may have arbitrary leading dimensions and are flattened in
  explicit caller order.
- Every layer has the same non-zero row count, feature size, and dtype.
- Features are finite floating-point values.
- Cached tensors own detached, contiguous CPU storage.
- Public tensor and metadata accessors return defensive copies.
- Row identifiers are unique, non-negative `torch.long` values.
- Layer order is ascending and deterministic.
- The feature specification contains a positive `output_size`.
- Dataset provenance accepts only identifier, revision, split, and filters, so
  callers cannot add raw-example fields to the v1 metadata envelope.
- No prompt, token text, or label is stored by this format.
- Embedded tensor digests detect accidental payload changes. Expert-package
  distribution additionally relies on PKG-001 whole-file hashes.
- CAL-001 is an unsharded in-memory format; its memory complexity is
  \(O(LND)\) for \(L\) cached layers.
- Every label bank records the exact CAL-001 `metadata_sha256`; alignment
  requires both that identity and row identifiers to match.
- Domain targets use `torch.long` and must reference a declared class.
- Domain class names and expert identifiers are non-empty and unique.
- String domain construction uses lexicographically sorted classes.
- Utility targets are finite floating-point matrices with a common dtype.
- Utility layers exactly match feature-bank layers when constructed from
  losses or checked with `assert_aligned`.
- Utility columns exactly match layer-local expert identifier order.
- Positive utility always means lower expert loss than incumbent loss.
- Loss tensors must share dtype and device within each layer.
- Label payloads contain targets and row IDs, but no prompts or token text.
- CAL-004 accepts only an exactly aligned `UtilityLabelBank`; categorical
  domain labels are not silently converted into regression targets.
- Ridge regularization is finite and strictly positive. This makes the
  centered feature system invertible even when features are rank deficient.
- The intercept is fitted but never ridge-regularized.
- Solves run on CPU in `torch.float64`, independently of cached feature dtype.
- Fitted parameters are finite in both solve precision and the feature-bank
  dtype.
- Fitted heads use temperature one and threshold zero and have
  `requires_grad=False`.
- Head append order is ascending layer order followed by the exact
  layer-local expert order stored in the utility bank.
- CAL-004 fitting is deterministic for identical inputs and does not mutate
  either bank.
- CAL-005 accepts only a non-empty router whose layer-local expert order
  exactly matches the held-out utility bank.
- CAL-005 accepts uncalibrated heads only: input temperature is one and input
  threshold is zero.
- Temperature and threshold row IDs are non-empty, disjoint, unique, known to
  the feature bank, and together cover every held-out row.
- Utility greater than zero is the positive binary activation target; zero
  utility is conservatively negative.
- Temperature is bounded to \([10^{-2},10^2]\) and fitted with 96 deterministic
  inverse-scale search iterations.
- Single-class temperature targets preserve temperature one rather than
  manufacturing an arbitrary confidence scale.
- Threshold selection uses the source head's deployed score dtype so strict
  comparisons and ties match runtime routing.
- Every per-head false-activation rate is at most the requested inclusive
  budget, subject to exact finite-sample counts.
- No-positive threshold partitions activate nothing. No-negative partitions
  define false-activation rate as zero and maximize positive recall.
- CAL-005 never mutates the source router. Output affine weights and biases are
  exact copies, global head order is preserved, and output heads are frozen.
- RTR-004 requires a CPU logical router whose dtype, layers, and expert order
  exactly match the replay banks.
- RTR-004 utilities are anchor-relative; class zero cannot stand for an
  arbitrary incumbent route.
- Recalibration is deterministic for fixed inputs, never mutates the source,
  preserves temperature and threshold metadata, and returns only frozen
  buffer state.
- Every RTR-004 report records the exact registry, compatibility, feature-bank,
  and utility-label identities needed by PKG-005.
- Reports contain row IDs, calibration bounds, iteration count, losses,
  thresholds, counts, rates, and fallback statuses, but no prompt text.
- CAL-006 candidates and incumbents are frozen, finite, feature-compatible,
  and layer-complete; candidate and incumbent expert IDs are disjoint.
- Stability and win row IDs are non-empty, unique, known, and disjoint, but
  may leave unused feature-bank rows for independent evaluation.
- The fixed-zero anchor participates in every incumbent maximum. `None`
  explicitly represents an anchor-only registry.
- Only strictly positive-utility win rows train a candidate; zero and negative
  utility are reported as ignored rather than forced to activate.
- Full-batch float64 optimization is deterministic and does not read or alter
  global pseudorandom state. The initialization remains an eligible best state.
- CAL-006 never mutates the incumbent, candidate, features, or utilities.
  Returned heads own new frozen storage and preserve global append order,
  temperature, threshold, dtype, and device.
- Post-refinement metrics use deployed score arithmetic. They supersede the
  original CAL-005 activation metrics because affine parameters have changed.
- CAL-006 is package-local fitting; it neither mutates a logical router nor
  persists a registry-specific router version.
- CAL-003 requires at least two experts and defaults to at most 64 evaluated
  pairs; exhaustive \(O(K^2)\) enumeration requires `max_pairs=None`.
- Pair sampling depends only on the non-negative seed and exact registry
  indices. It never reads or changes global pseudorandom state.
- Sampled pair loss columns are restored to lexicographic registry order, so
  data alignment and best-pair ties never depend on SHA-256 sampling rank.
- Anchor, single, and pair losses have one finite floating-point dtype, device,
  and token shape before their defensive CPU copies are created.
- Pair utility is positive exactly when a pair reduces anchor loss. Pair
  improvement is positive exactly when a pair beats every individual expert.
- The mixture acceptance gate uses a strict comparison; equality with the
  configured non-negative margin is rejected.
- CAL-003 does not infer unevaluated pair losses or claim exhaustive optimality
  for a bounded plan.
- COM-003 fitting rows reference exact CAL-003 plan-local pair indices, but
  the bounded affine head intentionally uses only the ordered score pair as
  deployment features.
- Oracle target weights are finite, non-negative, share score dtype/device,
  and sum to one per row. Their order matches the selected pair order.
- COM-003 fitting is seed-free full-batch CPU float64 and does not read or
  change global pseudorandom state.
- The zero-coefficient neutral head remains an eligible best state. Published
  deployed-dtype total loss cannot exceed its initial soft-target cross
  entropy.
- The returned head is CPU-resident in the source score dtype, contains
  immutable buffers only, and owns storage independent from inputs and
  optimizer state.
- Input and target digests include dtype, shape, and canonical tensor bytes;
  reports store no scores, target values, prompts, tokens, or hidden states.
- One measured pair loss does not determine an oracle mixture coefficient.
  Model execution and counterfactual weight sweeps are explicit upstream work.
- CAL-007 candidates use one exact expert registry order. The anchor is first,
  every single expert is present, and only canonical sampled pairs follow.
- Every candidate column uses the same finite loss definition and token shape.
- CAL-007 detaches, clones, and moves loss storage to CPU; public accessors are
  defensive and oracle evaluation never mutates caller tensors.
- Oracle ties are deterministic and prefer the anchor, then lower-index
  singles, then lower-index sampled pairs.
- Regret compares only routes explicitly present in the table. An evaluated
  router pair absent from the sparse table is an error, not an implicit loss.
- NB-004 fitting and calibration banks use disjoint global row IDs.
- Both NB-004 banks exactly match the source package's feature specification,
  converted layers, and expert order.
- Every calibrated head is frozen, finite, dtype-compatible, and has one exact
  source payload binding and report entry.
- The source package version cannot be reused for the derived artifact.
- Expert payload bytes and hashes are unchanged in the derived package.
- NB-004 records but does not embed feature or utility tensors in the deployed
  package; notebooks save those reproducibility banks separately.

## Hugging Face integration

The bank accepts ordinary PyTorch tensors returned by router feature modules.
Dataset provenance follows Hugging Face dataset identity conventions:
`dataset_id`, exact `revision`, `split`, and explicit `filters`.

The v1 format deliberately does not require `datasets.Dataset`, model hooks,
or streaming support. A caller can iterate a Dataset in deterministic order,
pass extracted batches to `from_feature_batches`, and later use
`features_for_layer` as the design matrix for CAL-004 ridge fitting.

Domain strings can originate from a deterministic Dataset column. Utility
targets can be built after a separate transform produces incumbent and forced
expert loss tensors. Counterfactual model execution is intentionally not
embedded in the data contract.

CAL-007 likewise accepts loss tensors produced by an external deterministic
Dataset transform or evaluation loop. It stores no logits, prompts, hidden
states, or executable model objects. This keeps model-family execution behind
the backend while the oracle remains a reusable CPU transform.

CAL-003 operates at the same boundary. A caller may use COM-001 or another
declared combination policy inside a deterministic Dataset transform, reduce
each pair output to the configured token loss, and pass only the aligned loss
tensors to `compute_pair_utilities()`. The plan is small JSON-compatible
metadata, but persistent pair-loss serialization is not part of this slice.

COM-003 uses the same offline boundary. A deterministic Dataset transform or
evaluation loop measures oracle pair weights and passes only ordered scores,
normalized targets, and CAL-003 plan indices to
`fit_learned_mixture_head()`. The returned buffer-only head uses standalone
safetensors serialization and can be moved with ordinary PyTorch module
semantics. Package-manifest and configuration-driven model attachment remain
separate work.

CAL-004 is an offline CPU fitting transform over those cached tensors. It
returns the existing RTR-002 router-head type, which can be serialized with
`save_safetensors()` and installed or uploaded later without carrying the
feature or label banks in the deployed router payload.

CAL-005 is also an offline post-hoc transform. It expects a held-out
feature/utility bank distinct from the bank used for CAL-004 ridge fitting;
the API cannot infer dataset independence, so exact dataset revisions, splits,
and filters remain mandatory provenance. Explicit row partitions prevent
temperature fitting from reusing threshold-selection rows within that bank.
The resulting router uses the existing RTR-002 safetensors format, which
already stores each head's temperature and threshold. The separate report can
be written as resolved experiment JSON beside that payload.

CAL-006 is another offline tensor transform. Feature and utility partitions
may come from deterministic Hugging Face Dataset filters for old/general and
new-domain examples, but raw examples remain outside the API. The incumbent
uses the model-neutral `RouterScoreBank` contract, so independently owned
RTR-002 heads and the compiled RTR-003 logical router are both valid score
sources. Refined candidates remain RTR-002 heads and retain their safe
safetensors serialization. Registry-specific association and restoration are
deferred to RTR-004 and PKG-005.

NB-004 makes that deployment boundary explicit:
`derive_calibrated_router_package()` consumes the exact banks, calibrated
router, and report; emits a new local package version; and leaves the original
expert-only package untouched. The derived manifest retains Hugging Face model,
tokenizer, and dataset identities while recording the calibration runtime and
source manifest digest.

Every feature or label payload is one safetensors file and can be included in
an experiment artifact or covered by an expert-package manifest hash. It
contains no executable Python or pickle content.

## Configuration and examples

```python
from anchored_moe import (
    CalibratedRouterPackageConfig,
    DomainLabelBank,
    IdentityRouterFeatureExtractor,
    MixtureWeightFittingConfig,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    compute_pair_utilities,
    derive_calibrated_router_package,
    evaluate_counterfactual_oracle,
    fit_independent_ridge,
    fit_learned_mixture_head,
    plan_expert_pairs,
    recalibrate_global_router,
    refine_stability_margins,
)

extractor = IdentityRouterFeatureExtractor(feature_size=4096)
bank = RouterFeatureBank.from_feature_batches(
    {
        18: [extractor(first_hidden_batch), extractor(second_hidden_batch)],
        19: [extractor(first_next_layer_batch), extractor(second_next_layer_batch)],
    },
    feature_specification=dict(extractor.feature_specification),
    dataset_provenance={
        "dataset_id": "organization/router-reference",
        "revision": "pinned-revision",
        "split": "train",
        "filters": [],
    },
    seed=13,
)
bank.save_safetensors("router-reference.safetensors")

domains = DomainLabelBank.from_domain_names(
    bank,
    domain_names,
    label_source="dataset:domain",
)
utilities = UtilityLabelBank.from_losses(
    bank,
    incumbent_losses_by_layer=incumbent_losses,
    expert_losses_by_layer=expert_losses,
    expert_ids_by_layer=expert_ids,
    loss_name="cross_entropy",
    incumbent_route="anchor",
)
router = fit_independent_ridge(
    bank,
    utilities,
    regularization=1e-3,
)
calibrated_router, calibration_report = (
    calibrate_temperature_and_thresholds(
        router,
        held_out_bank,
        held_out_utilities,
        temperature_row_ids=temperature_row_ids,
        threshold_row_ids=threshold_row_ids,
        false_activation_rate=0.01,
    )
)
calibrated_router.save_safetensors("calibrated-router.safetensors")
resolved_metadata = calibration_report.to_dict()

refined_router, stability_report = refine_stability_margins(
    calibrated_router,
    incumbent_router,
    refinement_bank,
    refinement_utilities,
    stability_row_ids=old_and_general_row_ids,
    win_row_ids=new_domain_row_ids,
    stability_margin=0.1,
    win_margin=0.1,
)
refined_router.save_safetensors("stability-refined-router.safetensors")
resolved_stability_metadata = stability_report.to_dict()

global_router, global_report = recalibrate_global_router(
    logical_registry_router,
    replay_bank,
    replay_utilities,
    registry_sha256=registry.snapshot().registry_sha256,
    compatibility_sha256=registry.snapshot().compatibility_sha256,
    learning_rate=5e-2,
    optimization_steps=256,
    drift_weight=1e-3,
)

routed_package = derive_calibrated_router_package(
    expert_only_package,
    "./code-experts-1.0.0-router-1",
    calibrated_router=calibrated_router,
    calibration_report=calibration_report,
    fitting_feature_bank=bank,
    fitting_utility_labels=utilities,
    calibration_feature_bank=held_out_bank,
    calibration_utility_labels=held_out_utilities,
    config=CalibratedRouterPackageConfig(
        specialization_version="1.0.0+router.1",
        calibration_dataset_version="router-data-v1",
        ridge_regularization=1e-3,
        runtime_versions=runtime_versions,
    ),
)

pair_plan = plan_expert_pairs(
    ("code", "math", "bio"),
    max_pairs=2,
    seed=13,
)
pair_utility = compute_pair_utilities(
    pair_plan,
    anchor_losses=anchor_losses,
    single_losses=single_losses,
    pair_losses=measured_pair_losses,
    loss_name="cross_entropy",
)
useful_pairs = pair_utility.pair_beats_best_single(
    minimum_margin=0.01,
)

mixture_head, mixture_report = fit_learned_mixture_head(
    pair_plan,
    selected_scores=ordered_pair_scores,
    target_weights=measured_oracle_weights,
    pair_indices=plan_local_pair_indices,
    config=MixtureWeightFittingConfig(
        learning_rate=5e-2,
        optimization_steps=256,
        l2_weight=1e-4,
    ),
)
mixture_head.save_safetensors("learned-mixture-head.safetensors")

route_losses = pair_utility.to_counterfactual_route_losses()
oracle = evaluate_counterfactual_oracle(
    route_losses,
    selected_candidate_indices,
)
print(oracle.mean_regret)
```

See `examples/verify_router_feature_bank.py` for deterministic collection,
serialization, compatibility validation, and exact reload without model
downloads. See `examples/verify_router_label_banks.py` for categorical and
individual-utility construction, alignment, serialization, and exact reload.
See `examples/verify_independent_ridge.py` for deterministic utility fitting
and anchored top-1 selection without model downloads.
See `examples/verify_temperature_thresholds.py` for separate fitting and
held-out calibration banks, explicit row partitions, immutable calibration,
budgeted threshold selection, and anchored top-1 routes.
See `examples/verify_stability_margins.py` for deterministic candidate-only
refinement, exact incumbent/source preservation, post-refinement violation
metrics, and sequential anchored top-1 routes.
See `examples/verify_registry_router_versions.py` for deterministic RTR-004
global recalibration followed by PKG-005 persistence and exact explicit
registry activation.
See `examples/verify_pair_utilities.py` for bounded SHA-256 pair planning,
pair utility, strict mixture margins, best-pair targets, and CAL-007 handoff.
See `examples/verify_learned_mixture_weights.py` for COM-003 oracle-target
fitting, immutable reports, learned activation combination, and safe reload.
See `examples/verify_top1_top2_oracle.py` for sparse candidate construction,
deterministic oracle selection, router regret, and top-1/top-2 comparison.
See `examples/fit_router.py` and
`notebooks/03_fit_and_calibrate_router.ipynb` for immutable NB-004 package
derivation and PKG-007 materialization.

## References

- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor format and metadata envelope.
- [Hugging Face Datasets loading documentation](https://huggingface.co/docs/datasets/loading)
  motivates explicit dataset identifier, revision, split, and filter
  provenance.
- [FIPS 180-4, Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4)
  defines SHA-256 used for metadata and tensor integrity.
- [PyTorch `binary_cross_entropy_with_logits` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)
  defines the stable soft-target objective used for COM-003 oracle weights.
- [PyTorch `Adam` documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
  defines the optimizer used by COM-003's deterministic full-batch CPU fit.
- Hoerl and Kennard,
  [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634),
  motivate the positive \(L_2\) penalty used by CAL-004. This implementation
  differs by leaving the intercept unregularized, solving layer-local
  multi-output utility targets together, and emitting frozen affine router
  heads rather than a general regression estimator.
- Guo et al.,
  [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599),
  motivate post-hoc temperature scaling. CAL-005 differs by independently
  calibrating binary positive-utility logits for each expert instead of one
  temperature over multiclass logits. Its held-out false-activation threshold
  policy is a framework-specific addition.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines frozen router features, shared reference banks, domain labels,
  forced-path individual and pair utility, mixture-improvement margins,
  bounded pair sampling, per-head temperatures and thresholds,
  false-activation budgets, separate learned mixture scores, counterfactual
  route oracles, router regret, and deterministic calibration.

CAL-001 and CAL-002 are framework infrastructure and do not derive an
optimization algorithm from a paper.
CAL-003's SHA-256-ranked bounded sampling and loss arithmetic are also
framework infrastructure rather than a paper-derived algorithm.
COM-003's affine soft-target fitting objective, deterministic optimizer,
neutral acceptance gate, and explicit oracle-target boundary are
project-specific framework choices rather than a paper-derived algorithm.
CAL-006 implements the blueprint's project-specific hinge equations. Its
full-batch optimizer, proximal drift term, candidate-only ownership boundary,
and post-refinement report are documented implementation choices rather than
paper-derived behavior.
CAL-007 is also framework evaluation infrastructure rather than a
paper-derived algorithm.
RTR-004 implements the architecture blueprint's router-only recalibration mode.
Its anchored best-route classification target, proximal drift term, full-batch
optimizer, and deployed-dtype acceptance gate are documented project-specific
choices rather than a paper-derived algorithm.
NB-004 only validates and packages CAL-004/CAL-005 outputs; it introduces no
additional paper-derived fitting algorithm.

## Tests and known limitations

Acceptance tests cover deterministic flattening, layer and row ordering,
gradient detachment, source immutability, defensive accessors, exact
safetensors round trips, byte-deterministic output, compatibility checks,
overwrite protection, malformed metadata and tensor rejection, embedded
digest checks, invalid provenance, shape and dtype mismatch, non-finite
features, duplicate row identifiers, router integration, source
documentation, and executable example behavior.

CAL-002 tests additionally cover lexicographic class encoding, exact
feature-bank binding, row alignment, the signed utility equation, expert
column order, loss shape and finite-value validation, gradient detachment,
defensive accessors, exact domain and utility round trips, format separation,
metadata and tensor tampering, router-order integration, and executable
example behavior.

CAL-003 tests cover deterministic SHA-256 bounded subsets, seed changes,
explicit exhaustive enumeration, registry-order restoration, pair counts,
invalid budgets and identities, anchor utility, improvement over best single,
strict mixture margins, stable best-pair ties, source immutability, defensive
CPU storage, CAL-007 conversion, COM-001 and EVAL-004 integration, public
documentation, and executable example behavior.

COM-003 tests cover CAL-003 plan-index alignment, ordered score/target
validation, normalized oracle weights, deterministic full-batch fitting,
neutral-initialization acceptance, final loss and weight error, input
immutability, tensor digests, per-pair counts, strict JSON report round trips,
frozen storage-independent deployment, score and activation gradients,
safetensors round trips and malformed payload rejection, INF-002 route
invariance, public documentation, and executable example behavior.

CAL-004 tests cover the centered closed-form solution, unregularized
intercepts, positive regularization, rank-deficient features, multiple layers
and experts, stable ordering, deterministic repeated fits, source
immutability, frozen parameters, dtype overflow rejection, exact feature-bank
alignment, fitted-head serialization, anchored top-1 integration, public
documentation, and executable example behavior.

CAL-005 tests cover binary-log-loss improvement, bounded temperature search,
explicit and canonical disjoint partitions, complete row coverage, held-out
false-activation budgets, deterministic score ties, deployed-dtype threshold
semantics, zero-utility negativity, no-positive and no-negative threshold
fallbacks, one-class temperature fallback, immutable source parameters and
metadata, exact global head order, incompatible router rejection,
JSON-compatible reports, calibrated-head serialization, CAL-004 integration
with a separate held-out bank, anchored top-1 routing, public documentation,
and executable example behavior.

CAL-006 tests cover exact feature/utility/head alignment, anchor-only and
specialist incumbents, fixed-zero incumbent maxima, explicit canonical
partitions, positive-utility win masking, both hinge equations, proximal
drift, deterministic repeated optimization, best-initialization fallback,
deployed-dtype metrics, source immutability, storage independence, frozen
outputs, append order, safetensors reload, CAL-005 integration, anchored
top-1 sequential routes, public documentation, malformed input rejection, and
executable example behavior.

CAL-007 tests cover canonical anchor/single/sparse-pair ordering, deterministic
ties, selected-route lookup, non-negative regret, route agreement, defensive
CPU storage, gradient detachment, malformed candidates, non-finite losses,
shape and index validation, public documentation, policy integration, and the
executable oracle example.

RTR-004 tests cover anchored target construction, joint loss improvement,
stable ties, source immutability, storage independence, frozen output,
deterministic repetition, strict layer/expert/dtype alignment, report
round-trip validation, registry binding, PKG-005 handoff, and the executable
offline example.

NB-004 tests additionally cover immutable source bytes, deterministic derived
identity, unchanged expert hashes, exact head/binding/report agreement,
disjoint fit/calibration identities, router payload hashing, anchored
anchor/expert decisions, strict reload, and PKG-007 materialization.

The current formats have no automatic model hooks, counterfactual model
execution, learned pair weights, serialized pair-utility bank, sharding,
streaming writer, memory mapping, compression,
distributed merge, top-2 pair-aware global objectives, accelerator fitting,
joint confidence calibration, or confidence intervals. CAL-006 does not enforce CAL-005's original
false-activation budget after changing affine parameters; its own
post-refinement report is authoritative. Domain-label ridge targets, weighted
or incremental fitting, automatic data splitting, and model-integrated
collection are also unsupported. Those remain later calibration features or
future format versions.
