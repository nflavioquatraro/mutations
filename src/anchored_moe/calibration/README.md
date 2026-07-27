# Calibration

## Scope

The calibration subpackage owns reusable data contracts for fitting and
calibrating router heads. CAL-001 implements a versioned router feature cache
and reference-bank format. CAL-002 defines categorical domain targets and
individual-expert utility targets aligned to that reference bank. CAL-004
fits independent affine router heads from aligned features and utility targets
using deterministic ridge regression. CAL-005 calibrates those heads with
per-head binary temperature scaling and held-out false-activation-constrained
threshold selection.

This subpackage does not extract hidden states through model hooks, execute
counterfactual model paths, construct pair utilities, refine stability margins,
or install packages. Those responsibilities remain with CAL-003, CAL-006,
CAL-007, model backends, routers, and the registry.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| CAL-001 | `router_feature_bank_format_version: 1` | `features.py` | Validated |
| CAL-002 | `router_label_bank_format_version: 1` | `labels.py` | Validated |
| CAL-004 | `fitting: independent_ridge`, `regularization: <positive float>` | `ridge.py` | Validated |
| CAL-005 | `calibration: temperature_threshold`, `false_activation_rate: <0..1>` | `temperature.py` | Validated |

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
- Reports contain row IDs, calibration bounds, iteration count, losses,
  thresholds, counts, rates, and fallback statuses, but no prompt text.

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

Every feature or label payload is one safetensors file and can be included in
an experiment artifact or covered by an expert-package manifest hash. It
contains no executable Python or pickle content.

## Configuration and examples

```python
from anchored_moe import (
    DomainLabelBank,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    fit_independent_ridge,
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

## References

- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor format and metadata envelope.
- [Hugging Face Datasets loading documentation](https://huggingface.co/docs/datasets/loading)
  motivates explicit dataset identifier, revision, split, and filter
  provenance.
- [FIPS 180-4, Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4)
  defines SHA-256 used for metadata and tensor integrity.
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
  forced-path individual utility, per-head temperatures and thresholds,
  false-activation budgets, reproducibility metadata, and deterministic
  calibration.

CAL-001 and CAL-002 are framework infrastructure and do not derive an
optimization algorithm from a paper.

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

The current formats have no automatic model hooks, counterfactual execution,
pair utility, sharding, streaming writer, memory mapping, compression,
distributed merge, joint multiclass calibration, confidence intervals, or
stability-margin refinement. Domain-label ridge targets, weighted or
incremental fitting, automatic data splitting, and model-integrated collection
are also unsupported. Those remain later calibration features or future
format versions.
