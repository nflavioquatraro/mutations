# Calibration

## Scope

The calibration subpackage owns reusable data contracts for fitting and
calibrating router heads. CAL-001 implements a versioned router feature cache
and reference-bank format. CAL-002 defines categorical domain targets and
individual-expert utility targets aligned to that reference bank.

This subpackage does not extract hidden states through model hooks, execute
counterfactual model paths, construct pair utilities, fit affine heads, choose
temperatures or thresholds, or install packages. Those responsibilities remain
with later CAL-003 through CAL-007 features, model backends, routers, and the
registry.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| CAL-001 | `router_feature_bank_format_version: 1` | `features.py` | Validated |
| CAL-002 | `router_label_bank_format_version: 1` | `labels.py` | Validated |

`RouterFeatureBank` is the immutable in-memory and safetensors-backed format.
`FeatureBankSerializationError` reports unsafe or incompatible payloads.
`DomainLabelBank` and `UtilityLabelBank` are explicit target types because
categorical domains and continuous layer-expert advantages have different
shapes and semantics. `LabelBankSerializationError` reports invalid payloads.

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
```

See `examples/verify_router_feature_bank.py` for deterministic collection,
serialization, compatibility validation, and exact reload without model
downloads. See `examples/verify_router_label_banks.py` for categorical and
individual-utility construction, alignment, serialization, and exact reload.

## References

- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor format and metadata envelope.
- [Hugging Face Datasets loading documentation](https://huggingface.co/docs/datasets/loading)
  motivates explicit dataset identifier, revision, split, and filter
  provenance.
- [FIPS 180-4, Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4)
  defines SHA-256 used for metadata and tensor integrity.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines frozen router features, shared reference banks, domain labels,
  forced-path individual utility, reproducibility metadata, and deterministic
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

The current formats have no automatic model hooks, counterfactual execution,
pair utility, sharding, streaming writer, memory mapping, compression,
distributed merge, fitting algorithm, or threshold selection. Those remain
later calibration features or future format versions.
