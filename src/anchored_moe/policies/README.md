# Policies

## Scope

The policies subpackage owns decisions made from already calibrated router
scores, already selected specialist outputs, and capacity metadata. POL-001
implements fixed-zero anchored hard top-1 selection. POL-002 ranks an
anchor-veto top-2 pair, POL-003 chooses adaptive anchor/top-1/top-2 route
arity, and POL-005 chooses a globally ranked top-2 pair restricted by explicit
PKG-003 compatibility groups. COM-001 combines an approved specialist pair
with score-softmax weights, COM-002 supplies equal-weight and
normalized-positive-evidence baselines, and BAL-001 rewrites unavailable
top-1 experts to the anchor.

This subpackage does not extract router features, produce or fit expert scores,
attach experts, execute MLPs, dispatch tokens, estimate serving capacity,
install packages, or record telemetry. Those responsibilities remain with
routers, calibration, modeling and inference, later capacity, registry, and
evaluation features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| POL-001 | `selection.policy: anchored_top1` | `selection.py` | Validated |
| POL-002 | `selection.policy: anchor_veto_top2` | `selection.py` | Validated |
| POL-003 | `selection.policy: adaptive_top1_top2` | `selection.py` | Validated |
| POL-005 | `selection.policy: global_compatibility_top2` | `selection.py` | Validated |
| COM-001 | `combination.policy: score_softmax` | `combination.py` | Validated |
| COM-002 | `combination.policy: equal_weight` | `combination.py` | Validated |
| COM-002 | `combination.policy: normalized_positive_evidence` | `combination.py` | Validated |
| BAL-001 | `capacity.overflow: anchor` | `capacity.py` | Validated |

`SelectionPolicy` is the selection strategy interface.
`AnchoredTop1Selection` returns `RouteDecision`,
`AnchorVetoTop2Selection` returns pair-shaped `Top2RouteDecision`, and
`AdaptiveTop1Top2Selection` returns pair-ranked `AdaptiveRouteDecision`
metadata with explicit route arity.
`CompatibilityConstrainedTop2Selection` binds one immutable
`MixtureCompatibilityIndex` and also returns `Top2RouteDecision`. INF-001
consumes the top-1 result, INF-002 consumes anchor-veto top-2, and
compatibility-aware and adaptive sparse execution remain separate.

`OverflowToAnchorCapacity` consumes a `RouteDecision` and optional per-token
expert availability. `CapacityDecision` exposes the effective routes and the
tokens that overflowed.

`CombinationPolicy` is the activation-combination interface.
`ScoreSoftmaxCombination` consumes only already approved, already ordered
specialist pairs and returns `ActivationCombinationResult`, containing the
weighted activation and exact deployed weights. `EqualWeightCombination` and
`NormalizedPositiveEvidenceCombination` preserve the same result contract so
INF-002 can compare weighting strategies without changing selection, sparse
execution, or output reconstruction.

## Mathematical formulation

At decoder layer \(\ell\), let calibrated expert scores for each token be

\[
G_\ell(h)
=
\left[g_{\ell,1}(h),\ldots,g_{\ell,K}(h)\right].
\]

The anchor is always candidate zero with fixed score

\[
g_{\ell,0}(h)=0.
\]

Anchored hard top-1 chooses

\[
r_\ell(h)
=
\arg\max\left\{
0,g_{\ell,1}(h),\ldots,g_{\ell,K}(h)
\right\}.
\]

Candidate indices have a public, stable meaning:

\[
r_\ell(h)=
\begin{cases}
0, & \text{anchor},\\
k, & \text{expert at position }k-1\text{ in `expert_ids`}.
\end{cases}
\]

For expert score tensor
\(G\in\mathbb{R}^{B\times S\times K}\), selected indices and scores both
have shape \(B\times S\). The implementation accepts arbitrary leading
dimensions and treats only the final dimension as candidates.

The policy deliberately defines stable ties:

1. the anchor appears before every expert and therefore wins any tie at zero;
2. experts appear in router append order, so the earliest appended expert wins
   an equal positive score.

For anchor-veto top-2, let
\((c_{t,1},c_{t,2})\) be the first two candidates after a stable descending
rank of

\[
\left[0,g_{t,1},\ldots,g_{t,K}\right].
\]

The specialist-pair approval indicator is

\[
m_t =
\mathbb{1}[c_{t,1}\ne 0 \land c_{t,2}\ne 0].
\]

Because the anchor is first among scores tied at zero, \(m_t=1\) exactly when
at least two specialists have scores strictly greater than zero. Selected
indices and scores have shape
\(D_1\times\cdots\times D_n\times 2\). An empty registry duplicates the anchor
only to preserve this pair-shaped metadata contract; it never represents two
executions.

For POL-005, let \(c(k)\) be expert \(k\)'s explicit PKG-003 group or
\(\bot\) when no group is declared. The eligible pair set for token \(t\) is

\[
\mathcal P_t
=
\left\{
(i,j):
i<j,\;
c(i)=c(j)\ne\bot,\;
g_{t,i}>0,\;
g_{t,j}>0
\right\}.
\]

For each pair, scores are placed in descending order with append order
breaking an exact tie. POL-005 chooses the lexicographic maximum

\[
(i^*,j^*)
=
\arg\max_{(i,j)\in\mathcal P_t}
\left(g_{t,(1)},g_{t,(2)}\right).
\]

Lexicographic ranking preserves top-2 meaning: the strongest compatible first
member is maximized before the supporting second member. Exact pair-score ties
retain pair enumeration order. When \(\mathcal P_t=\varnothing\), the best
specialist is retained beside the fixed-zero anchor in score order. The
decision is therefore anchor-vetoed without hiding which individual expert
had the strongest evidence.

For POL-003, let \(g_{t,(1)}\) and \(g_{t,(2)}\) be the first two specialist
scores after stable descending ranking, and let
\(\tau_{\mathrm{mix}}\ge 0\). The active specialist count is

\[
a_t =
\begin{cases}
0, & K=0\ \text{or}\ g_{t,(1)}\le 0,\\
1, & g_{t,(1)}>0\ \land\
     (K=1\ \text{or}\ g_{t,(2)}\le\tau_{\mathrm{mix}}),\\
2, & K\ge2\ \land\ g_{t,(2)}>\tau_{\mathrm{mix}}.
\end{cases}
\]

Thus \(a_t=0\) means anchor, \(a_t=1\) means top-1, and \(a_t=2\) means top-2.
The strict final inequality makes an exact mixture-threshold tie remain
top-1. Non-negative \(\tau_{\mathrm{mix}}\) also guarantees that every top-2
route contains two strictly positive specialist scores.

`AdaptiveRouteDecision` retains the two highest specialist indices and scores
with shape \(D_1\times\cdots\times D_n\times2\), while route arity has token
shape \(D_1\times\cdots\times D_n\). If fewer than two specialists exist,
candidate index zero and score zero pad the missing ranking slots. Padding is
never active.

For an approved pair \(i,j\), COM-001 computes

\[
\alpha_i =
\frac{\exp(g_i/T_{\mathrm{mix}})}
{\exp(g_i/T_{\mathrm{mix}})+\exp(g_j/T_{\mathrm{mix}})},
\qquad
\alpha_j = 1-\alpha_i,
\]

\[
y_t = \alpha_i E_i(h_t)+\alpha_j E_j(h_t),
\qquad T_{\mathrm{mix}}>0.
\]

For scores with shape
\(D_1\times\cdots\times D_n\times2\) and selected outputs with shape
\(D_1\times\cdots\times D_n\times2\times O_1\cdots\times O_m\), weights have
the score shape and the combined output has shape
\(D_1\times\cdots\times D_n\times O_1\cdots\times O_m\).

COM-002's fixed baseline ignores score magnitude after selection:

\[
\alpha_i=\alpha_j=\frac{1}{2},
\qquad
y_t=\frac{1}{2}E_i(h_t)+\frac{1}{2}E_j(h_t).
\]

Its evidence baseline first clips calibrated scores at zero:

\[
e_k=\max(g_k,0),
\qquad
\alpha_k=\frac{e_k}{e_i+e_j},
\qquad
e_i+e_j>0.
\]

The implementation rejects a pair when \(e_i+e_j=0\), because normalized
evidence is undefined and silently choosing equal or anchor weights would
introduce a second policy. POL-002-approved pairs have \(g_i>0\) and \(g_j>0\),
so the rejection protects standalone callers without affecting valid INF-002
routes.

For selected route \(r_t\) and availability
\(C_{t,k}\in\{\mathrm{false},\mathrm{true}\}\), BAL-001 computes

\[
r'_t =
\begin{cases}
r_t, & r_t=0,\\
r_t, & C_{t,r_t-1}=\mathrm{true},\\
0, & C_{t,r_t-1}=\mathrm{false}.
\end{cases}
\]

Availability has shape \(D_1\times\cdots\times D_n\times K\), while the
effective route and overflow mask have token shape
\(D_1\times\cdots\times D_n\).

## Algorithms

For one tensor of calibrated expert scores:

1. Require a floating-point tensor with a final candidate dimension.
2. validate that all scores are finite;
3. validate one unique expert identifier for every score column;
4. create an all-zero anchor column with matching dtype and device;
5. prepend the anchor column to expert scores;
6. apply `torch.argmax` along the candidate dimension;
7. gather each winning score;
8. return a `RouteDecision` containing indices, scores, expert order, and
   derived anchor/expert masks.

An empty registry has score shape `[..., 0]`. Prepending the anchor still
produces one valid candidate, so every token deterministically selects index
zero.

Anchor-veto top-2:

1. Apply the same score, identifier, shape, and finite-value validation as
   anchored top-1.
2. Prepend the fixed-zero anchor column.
3. Stable-sort candidates in descending score order.
4. Retain the first two indices and scores.
5. Mark a token anchor-vetoed if either retained index is zero.
6. For an empty registry, return a duplicated anchor pair and a false
   specialist-pair mask.

Compatibility-constrained top-2:

1. Apply shared score, identifier, shape, and finite-value validation.
2. Require the candidate expert set to exactly match the immutable PKG-003
   index and realign groups by stable expert ID.
3. Construct an anchor-veto fallback from the best individual specialist.
4. Enumerate same-group expert pairs in candidate append order.
5. Require both pair scores to be strictly positive.
6. Rank each pair internally by score with append-order ties.
7. Retain the lexicographically strongest score pair; retain the first pair
   on an exact pair-score tie.
8. Return the fallback containing the anchor when no pair is eligible.

Adaptive anchor/top-1/top-2:

1. Apply the shared score, identifier, shape, and finite-value validation.
2. Require a finite non-negative mixture threshold.
3. Stable-sort specialists only in descending score order.
4. Retain the first two specialists and pad missing slots with candidate zero.
5. Choose anchor unless the first specialist is strictly positive.
6. Choose top-2 only when a second specialist exists and its score is strictly
   greater than the mixture threshold; otherwise choose top-1.
7. Return ranked metadata, route arity, mode masks, and active per-expert masks.

Score-softmax combination:

1. Require finite pair scores and matching pair-ordered activation prefixes.
2. Require scores and activations to share dtype and device so combination
   cannot silently promote the model output.
3. Divide selected scores by the finite positive mixture temperature.
4. Apply numerically stable `torch.softmax` over the pair dimension.
5. Broadcast weights across all trailing activation dimensions.
6. Sum the two weighted activations and return both output and weights.

Equal-weight combination:

1. Apply the same pair shape, finite-score, dtype, and device validation.
2. Create pair weights equal to one half in the shared input dtype and device.
3. Broadcast, combine, and return output and exact deployed weights.

Normalized-positive-evidence combination:

1. Apply the shared combination validation.
2. Clamp each selected score below at zero.
3. Sum the two evidence values for each token and reject any zero total.
4. Divide each evidence value by its pair total.
5. Broadcast, combine, and return output and exact deployed weights.

Capacity fallback:

1. Treat omitted availability as all experts available.
2. Gather availability for each selected expert in stable expert order.
3. Leave existing anchor routes unchanged.
4. Mark unavailable specialist selections in the overflow mask.
5. Replace their candidate index and selected score with anchor values of zero.
6. Return every token in a `CapacityDecision`; none are dropped.

## Assumptions and invariants

- Expert scores are already calibrated against the anchor's fixed-zero
  evidence.
- The final score dimension exactly follows the supplied `expert_ids` order.
- Expert identifiers are non-empty, unique, and contain no surrounding
  whitespace.
- Candidate zero always means the anchor.
- Candidate `i + 1` always means `expert_ids[i]`.
- Anchor-first and append-order tie-breaking are deterministic.
- POL-002 approves a pair only when the anchor is absent from both ranked
  positions.
- POL-002 does not assert that a specialist pair is compatible or available.
- POL-003 route arity is always zero, one, or two.
- POL-003 retains ranked specialists even for anchor and top-1 decisions so
  threshold behavior remains auditable.
- POL-003 padding uses candidate zero but can never be active.
- POL-003 exact anchor-score and mixture-threshold ties choose the more
  conservative anchor and top-1 modes respectively.
- POL-003 does not execute, combine, capacity-check, or compatibility-check
  selected specialists.
- POL-005 treats missing group metadata as non-mixable.
- POL-005 requires exact candidate-set equality with its immutable
  compatibility index; candidate order may be realigned by stable ID.
- POL-005 selects only distinct, same-group specialists with two strictly
  positive scores.
- POL-005 returns an anchor-containing fallback when no compatible pair is
  eligible.
- POL-005 pair and score ties preserve candidate append order.
- COM-001 receives specialists in POL-002 ranked order and never mixes the
  anchor.
- COM-001 requires a finite, strictly positive `mix_temperature`.
- COM-001 preserves the shared activation/score dtype and device.
- COM-001 weights are non-negative, finite, and sum to one.
- COM-002 equal weights are exactly one half and intentionally have no score
  gradient.
- COM-002 normalized-positive-evidence weights are non-negative, sum to one,
  and preserve gradients through strictly positive scores.
- A normalized-positive-evidence pair must contain at least one strictly
  positive selected score.
- All combination strategies preserve the shared activation/score dtype and
  device and return exact deployed weights.
- Selection and combination do not execute expert modules.
- Scores containing NaN or infinity are rejected rather than routed.
- Selection does not mutate the input scores.
- Selected indices use `torch.long`; selected scores preserve score dtype and
  device.
- Route decisions expose masks but contain no prompts, tokens, hidden states,
  or other sensitive content.
- Selection does not claim that the selected expert has capacity.
- BAL-001 consumes explicit availability but does not estimate capacity.
- Overflow never selects another expert and never drops a token.

## Hugging Face integration

The policies operate on ordinary PyTorch tensors and do not depend on a
specific Transformers model family. INF-001 integrates them with `RoutedMLP`
only after an explicit `configure_anchored_top1()` call.

`AnchoredMoEConfig` continues to reject
`selection.policy: anchored_top1`, `selection.policy: anchor_veto_top2`,
`selection.policy: adaptive_top1_top2`, and
`selection.policy: global_compatibility_top2` for model-level configuration
because runtime strategy construction is not yet configuration-driven. Expert
package manifests may declare `anchored_top1`; top-2 and adaptive package
attachment remain unsupported. Standalone experiments can use all selection
strategies explicitly. INF-002 composes POL-002 with any `CombinationPolicy`
in a standalone sparse dispatcher; it does not yet consume POL-005. This
avoids silently accepting an incomplete model configuration.

## Configuration and examples

```python
from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CompatibilityConstrainedTop2Selection,
    EqualWeightCombination,
    MixtureCompatibilityIndex,
    NormalizedPositiveEvidenceCombination,
    OverflowToAnchorCapacity,
    ScoreSoftmaxCombination,
)

expert_ids = router.expert_ids_for_layer(layer_index)
expert_scores = router(features, layer_index=layer_index)

decision = AnchoredTop1Selection()(
    expert_scores,
    expert_ids=expert_ids,
)

anchor_tokens = decision.anchor_mask
code_tokens = decision.mask_for_expert("code:0")

effective = OverflowToAnchorCapacity()(
    decision,
    expert_availability=availability,
)

top2 = AnchorVetoTop2Selection()(
    expert_scores,
    expert_ids=expert_ids,
)
compatibility = MixtureCompatibilityIndex(
    expert_ids=("code:0", "code:1", "math:0"),
    group_ids=("code-v1", "code-v1", None),
)
compatible_top2 = CompatibilityConstrainedTop2Selection(compatibility)(
    expert_scores,
    expert_ids=expert_ids,
)
adaptive = AdaptiveTop1Top2Selection(mix_threshold=0.25)(
    expert_scores,
    expert_ids=expert_ids,
)
anchor_tokens = adaptive.anchor_mask
single_expert_tokens = adaptive.top1_mask
mixed_expert_tokens = adaptive.top2_mask
approved_scores = top2.selected_scores[top2.expert_pair_mask]
mixed = ScoreSoftmaxCombination(mix_temperature=1.0)(
    pair_ordered_expert_outputs,
    selected_scores=approved_scores,
)
equal_baseline = EqualWeightCombination()(
    pair_ordered_expert_outputs,
    selected_scores=approved_scores,
)
evidence_baseline = NormalizedPositiveEvidenceCombination()(
    pair_ordered_expert_outputs,
    selected_scores=approved_scores,
)
```

The public strategy names are `anchored_top1`, `anchor_veto_top2`,
`adaptive_top1_top2`, `global_compatibility_top2`, `score_softmax`,
`equal_weight`, and `normalized_positive_evidence`. See
`examples/verify_anchored_top1.py`, `examples/verify_anchor_veto_top2.py`,
`examples/verify_compatibility_top2.py`,
`examples/verify_adaptive_selection.py`, and
`examples/verify_combination_baselines.py` for deterministic integration.

## References

- [PyTorch `argmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html)
  defines first-maximum tie behavior used for anchor-first and append-order
  decisions.
- [PyTorch `gather` documentation](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
  defines collection of selected scores and expert availability.
- [PyTorch `argsort` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argsort.html)
  defines the stable descending ordering used for deterministic top-2 ties.
- [PyTorch `where` documentation](https://docs.pytorch.org/docs/stable/generated/torch.where.html)
  defines tokenwise replacement while comparing compatible pair candidates.
- [PyTorch `softmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  defines normalized exponentiation used for COM-001 mixture weights.
- [PyTorch `clamp` documentation](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html)
  defines the positive-evidence clipping operation used by COM-002.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines fixed-zero anchor evidence, anchored hard top-1, anchor-veto top-2,
  adaptive anchor/top-1/top-2, score-softmax activation combination, and both
  COM-002 baselines.

POL-001, POL-002, POL-003, POL-005, COM-001, COM-002, and BAL-001 are
framework infrastructure and do not derive a training or optimization
algorithm from a paper.

## Tests and known limitations

Acceptance tests cover mixed anchor/expert decisions, exact selected scores,
anchor masks, per-expert masks, anchor ties, expert ties, empty registries,
arbitrary leading dimensions, dtype and device preservation, repeatability,
input immutability, malformed identifiers, shape mismatches, non-floating and
non-finite scores, public result validation, affine-router integration,
expert-package policy metadata, available routes, unavailable specialist
fallback, empty registries, availability shape/dtype validation, and no-token
dropping. POL-002 tests additionally cover the strict two-positive-specialist
gate, ranked scores and indices, anchor and append-order ties, empty and
single-expert registries, pair masks, and affine-router integration. COM-001
tests cover exact and extreme-score weights, temperature behavior, arbitrary
leading and trailing shapes, dtype/device preservation, gradient flow, input
immutability, and malformed public construction. COM-002 tests cover exact
equal and positive-evidence weights and outputs, negative-score clipping,
zero-evidence rejection, arbitrary shapes, dtype/device preservation, input
immutability, expected score-gradient behavior, public docstrings, and
composition through INF-002.

POL-003 tests cover all three route arities, exact anchor and mixture-threshold
ties, append-order ranking ties, active per-expert masks, empty and
single-expert padding, arbitrary leading dimensions, dtype/device preservation,
input immutability, invalid thresholds, malformed public decisions, and
affine-router integration.

POL-005 tests cover incompatible high-score rejection, selection of a
lower-ranked compatible pair, lexicographic pair ranking, append-order ties,
missing-group anchor fallback, empty and single-expert registries, candidate
realignment, stale compatibility rejection, dtype/device preservation, input
immutability, registry installation and unloading, explicit binding
requirements, duplicate expert IDs, and converted-layer coverage.

INF-001 executes effective top-1 routes, INF-002 executes standalone
anchor-veto specialist pairs, and EVAL-002 records top-1 routes. Capacity
estimation, second-choice fallback, top-2 degradation, adaptive sparse
execution, compatibility-aware dispatch, top-2 telemetry, decision
serialization, package attachment, `RoutedMLP` top-2 integration, and
model-level top-1/top-2 configuration remain future work.
