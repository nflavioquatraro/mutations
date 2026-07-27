# Policies

## Scope

The policies subpackage owns decisions made from already calibrated router
scores, already selected specialist outputs, and capacity metadata. POL-001
implements fixed-zero anchored hard top-1 selection. POL-002 ranks an
anchor-veto top-2 pair, COM-001 combines an approved specialist pair with
score-softmax weights, and BAL-001 rewrites unavailable top-1 experts to the
anchor.

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
| COM-001 | `combination.policy: score_softmax` | `combination.py` | Validated |
| BAL-001 | `capacity.overflow: anchor` | `capacity.py` | Validated |

`SelectionPolicy` is the selection strategy interface.
`AnchoredTop1Selection` returns `RouteDecision`, while
`AnchorVetoTop2Selection` returns pair-shaped `Top2RouteDecision` metadata.
The top-1 result is consumed by INF-001 dispatch and EVAL-002 evaluation
components. Top-2 dispatch remains separate work.

`OverflowToAnchorCapacity` consumes a `RouteDecision` and optional per-token
expert availability. `CapacityDecision` exposes the effective routes and the
tokens that overflowed.

`CombinationPolicy` is the activation-combination interface.
`ScoreSoftmaxCombination` consumes only already approved, already ordered
specialist pairs and returns `ActivationCombinationResult`, containing the
weighted activation and exact deployed weights.

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

Score-softmax combination:

1. Require finite pair scores and matching pair-ordered activation prefixes.
2. Require scores and activations to share dtype and device so combination
   cannot silently promote the model output.
3. Divide selected scores by the finite positive mixture temperature.
4. Apply numerically stable `torch.softmax` over the pair dimension.
5. Broadcast weights across all trailing activation dimensions.
6. Sum the two weighted activations and return both output and weights.

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
- COM-001 receives specialists in POL-002 ranked order and never mixes the
  anchor.
- COM-001 requires a finite, strictly positive `mix_temperature`.
- COM-001 preserves the shared activation/score dtype and device.
- COM-001 weights are non-negative, finite, and sum to one.
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
`selection.policy: anchored_top1` and `selection.policy: anchor_veto_top2` for
model-level configuration because runtime strategy construction is not yet
configuration-driven. Expert package manifests may declare `anchored_top1`;
top-2 package attachment remains unsupported. Standalone experiments can use
both selection strategies explicitly, and INF-002 composes POL-002 with
COM-001 in a standalone sparse dispatcher. This avoids silently accepting an
incomplete model configuration.

## Configuration and examples

```python
from anchored_moe import (
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
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
approved_scores = top2.selected_scores[top2.expert_pair_mask]
mixed = ScoreSoftmaxCombination(mix_temperature=1.0)(
    pair_ordered_expert_outputs,
    selected_scores=approved_scores,
)
```

The public strategy names are `anchored_top1`, `anchor_veto_top2`, and
`score_softmax`. See `examples/verify_anchored_top1.py` and
`examples/verify_anchor_veto_top2.py` for deterministic integration with
identity features and append-only affine heads.

## References

- [PyTorch `argmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html)
  defines first-maximum tie behavior used for anchor-first and append-order
  decisions.
- [PyTorch `gather` documentation](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
  defines collection of selected scores and expert availability.
- [PyTorch `argsort` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argsort.html)
  defines the stable descending ordering used for deterministic top-2 ties.
- [PyTorch `softmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  defines normalized exponentiation used for COM-001 mixture weights.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines fixed-zero anchor evidence, anchored hard top-1, anchor-veto top-2,
  and score-softmax activation combination.

POL-001, POL-002, COM-001, and BAL-001 are framework infrastructure and do not
derive a training or optimization algorithm from a paper.

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
immutability, and malformed public construction.

INF-001 executes effective top-1 routes, INF-002 executes standalone
anchor-veto specialist pairs, and EVAL-002 records top-1 routes. Capacity
estimation, second-choice fallback, top-2 degradation, pair compatibility,
top-2 telemetry, decision serialization, package attachment, `RoutedMLP`
top-2 integration, and model-level top-1/top-2 configuration remain future
work.
