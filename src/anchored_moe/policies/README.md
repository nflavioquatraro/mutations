# Policies

## Scope

The policies subpackage owns decisions made from already calibrated router
scores, already selected specialist outputs, and capacity metadata. POL-001
implements fixed-zero anchored hard top-1 selection. POL-002 ranks an
anchor-veto top-2 pair, POL-003 chooses adaptive anchor/top-1/top-2 route
arity, POL-004 chooses anchor/package top-1 and then an unconditional local
top-2 pair inside the winning package, and POL-005 chooses a globally ranked
top-2 pair restricted by explicit PKG-003 compatibility groups. POL-006
chooses a confidence-qualified
specialist prefix under an explicit per-token maximum-expert budget. COM-001
combines an approved specialist pair with score-softmax weights, COM-002
supplies equal-weight and normalized-positive-evidence baselines, and COM-003
uses a separately fitted frozen affine head to produce mixture weights without
changing pair selection. COM-004 keeps the dense anchor active and adds zero,
one, or two selected anchor-relative expert deltas. COM-005 instead merges
one or two selected PEFT/LoRA experts' effective projection parameters before
one logical MLP execution. BAL-001 rewrites
unavailable top-1 experts to the anchor. BAL-002 consumes an approved top-2
pair and degrades it deterministically to its available survivor or the
anchor; the same availability semantics apply to adaptive COM-004 routes.

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
| POL-004 | `selection.policy: hierarchical_package_top1_local_top2` | `hierarchical.py` | Validated |
| POL-005 | `selection.policy: global_compatibility_top2` | `selection.py` | Validated |
| POL-006 | `selection.policy: dynamic_k` | `selection.py` | Validated |
| COM-001 | `combination.policy: score_softmax` | `combination.py` | Validated |
| COM-002 | `combination.policy: equal_weight` | `combination.py` | Validated |
| COM-002 | `combination.policy: normalized_positive_evidence` | `combination.py` | Validated |
| COM-003 | `combination.policy: learned_mixture_weight` | `learned_combination.py` | Validated |
| COM-004 | `combination.policy: always_on_anchor_deltas` | `combination.py` | Validated |
| COM-005 | `combination.policy: parameter_space_lora_merge` | `parameter_merging.py` | Validated |
| BAL-001 | `capacity.overflow: anchor` | `capacity.py` | Validated |
| BAL-002 | `capacity.degradation: top2_to_top1` | `capacity.py` | Validated |

`SelectionPolicy` is the broad selection strategy interface.
`Top2SelectionPolicy` narrows it to policies that always return
pair-shaped `Top2RouteDecision` metadata.
`AnchoredTop1Selection` returns `RouteDecision`,
`AnchorVetoTop2Selection` returns pair-shaped `Top2RouteDecision`, and
`AdaptiveTop1Top2Selection` returns pair-ranked `AdaptiveRouteDecision`
metadata with explicit route arity. `DynamicKSelection` returns
`DynamicKRouteDecision` metadata whose ranking width is the configured
per-token compute bound and whose active prefix varies by confidence.
`CompatibilityConstrainedTop2Selection` binds one immutable
`MixtureCompatibilityIndex` and also returns `Top2RouteDecision`. INF-001
consumes the top-1 result, while INF-002 consumes either top-2 policy through
the narrow interface. COM-004 consumes adaptive routes through its explicit
delta dispatcher; arbitrary-\(k\) sparse execution remains separate.
`HierarchicalPackageTop1LocalTop2Selection` instead exposes a staged API:
`select_packages()` returns ordinary anchored `RouteDecision` metadata and
`select_locals()` reconstructs one `HierarchicalRouteDecision` containing both
the package audit trail and a flattened dispatch-compatible
`Top2RouteDecision`.

`OverflowToAnchorCapacity` consumes a `RouteDecision` and optional per-token
expert availability. `CapacityDecision` exposes the effective routes and the
tokens that overflowed.
`Top2ToTop1Capacity` consumes a `Top2RouteDecision`.
`Top2CapacityDecision` exposes packed effective execution slots, route arity,
and pair-shaped overflow metadata without changing the original decision.
`AdaptiveToAnchorCapacity` provides the same packed contract for POL-003
without activating specialists outside its retained pair.

`CombinationPolicy` is the activation-combination interface.
`ScoreSoftmaxCombination` consumes only already approved, already ordered
specialist pairs and returns `ActivationCombinationResult`, containing the
weighted activation and exact deployed weights. `EqualWeightCombination` and
`NormalizedPositiveEvidenceCombination` preserve the same result contract so
INF-002 can compare weighting strategies without changing selection, sparse
execution, or output reconstruction. `LearnedMixtureWeightHead` stores a
buffer-only affine map from an ordered selected-score pair to separate mixing
logits. `LearnedMixtureWeightCombination` deploys its normalized weights
through the same result contract. The head's standalone safetensors artifact
contains only finite coefficients and strict format metadata.
`AlwaysOnAnchorDeltaCombination` consumes one complete anchor activation plus
pair-shaped delta slots and returns `AnchorDeltaCombinationResult`.
`LoRAParameterSpaceMerge` consumes one or two compatible
`PEFTLoRAExpert` objects plus normalized per-token coefficients and returns
`ParameterSpaceMergeResult`. Unlike `CombinationPolicy`, it combines
parameters before execution rather than consuming already executed outputs.

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

For POL-004, package evidence is

\[
Q_t=[q_{t,1},\ldots,q_{t,P}],
\qquad q_{t,0}=0,
\]

and the first-stage package route is

\[
c_t=\operatorname*{argmax}_{p\in\{0,\ldots,P\}}q_{t,p}.
\]

The anchor is candidate zero and wins an exact zero tie. If \(c_t=0\), the
flattened expert pair is \((0,0)\). Otherwise RTR-005 supplies only package
\(c_t\)'s local scores

\[
L_{t,c_t}
=
[g_{t,c_t,1},\ldots,g_{t,c_t,K_{c_t}}],
\qquad K_{c_t}\ge2.
\]

POL-004 stable-sorts these specialist-only scores and returns their first two
indices. No second anchor comparison is applied:

\[
(e_{t,1},e_{t,2})
=
\operatorname{Top2Stable}(L_{t,c_t}).
\]

This is intentional. Package-local scores need only preserve within-package
order, so two negative local scores remain a valid pair after the package gate
has already approved specialization. Local indices are translated into one
globally unique package-major expert order before returning pair-shaped
metadata to INF-002.

For POL-006, let \(K_{\max}\ge1\) be the configured per-token specialist
budget and let \(\tau_{\mathrm{act}}\ge0\) be the confidence threshold.
After a stable descending specialist-only ranking, the active arity is

\[
a_t
=
\min\left(
K_{\max},
\sum_{i=1}^{K}
\mathbb{1}\left[g_{t,i}>\tau_{\mathrm{act}}\right]
\right).
\]

The active set is the first \(a_t\) entries of the retained ranking. Thus
\(a_t=0\) selects the anchor and \(1\le a_t\le K_{\max}\) selects a bounded
specialist prefix. The strict inequality makes an exact confidence-threshold
tie inactive. A non-negative threshold preserves the fixed-zero anchor
interpretation: selected specialists always carry strictly positive evidence.

`DynamicKRouteDecision` retains
\(\min(K,K_{\max})\) real candidates and pads the remaining ranking width with
candidate-zero and score-zero entries. Its tensors have shapes
\(D_1\times\cdots\times D_n\times K_{\max}\) for ranked indices and scores,
and \(D_1\times\cdots\times D_n\) for arity. The cap is a deterministic
per-token compute bound; it is not a batch-wide capacity allocation.

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

TRN-003 reuses this exact COM-001 tensor contract during local training. Its
package contains exactly two experts, so the selected pair is fixed and both
scores remain differentiable; anchor veto and capacity policies are not
evaluated inside that forced package-local training topology.

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

COM-003 deliberately separates the score used to select a pair from the logit
used to interpolate its outputs. For ordered selected scores
\(s_t=[g_{t,1},g_{t,2}]^\top\),

\[
z_t=w^\top s_t+b,
\qquad
\alpha_t=\operatorname{softmax}([z_t,0]),
\]

\[
y_t=\alpha_{t,1}E_1(h_t)+\alpha_{t,2}E_2(h_t).
\]

The fixed second logit makes the binary affine model identifiable. Pair order
comes from the selection policy, so \(\alpha_{t,1}\) always applies to the
first-ranked expert rather than to one global expert identity.

For COM-004, let
\(\Delta_{\ell,k}(h)=E_{\ell,k}(h)-A_\ell(h)\). After capacity handling,
effective route arity \(a'_t\in\{0,1,2\}\) gives

\[
y_t =
\begin{cases}
A_\ell(h_t), & a'_t=0,\\
A_\ell(h_t)+\Delta_{\ell,i}(h_t), & a'_t=1,\\
A_\ell(h_t)+\alpha_i\Delta_{\ell,i}(h_t)
              +\alpha_j\Delta_{\ell,j}(h_t), & a'_t=2.
\end{cases}
\]

Top-2 delta weights use score softmax; a top-1 survivor uses \([1,0]\), and an
anchor route uses \([0,0]\). Anchor routes select the original anchor
activation rather than relying on addition with zero.

For COM-005, LoRA expert \(k\) contributes the effective projection update
\(\Delta W_k=s_kB_kA_k\). For a capacity-approved ordered pair \(i,j\), the
frozen COM-003 head supplies \(\alpha_t\) and the reference executor uses

\[
W_t=W+\alpha_{t,i}\Delta W_i+\alpha_{t,j}\Delta W_j.
\]

Every targeted projection in the nonlinear MLP receives its corresponding
token-dependent \(W_t\) before the MLP executes. The implementation never
interpolates \(A\) and \(B\) separately, because doing so introduces unwanted
cross terms. Parameter-space merging is intentionally not assumed equal to
COM-003 activation-space interpolation.

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

For a POL-002 or POL-005 approved pair \((c_{t,1},c_{t,2})\), BAL-002
intersects the pair with explicit expert availability. Let \(q_{t,s}\) state
whether ranked slot \(s\) is available. The effective arity is

\[
a'_t = q_{t,1} + q_{t,2}.
\]

When \(a'_t=2\), both specialists retain their selected order. When
\(a'_t=1\), the survivor is packed into execution slot zero and receives
coefficient one. When \(a'_t=0\), the anchor executes. An anchor-vetoed pair
always remains arity zero and cannot report overflow because it never
requested specialist execution. BAL-002 never searches beyond the selected
pair.

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

Hierarchical package top-1/local top-2:

1. Apply POL-001 to finite package scores with the fixed-zero anchor prepended.
2. Ask RTR-005 to score only the flattened rows assigned to each winning
   non-anchor package.
3. Require one local score group for every active package and none for inactive
   packages; each group's indices must exactly equal the first-stage rows.
4. Validate the complete package-major topology, at least two local experts
   per package, and global expert-ID uniqueness.
5. Stable-sort each selected package's local scores without an anchor column or
   positivity threshold.
6. Translate the two local positions into flattened global candidate indices.
7. Preserve anchor rows as `[0, 0]` and return both package routes and the
   dispatch-compatible expert-pair decision.

Dynamic-\(k\) confidence and budget selection:

1. Apply the shared score, identifier, shape, and finite-value validation.
2. Require a positive integer maximum-expert budget and a finite non-negative
   activation threshold.
3. Stable-sort all specialists in descending score order.
4. Retain at most the configured maximum-expert count.
5. Count retained scores strictly greater than the activation threshold.
6. Use arity zero for anchor fallback and otherwise activate that many entries
   from the ranked prefix.
7. Pad an undersized or empty registry with inactive candidate-zero and
   score-zero entries.
8. Return ranked metadata, bounded route arity, arity masks, and active
   per-expert masks without executing or combining experts.

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

Learned mixture-weight combination:

1. Apply the shared combination shape, dtype, device, and finite validation.
2. Require the selected scores to match the frozen head's dtype and device.
3. Compute one affine first-expert mixing logit from the ordered score pair.
4. Append a fixed-zero second logit and apply stable pair softmax.
5. Broadcast, combine, and return output and exact deployed weights.

Always-on anchor-delta combination:

1. Require one complete anchor activation for every token.
2. Require pair-shaped deltas, scores, and route arity zero, one, or two.
3. Require every inactive delta slot to be exactly zero.
4. Assign zero weights to anchor routes and \([1,0]\) to top-1 routes.
5. Apply score softmax only to top-2 routes.
6. Add the weighted deltas to the anchor for active routes.
7. Select the unchanged anchor activation for arity-zero routes.

Parameter-space LoRA merging:

1. Require one or two selected `PEFTLoRAExpert` objects.
2. Resolve every effective LoRA update \(sBA\) in canonical module-name order.
3. Require identical resolved projection topology and byte-equal private/live
   anchor weights and biases.
4. Use unit weight for a capacity-degraded top-1 route or the frozen COM-003
   head for an ordered top-2 score pair.
5. Construct live-anchor projection overrides independently for each token.
6. Execute the live anchor with `torch.func.functional_call`; never mutate
   registered parameters.
7. Return exact applied weights and projection names with the merged output.

Capacity fallback:

1. Treat omitted availability as all experts available.
2. Gather availability for each selected expert in stable expert order.
3. Leave existing anchor routes unchanged.
4. Mark unavailable specialist selections in the overflow mask.
5. Replace their candidate index and selected score with anchor values of zero.
6. Return every token in a `CapacityDecision`; none are dropped.

Top-2 capacity degradation:

1. Treat omitted availability as all experts available.
2. Leave every anchor-containing selection at effective arity zero.
3. Gather availability only for approved specialist-pair slots.
4. Retain both available specialists in ranked order.
5. Promote a lone second-ranked survivor into effective slot zero.
6. Return the anchor when neither selected specialist is available.
7. Record unavailable approved occurrences in a pair-shaped overflow mask.

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
- POL-004 applies anchor abstention only at the package stage.
- POL-004 local top-2 is unconditional, stable, and restricted to two distinct
  experts owned by the selected package.
- POL-004 requires exact selected-row coverage from RTR-005 and rejects
  missing, extra, duplicated, reordered, cross-package, or dtype/device
  mismatched local score groups before dispatch.
- POL-004 flattens experts in package order while retaining aligned package
  ownership metadata; anchor rows remain `[0, 0]`.
- POL-004 does not score routers, execute experts, combine activations, fit
  package evidence, or create package metadata.
- POL-006 route arity is between zero and `max_experts`, inclusive.
- POL-006 activates specialists only on a strict threshold exceedance and
  preserves append order on exact score ties.
- POL-006 padding uses candidate zero and score zero but can never appear in
  the active ranking prefix.
- POL-006's `max_experts` is a per-token compute ceiling, not an availability
  guarantee or a batch capacity assignment.
- POL-006 does not learn difficulty, enforce capacity, execute experts,
  combine their outputs, or claim compatibility among the selected experts.
- POL-005 treats missing group metadata as non-mixable.
- POL-005 requires exact candidate-set equality with its immutable
  compatibility index; candidate order may be realigned by stable ID.
- POL-005 selects only distinct, same-group specialists with two strictly
  positive scores.
- POL-005 returns an anchor-containing fallback when no compatible pair is
  eligible.
- POL-005 pair and score ties preserve candidate append order.
- COM-001 receives specialists in configured `Top2SelectionPolicy` ranked
  order and never mixes the anchor.
- COM-001 requires a finite, strictly positive `mix_temperature`.
- COM-001 preserves the shared activation/score dtype and device.
- COM-001 weights are non-negative, finite, and sum to one.
- COM-002 equal weights are exactly one half and intentionally have no score
  gradient.
- COM-002 normalized-positive-evidence weights are non-negative, sum to one,
  and preserve gradients through strictly positive scores.
- A normalized-positive-evidence pair must contain at least one strictly
  positive selected score.
- COM-003 receives the exact ordered score pair that drove selection, but its
  learned affine coefficients affect only mixture weights.
- COM-003 deployment coefficients are immutable buffers with no trainable
  parameters and no storage alias to fitting state.
- The learned head requires score dtype/device equality. Callers move the
  buffer-only head explicitly with ordinary PyTorch `.to(...)` semantics.
- COM-003 does not infer oracle weights, execute counterfactual routes, or
  identify an optimal coefficient from one CAL-003 pair-loss observation.
- COM-004 always executes the anchor and accepts only anchor-relative experts.
- COM-004 anchor rows return the already-computed dense activation exactly.
- COM-004 top-1 weights are \([1,0]\); top-2 weights are non-negative and sum
  to one; inactive delta slots must be exactly zero.
- COM-005 accepts only one or two selected PEFT/LoRA experts over matching
  floating-point `torch.nn.Linear` projections.
- COM-005 validates every selected expert's private base exactly against the
  live anchor before execution.
- COM-005 merges graph-connected effective \(sBA\) matrices, not LoRA factors.
- COM-005 uses a frozen COM-003 head only for top-2 routes; a degraded top-1
  route receives unit weight and an anchor route receives no merge.
- A globally zero adapter set returns exact batched-anchor values while a
  straight-through functional graph preserves LoRA and hidden-state gradients.
- Adaptive availability fallback never activates an unrequested specialist,
  never searches beyond the retained pair, and returns to the anchor when no
  requested expert is available.
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
- BAL-002 only considers the approved pair and never searches the registry for
  another expert.
- BAL-002 effective arity is always zero, one, or two; inactive slots use
  candidate-zero and score-zero padding.
- Overflow never drops a token, and the anchor cannot reject overflow.

## Hugging Face integration

The policies operate on ordinary PyTorch tensors and do not depend on a
specific Transformers model family. INF-001 integrates top-1 with `RoutedMLP`
after `configure_anchored_top1()`, while COM-004 integrates adaptive or top-1
delta execution after `configure_always_on_anchor_deltas()`.
COM-005 integrates a pair-only Qwen3-compatible reference runtime through
`RoutedMLP.configure_parameter_space_lora_merge()`.

`AnchoredMoEConfig` continues to reject
`selection.policy: anchored_top1`, `selection.policy: anchor_veto_top2`,
`selection.policy: adaptive_top1_top2`, and
`selection.policy: hierarchical_package_top1_local_top2`,
`selection.policy: global_compatibility_top2`, and
`selection.policy: dynamic_k` for model-level configuration because runtime
strategy construction is not yet configuration-driven. Expert package
manifests may declare `anchored_top1`; top-2, adaptive, and dynamic-\(k\)
package attachment remain unsupported. Standalone experiments can use all
selection strategies explicitly. INF-002 composes either POL-002 or POL-005
with any `CombinationPolicy` and BAL-002 in a standalone sparse dispatcher.
POL-004 passes its flattened pair through
`AnchoredTop2Dispatcher.dispatch_preselected()`, which reuses the same BAL-002
capacity, combination, sparse execution, and reconstruction path without
pretending that staged package/local scores are one flat global score bank.
`capacity.degradation: top2_to_top1` documents that direct API strategy;
`AnchoredMoEConfig` does not accept it until top-2 model attachment and
configuration-driven policy construction exist. This does not silently enable
an incomplete model-level configuration.

## Configuration and examples

```python
from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AlwaysOnAnchorDeltaCombination,
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CompatibilityConstrainedTop2Selection,
    DynamicKSelection,
    EqualWeightCombination,
    HierarchicalPackageTop1LocalTop2Selection,
    LearnedMixtureWeightCombination,
    LoRAParameterSpaceMerge,
    MixtureCompatibilityIndex,
    NormalizedPositiveEvidenceCombination,
    OverflowToAnchorCapacity,
    ScoreSoftmaxCombination,
    Top2ToTop1Capacity,
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
capacity_adjusted = Top2ToTop1Capacity()(
    top2,
    expert_availability=availability,
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
dynamic = DynamicKSelection(
    max_experts=3,
    activation_threshold=0.25,
)(
    expert_scores,
    expert_ids=expert_ids,
)
anchor_tokens = dynamic.anchor_mask
three_expert_tokens = dynamic.mask_for_arity(3)
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
learned_mixture = LearnedMixtureWeightCombination(fitted_mixture_head)(
    pair_ordered_expert_outputs,
    selected_scores=approved_scores,
)
delta_mixture = AlwaysOnAnchorDeltaCombination(mix_temperature=1.0)(
    anchor_output,
    pair_ordered_deltas,
    selected_scores=effective_delta_scores,
    route_arity=effective_delta_arity,
)
parameter_merger = LoRAParameterSpaceMerge(fitted_mixture_head)

hierarchical_policy = HierarchicalPackageTop1LocalTop2Selection()
package_routes = hierarchical_policy.select_packages(
    hierarchical_router.score_packages(features, layer_index=layer_index),
    package_ids=hierarchical_router.package_ids_for_layer(layer_index),
)
local_groups = hierarchical_router.score_selected_locals(
    features,
    package_indices=package_routes.selected_indices,
    package_ids=package_routes.expert_ids,
    layer_index=layer_index,
)
hierarchical_routes = hierarchical_policy.select_locals(
    package_routes,
    local_groups,
    expert_ids_by_package=expert_ids_by_package,
)
```

The public strategy names are `anchored_top1`, `anchor_veto_top2`,
`adaptive_top1_top2`, `hierarchical_package_top1_local_top2`,
`global_compatibility_top2`, `dynamic_k`,
`score_softmax`, `equal_weight`, `normalized_positive_evidence`,
`learned_mixture_weight`, `always_on_anchor_deltas`,
`parameter_space_lora_merge`, and
`top2_to_top1`. See
`examples/verify_anchored_top1.py`, `examples/verify_anchor_veto_top2.py`,
`examples/verify_compatibility_top2.py`,
`examples/verify_adaptive_selection.py`,
`examples/verify_dynamic_k_selection.py`, and
`examples/verify_combination_baselines.py` for deterministic integration.
See `examples/verify_hierarchical_routing.py` for the staged RTR-005/POL-004
package gate, selected-row local scoring, negative-score local top-2, and
flattened route reconstruction.
See `examples/verify_learned_mixture_weights.py` for COM-003 fitting,
buffer-only deployment, activation combination, and safe reload.
See `examples/verify_always_on_anchor_deltas.py` for COM-004 anchor, top-1,
and adaptive top-2 LoRA-delta execution.
See `examples/verify_parameter_space_merge.py` for COM-005 effective-matrix
merging and tiny routed-layer execution.

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
  defines normalized exponentiation used for COM-001 and COM-003 mixture
  weights.
- [PyTorch `clamp` documentation](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html)
  defines the positive-evidence clipping operation used by COM-002.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines fixed-zero anchor evidence, anchored hard top-1, anchor-veto top-2,
  adaptive anchor/top-1/top-2, hierarchical package/local routing,
  dynamic-\(k\) allocation, score-softmax
  activation combination, both COM-002 baselines, separate learned COM-003
  mixing scores, and capacity-aware top-2 degradation.
- [Harder Tasks Need More Experts: Dynamic Routing in MoE Models](https://arxiv.org/abs/2403.07652)
  motivates varying the number of active experts with routing confidence.
  POL-006 uses an explicit calibrated-score threshold rather than the paper's
  learned dynamic routing mechanism.
- [DA-MoE: Towards Dynamic Expert Allocation for Mixture-of-Experts Models](https://arxiv.org/abs/2409.06669)
  motivates token-dependent expert counts. POL-006 provides deterministic
  selection metadata and a fixed per-token cap rather than learning token
  importance or a compute-allocation objective.
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
  motivates separating shared computation from routed specialist computation.
  COM-004 differs by retaining one pretrained dense MLP as an immutable
  always-on anchor and adding independently trained plug-in deltas.
- [Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745)
  motivates constructing one input-dependent expert in parameter space.
- [X-LoRA: Mixture of Low-Rank Adapter Experts](https://arxiv.org/abs/2402.07148)
  motivates layerwise token-dependent composition of frozen LoRA experts.
  COM-005 differs from both methods by using an already fitted frozen COM-003
  head, exactly one selected pair, and a per-token functional correctness
  backend rather than SMEAR training or X-LoRA's scaling network.

POL-001, POL-002, POL-003, POL-004, POL-005, COM-001, COM-002, COM-003,
BAL-001, and BAL-002 are framework infrastructure and do not derive a training
or optimization algorithm from a paper. COM-004 and POL-006 are
paper-motivated, with project-specific differences documented above. COM-005
is SMEAR/X-LoRA-inspired with the narrower reference-runtime differences
documented above.

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

COM-003 tests cover exact learned weights and activations, score and activation
gradients, frozen buffer-only deployment, coefficient validation, dtype/device
agreement, deterministic safetensors bytes, strict metadata and tensor
schemas, overwrite protection, exact reload, deterministic fitting, report
round trips, input immutability, CAL-003 plan alignment, malformed oracle
targets, unchanged top-2 selection, INF-002 execution, public docstrings, and
executable example behavior. Package-manifest integration, model-level
configuration, counterfactual weight-grid execution, pair-identity features,
and accelerator/distributed fitting remain future work.

COM-004 tests cover exact anchor fallback, top-1 and score-softmax top-2 delta
weights, inactive-slot rejection, adaptive availability degradation, sparse
expert calls, score/hidden/expert gradients, zero-update LoRA deltas, and a
tiny Qwen3 routed-layer integration. Package-driven delta attachment,
telemetry, native delta-only PEFT kernels, and arbitrary-\(k\) combination
remain future work.

COM-005 tests cover the effective \(sBA\) equation, token-dependent learned
weights, unit-weight endpoints, exact zero-update anchor fallback, gradients
to scores, hidden states, and both experts' LoRA factors, private-base drift,
malformed weights and expert counts, anchor/top-1/top-2 dispatch, ranked weight
alignment, frozen-head registration, immutable live anchor state, and tiny
Qwen3 `RoutedMLP` execution. Full-MLP merging, more than two experts,
package-driven LoRA attachment, static merged artifacts, fused kernels,
performance claims, and distributed execution remain unsupported.

BAL-002 tests cover both-available retention, first- and second-ranked
survivors, promotion into effective slot zero, neither-available anchor
fallback, anchor-veto preservation, pair-shaped overflow metadata, omitted
availability, validation before execution, sparse dispatcher call counts,
mixture-weight semantics, and public result invariants.

POL-003 tests cover all three route arities, exact anchor and mixture-threshold
ties, append-order ranking ties, active per-expert masks, empty and
single-expert padding, arbitrary leading dimensions, dtype/device preservation,
input immutability, invalid thresholds, malformed public decisions, and
affine-router integration.

POL-004 tests cover fixed-zero package selection, package-order ties,
unconditional local top-2 with negative scores, local-order ties, exact
selected-row coverage, global expert flattening, cross-package identity
rejection, anchor rows, source-score immutability, sparse preselected dispatch,
capacity/combination reuse, and the executable hierarchical example. The
initial policy requires rectangular RTR-005 topology and does not support
local top-1, a second local anchor veto, ragged package layers, prompt
shortlisting, or configuration-driven model attachment.

POL-006 tests cover arities from anchor through the configured budget, strict
confidence-threshold ties, global strongest-prefix retention, append-order
ties, active arity and per-expert masks, empty and undersized registry padding,
arbitrary leading dimensions, dtype/device and score-gradient preservation,
input immutability, invalid confidence and budget configuration, malformed
public decisions, and affine-router integration.

POL-005 tests cover incompatible high-score rejection, selection of a
lower-ranked compatible pair, lexicographic pair ranking, append-order ties,
missing-group anchor fallback, empty and single-expert registries, candidate
realignment, stale compatibility rejection, dtype/device preservation, input
immutability, registry installation and unloading, explicit binding
requirements, duplicate expert IDs, converted-layer coverage, sparse INF-002
execution, exact anchor fallback, gradient preservation, and validation before
candidate execution.

INF-001 executes effective top-1 routes. INF-002 executes capacity-adjusted
standalone anchor-veto, compatibility-constrained, or preselected
hierarchical routes, and EVAL-002 records top-1 routes. COM-004 executes
adaptive sparse routes only for anchor-relative experts. Capacity estimation,
fallback beyond the selected pair, arbitrary-\(k\) sparse execution,
arbitrary-\(k\) activation combination and compatibility, top-2 telemetry,
decision serialization, package attachment, `RoutedMLP` replacement-top-2
integration, and model-level routed strategy configuration remain future
work.
