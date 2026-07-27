# Policies

## Scope

The policies subpackage owns decisions made from already calibrated router
scores and capacity metadata. POL-001 implements fixed-zero anchored hard
top-1 selection. BAL-001 rewrites unavailable selected experts to the anchor.

This subpackage does not extract router features, produce or fit expert scores,
attach experts, execute MLPs, combine outputs, estimate serving capacity,
install packages, or record telemetry. Those responsibilities remain with
routers, calibration, modeling and inference, combination, later capacity,
registry, and evaluation features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| POL-001 | `selection.policy: anchored_top1` | `selection.py` | Validated |
| BAL-001 | `capacity.overflow: anchor` | `capacity.py` | Validated |

`SelectionPolicy` is the strategy interface. `AnchoredTop1Selection` implements
the initial policy, and `RouteDecision` defines the stable result consumed by
INF-001 dispatch and EVAL-002 evaluation components.

`OverflowToAnchorCapacity` consumes a `RouteDecision` and optional per-token
expert availability. `CapacityDecision` exposes the effective routes and the
tokens that overflowed.

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
`selection.policy: anchored_top1` for model-level configuration because
registry installation does not yet materialize payloads or attach them to a
complete model.
Expert package manifests may declare `anchored_top1`, and standalone
experiments or routed wrappers can use it explicitly. This avoids silently
accepting an incomplete model configuration.

## Configuration and examples

```python
from anchored_moe import AnchoredTop1Selection, OverflowToAnchorCapacity

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
```

The public strategy name is `anchored_top1`. See
`examples/verify_anchored_top1.py` for deterministic integration with identity
features and append-only affine heads.

## References

- [PyTorch `argmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html)
  defines first-maximum tie behavior used for anchor-first and append-order
  decisions.
- [PyTorch `gather` documentation](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
  defines collection of selected scores and expert availability.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines fixed-zero anchor evidence, anchored hard top-1 selection, and the
  stable reference inference policy.

POL-001 and BAL-001 are framework infrastructure and do not derive a training
or optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover mixed anchor/expert decisions, exact selected scores,
anchor masks, per-expert masks, anchor ties, expert ties, empty registries,
arbitrary leading dimensions, dtype and device preservation, repeatability,
input immutability, malformed identifiers, shape mismatches, non-floating and
non-finite scores, public result validation, affine-router integration,
expert-package policy metadata, available routes, unavailable specialist
fallback, empty registries, availability shape/dtype validation, and no-token
dropping.

INF-001 executes effective routes and EVAL-002 records them. Capacity
estimation, second-choice fallback, top-2 degradation, decision serialization,
and model-level `selection.policy: anchored_top1` configuration remain future
work.
