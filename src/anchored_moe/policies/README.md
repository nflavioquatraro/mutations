# Policies

## Scope

The policies subpackage owns decisions made from already calibrated router
scores. POL-001 implements fixed-zero anchored hard top-1 selection and returns
structured, per-token route metadata.

This subpackage does not extract router features, produce or fit expert scores,
attach experts, execute MLPs, combine outputs, enforce capacity, install
packages, or record telemetry. Those responsibilities remain with routers,
calibration, modeling and inference, combination, capacity, registry, and
evaluation features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| POL-001 | `selection.policy: anchored_top1` | `selection.py` | Validated |

`SelectionPolicy` is the strategy interface. `AnchoredTop1Selection` implements
the initial policy, and `RouteDecision` defines the stable result consumed by
future dispatch and evaluation components.

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
- The policy does not claim that the selected expert has capacity. BAL-001
  will define overflow-to-anchor behavior.

## Hugging Face integration

The policy operates on ordinary PyTorch tensors produced by router modules and
does not depend on a specific Transformers model family. It is intentionally
separate from `RoutedMLP`: selecting a candidate does not alter the exact
forced-anchor execution path until an explicit dispatch integration is added.

`AnchoredMoEConfig` continues to reject
`selection.policy: anchored_top1` for model execution because routed expert
dispatch is not implemented yet. Expert package manifests may declare
`anchored_top1` under `routing_policies_supported`, and standalone experiments
can construct the policy directly. This avoids silently accepting a model
configuration that would still execute only the anchor.

## Configuration and examples

```python
from anchored_moe import AnchoredTop1Selection

expert_ids = router.expert_ids_for_layer(layer_index)
expert_scores = router(features, layer_index=layer_index)

decision = AnchoredTop1Selection()(
    expert_scores,
    expert_ids=expert_ids,
)

anchor_tokens = decision.anchor_mask
code_tokens = decision.mask_for_expert("code:0")
```

The public strategy name is `anchored_top1`. See
`examples/verify_anchored_top1.py` for deterministic integration with identity
features and append-only affine heads.

## References

- [PyTorch `argmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html)
  defines first-maximum tie behavior used for anchor-first and append-order
  decisions.
- [PyTorch `gather` documentation](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
  defines collection of the selected per-token scores.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines fixed-zero anchor evidence, anchored hard top-1 selection, and the
  stable reference inference policy.

POL-001 is framework infrastructure and does not derive a training or
optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover mixed anchor/expert decisions, exact selected scores,
anchor masks, per-expert masks, anchor ties, expert ties, empty registries,
arbitrary leading dimensions, dtype and device preservation, repeatability,
input immutability, malformed identifiers, shape mismatches, non-floating and
non-finite scores, public result validation, affine-router integration, and
expert-package policy metadata.

The current implementation does not execute selected experts, check capacity,
log routes, serialize decisions, or expose model-integrated
`selection.policy: anchored_top1`. Those behaviors remain part of the routed
dispatch follow-up, BAL-001, EVAL-002, and later configuration integration.
