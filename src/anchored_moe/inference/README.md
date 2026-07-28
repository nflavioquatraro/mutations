# Inference

## Scope

The inference subpackage owns token dispatch, output reconstruction,
privacy-safe in-memory route telemetry, and explicit registry-to-model runtime
synchronization. INF-001 connects already calibrated router scores, anchored
top-1 selection, BAL-001 capacity fallback, the immutable anchor, and expert
module objects for one routed layer. INF-002 connects any pair-shaped
`Top2SelectionPolicy`, sparse specialist execution, and configurable
activation combination through a standalone model-family-neutral dispatcher.
BAL-002 adds explicit per-token availability and deterministic top-2 to top-1
or anchor degradation to that dispatcher.
POL-002 supplies ordinary anchor-veto selection and POL-005 supplies
compatibility-constrained selection. COM-001 score-softmax remains the
default, COM-002 supplies two baselines, and COM-003 supplies separately
learned frozen mixture weights. PKG-007 safely materializes
top-1 runtime objects from active local packages. During that materialization,
RTR-003 compiles package-owned independent affine heads into one frozen
logical score bank per routed layer.

It does not install packages implicitly, estimate serving capacity, attach
top-2 dispatch to `RoutedMLP`, construct compatibility metadata, or manage
distributed all-to-all execution. CORE-005 exposes the existing explicit
composition boundary through
`AnchoredMoEForCausalLM.synchronize_registry`; inference remains responsible
for materialization rather than package lifecycle. CORE-006 persists the
already synchronized anchored-top-1 composition and invokes the same explicit
materialization boundary while restoring a saved checkpoint.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| INF-001 | `AnchoredTop1Dispatcher`, `RoutedMLP.configure_anchored_top1` | `dispatch.py`, `modeling.py` | Validated |
| INF-002, BAL-002 | `AnchoredTop2Dispatcher`, `Top2DispatchResult` | `dispatch.py` | Validated |
| EVAL-002 | `RouteTelemetry`, `RouteRecord`, `RouteSummary` | `telemetry.py` | Validated |
| PKG-007 | `synchronize_anchored_top1_registry` | `composition.py`, `modeling.py` | Validated |
| TRN-001 | `RoutedMLP.configure_forced_expert` | `modeling.py` | Validated |
| CORE-005 | `AnchoredMoEForCausalLM.synchronize_registry` | `modeling.py` | Validated |
| CORE-006 | synchronized composition save/reload | `modeling.py`, `registry/composition.py` | Validated |

## Mathematical formulation

For token-leading hidden states
\(h\in\mathbb{R}^{D_1\times\cdots\times D_n\times H}\), calibrated scores
\(G\in\mathbb{R}^{D_1\times\cdots\times D_n\times K}\), and effective route
\(r'_t\), dispatch computes

\[
y_t =
\begin{cases}
A(h_t), & r'_t=0,\\
E_{r'_t}(h_t), & r'_t\in\{1,\ldots,K\}.
\end{cases}
\]

Every token belongs to exactly one candidate group. Candidate zero is always
the anchor, while candidate \(k\) corresponds to `expert_ids[k - 1]`. The
output preserves the complete token-leading shape and uses the common trailing
shape returned by the anchor and experts.

For INF-002, the configured POL-002 or POL-005 policy returns ranked candidates
\((c_{t,1},c_{t,2})\) and the approved specialist-pair indicator

\[
m_t=\mathbb{1}[c_{t,1}\ne0\land c_{t,2}\ne0].
\]

BAL-002 intersects an approved pair with explicit availability and produces
effective arity \(a'_t\in\{0,1,2\}\). Dispatch computes

\[
y_t =
\begin{cases}
A(h_t), & a'_t=0,\\
E_{c'_{t,1}}(h_t), & a'_t=1,\\
\alpha_{t,1}E_{c_{t,1}}(h_t)
+\alpha_{t,2}E_{c_{t,2}}(h_t), & a'_t=2.
\end{cases}
\]

The configured `CombinationPolicy` computes
\(\left[\alpha_{t,1},\alpha_{t,2}\right]\). COM-001 uses score-softmax,
COM-002 can use fixed equal weights or normalized positive evidence, and
COM-003 derives a separate mixing logit from the ordered selected scores.
All policies return non-negative, sum-to-one weights in the shared activation
dtype and device.

Every effective anchor token executes the anchor once. An arity-one token
executes its available survivor directly, and an arity-two token executes two
distinct specialists. Each specialist is gathered into one sparse batch
regardless of its original rank. `Top2DispatchResult.mixture_weights` has
shape \(D_1\times\cdots\times D_n\times2\): arity-two tokens contain
\(\alpha_t\), arity-one tokens contain \([1,0]\), and anchor tokens contain
\([0,0]\).

Telemetry records selected candidate indices, effective candidate indices,
selected scores, and overflow flags:

\[
R_\ell =
(\ell, r, g_r, r', o, \text{expert IDs}).
\]

It never records \(h\), token IDs, token text, logits, or expert activations.

For active packages \(P=[p_1,\ldots,p_n]\), package-local serialized router
order, and layer \(\ell\), composition constructs the deterministic expert
order

\[
K_\ell =
K_{\ell,p_1}\mathbin{\Vert}\cdots\mathbin{\Vert}K_{\ell,p_n}.
\]

The router keys and explicit payload-binding keys must be equal before any
routed wrapper changes:

\[
\operatorname{keys}(G_p)=\operatorname{keys}(B_p).
\]

## Algorithms

Anchored top-1 dispatch:

1. Validate score shape, expert count, devices, and module types before
   executing a candidate.
2. Apply anchored hard top-1 selection.
3. Apply overflow-to-anchor capacity fallback.
4. Flatten only token-leading dimensions.
5. Visit the anchor and experts in stable candidate order.
6. Gather each non-empty candidate's hidden states and execute that MLP once.
7. Require all candidate outputs to share trailing shape, dtype, and device.
8. Reconstruct output order with differentiable `torch.index_copy`.
9. Optionally copy route metadata into a bounded CPU telemetry history.

Pair-shaped top-2 dispatch:

1. Validate score shape, score/activation dtype, expert count, devices,
   forwarding containers, and policy types before executing any candidate.
2. Apply the configured `Top2SelectionPolicy`; require pair-shaped metadata
   with unchanged candidate IDs, token shape, score dtype, and device.
3. Apply BAL-002 to explicit availability. Retain both available specialists,
   promote a lone survivor to slot zero, or fall back to the anchor.
4. Gather all effective anchor tokens and execute the anchor in one sparse
   call.
5. For each specialist in stable append order, find effective occurrences,
   gather their hidden states, and execute one sparse call.
6. Reconstruct specialist outputs in exact effective slot order with
   differentiable `torch.index_copy`.
7. Copy arity-one specialist outputs directly with weights \([1,0]\).
8. Apply the configured COM-001, COM-002, or COM-003 policy only to arity-two
   scores and pair-ordered activations.
9. Reconstruct anchor, top-1, and combined outputs in original token order.
10. Return original routes, capacity metadata, and full-shape weights.

`RoutedMLP` remains in exact `forced_anchor` mode until
`configure_anchored_top1()` is called explicitly. An empty configured expert
set constructs an empty score dimension and dispatches every token to the
anchor. INF-002 is standalone and does not add another `RoutedMLP` mode.

For training, `configure_forced_expert()` provides a separate router-free mode
in which every token executes one expert directly. The training subpackage
owns topology validation and parameter isolation; inference dispatch and
selection are bypassed.

Registry synchronization:

1. Discover every converted `RoutedMLP` through the selected backend.
2. For an empty active registry, explicitly reset all wrappers to the exact
   forced-anchor path.
3. Revalidate every active package's live manifest, hashes, and payloads.
4. Require supported full-MLP, identity-feature, affine-head, and anchored
   top-1 strategy metadata.
5. Parse the explicit payload bindings; never derive ownership from filenames.
6. Load and validate every router bank and require exact router/binding keys.
7. Materialize every expert against its layer's immutable anchor structure.
8. Compile each complete layer's package-owned affine heads into a frozen
   RTR-003 logical router with copied, non-aliased buffers.
9. Build all layer runtimes and validate dtype, device, ordering, and model
   layer coverage before replacing any existing routed runtime.
10. Return a `RegistryCompositionReport` containing registry identity and
   applied expert order.

## Assumptions and invariants

- INF-001 executes exactly one effective candidate per token.
- INF-002 executes the anchor, one available selected specialist, or two
  available distinct selected specialists per token.
- Neither dispatcher drops a token.
- Anchor candidate zero is permanent.
- Experts execute in affine-router append order.
- Inactive experts are not executed.
- Pair overflow degrades to the selected survivor, then to the anchor; it
  never searches beyond the selected pair.
- INF-002 vetoes specialist execution whenever the anchor appears in the
  ranked pair.
- INF-002 preserves first/second ranked order when assigning configured
  combination weights.
- INF-002 emits zero specialist weights for effective anchor tokens,
  \([1,0]\) for effective top-1 tokens, and non-negative sum-to-one weights for
  effective top-2 tokens.
- INF-002 checks explicit availability but does not estimate capacity. Pair
  compatibility is enforced only by a configured compatibility-constrained
  policy; the dispatcher never infers it.
- The feature extractor and `RouterScoreBank` are frozen at inference
  attachment.
- Existing expert gradient flags are preserved.
- Anchor and expert outputs must share trailing shape, dtype, and device.
- Input token order is restored deterministically.
- Telemetry is opt-in, bounded, detached, copied to CPU, and contains no model
  inputs or activations.
- Additional MLP arguments are forwarded unchanged. Token-aligned auxiliary
  tensors requiring sparse indexing are not supported by INF-001 or INF-002.
- PKG-007 implements local explicit top-1 attachment. Top-2 package
  attachment, `RoutedMLP` integration, capacity estimation, telemetry,
  compatibility-index construction, and distributed dispatch remain separate
  features.
- Registry installation and unloading never mutate a model. Synchronization is
  a distinct, explicit API call.
- Active package order and serialized router-head order determine runtime
  append order.
- Package payloads retain RTR-002 independent-head serialization; the active
  registry uses an RTR-003 compiled copy with no source-storage aliases.
- Binding keys and router keys must match exactly.
- Missing bindings, duplicate expert IDs, unsupported strategies, layer
  mismatches, or incompatible tensor schemas fail before wrapper mutation.
- Synchronizing an empty registry removes loaded experts and restores exact
  forced-anchor calls.
- Forced-expert mode has exactly one expert, no router heads, and no capacity
  or selection decision.

For \(N=\prod_i D_i\), router selection and initial mask-scan grouping are
\(O(NK)\). INF-001 executes one MLP per token. INF-002 executes one MLP for an
anchor or effective top-1 token and two MLPs for an effective top-2 token;
combination is linear in the selected activation size.

## Hugging Face integration

The Qwen3 backend installs `RoutedMLP` around selected decoder MLPs. INF-001
configures such a wrapper explicitly after expert and router objects are
constructed. PKG-007 uses the same model-family-neutral backend interface to
discover wrappers and synchronize an active local registry. The wrapper
preserves the Qwen3 MLP tensor return contract, so surrounding Hugging Face
forward and generation paths require no changes.

INF-002 operates on ordinary PyTorch modules and tensors, so affine router
heads and `FullMLPExpert` objects can be exercised without a model-family
adapter. Its dispatcher accepts model-family-neutral `Top2SelectionPolicy`
and `CombinationPolicy` interfaces plus BAL-002 explicit availability,
defaulting to POL-002, COM-001, and `Top2ToTop1Capacity`.
POL-005 can instead consume a PKG-003 compatibility index supplied explicitly
by the caller. The dispatcher is deliberately not exposed through `RoutedMLP`,
`AnchoredMoEForCausalLM`, registry synchronization, or declarative
configuration yet.

CORE-005 wraps the native causal LM in a `PreTrainedModel` facade. Its
`forward()` and `generate()` delegate to that native implementation, while
`install()` and `unload()` change only its `ExpertRegistry`.
`synchronize_registry()` is the explicit operation that applies PKG-007.
`AnchoredMoEConfig` still rejects declarative model-level `anchored_top1`
configuration because synchronization, rather than configuration parsing,
currently proves that all required package payloads exist.

CORE-006 requires this explicit synchronization before saving. Loading a
composition validates its package snapshot first, constructs the model around
the referenced external anchor, and then calls the same synchronization path
to restore the saved behavior. It does not change the rule that later install,
unload, upgrade, and rollback operations require another explicit
`synchronize_registry()` call.

## Configuration and examples

```python
telemetry = RouteTelemetry(layer_index=3)
routed_mlp.configure_anchored_top1(
    layer_index=3,
    router_feature_extractor=features,
    router_heads=heads,
    experts={"code:0": code_expert},
    route_telemetry=telemetry,
)

output = routed_mlp(
    hidden_states,
    expert_availability=availability,
)
print(telemetry.summarize())
```

Standalone top-2 dispatch:

```python
result = AnchoredTop2Dispatcher(
    combination_policy=ScoreSoftmaxCombination(mix_temperature=1.0),
).dispatch(
    hidden_states,
    expert_scores=scores,
    expert_ids=("code:0", "math:0"),
    anchor_mlp=anchor,
    experts=(code_expert, math_expert),
    expert_availability=availability,
)

output = result.output
mixed_weights = result.mixture_weights[
    result.capacity.top2_mask
]
degraded_tokens = result.capacity.top1_mask

# Use EqualWeightCombination() or
# NormalizedPositiveEvidenceCombination() for COM-002 comparisons.
# Use LearnedMixtureWeightCombination(fitted_head) for COM-003.
```

Compatibility-aware top-2 uses the same dispatcher:

```python
compatibility = registry.mixture_compatibility(layer_index=18)
result = AnchoredTop2Dispatcher(
    selection_policy=CompatibilityConstrainedTop2Selection(compatibility),
).dispatch(
    hidden_states,
    expert_scores=scores,
    expert_ids=compatibility.expert_ids,
    anchor_mlp=anchor,
    experts=experts_in_compatibility_order,
)
```

Explicit package composition:

```python
registry.install(package)
report = synchronize_anchored_top1_registry(
    model,
    backend=Qwen3Backend(),
    registry=registry,
)

registry.unload(package.identity)
synchronize_anchored_top1_registry(
    model,
    backend=Qwen3Backend(),
    registry=registry,
)
```

See `examples/verify_routed_top1.py`, `examples/verify_top2_dispatch.py`,
`examples/verify_compatibility_top2.py`,
`examples/verify_combination_baselines.py`,
`examples/verify_learned_mixture_weights.py`, and
`examples/verify_package_runtime.py` for deterministic offline examples. The
last uses `AnchoredMoEForCausalLM` for the complete public top-1 composition
lifecycle. The NB-007 `examples/export_reload_lifecycle.py` companion and
`notebooks/06_export_and_reload.ipynb` extend that workflow through composition
reload, explicit removal and reinsertion synchronization, package upgrade, and
exact rollback.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines anchored top-1 inference, anchor-veto weighted top-2, exact fallback,
  stable ordering, and overflow-to-anchor behavior.
- [PyTorch `index_select` documentation](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html)
  defines sparse token gathering.
- [PyTorch `index_copy` documentation](https://docs.pytorch.org/docs/stable/generated/torch.index_copy.html)
  defines differentiable output reconstruction.
- [PyTorch `ModuleList` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
  motivates registering ordered expert modules without encoding expert IDs as
  Python attribute names.
- [PyTorch `softmax` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  defines the normalized score weights applied after ranked pair
  reconstruction.

INF-001, INF-002, BAL-002, and EVAL-002 are framework infrastructure and do
not derive a training or optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover mixed anchor/expert routes, sparse candidate call
counts, deterministic reconstruction, overflow, empty registries, gradient
flow, validation before execution, bounded telemetry, aggregate counts,
privacy-sensitive field exclusion, and exact Qwen3 hidden-state/logit equality
when a selected expert is an anchor copy. PKG-007 adds package-driven
multi-layer materialization, exact removal fallback, deterministic reinsertion,
and malformed router/binding failure before model mutation. CORE-006 adds
stale-runtime rejection and exact routed-logit, route-mode, expert-order, and
registry-hash restoration after local save/reload.

INF-002 acceptance tests cover mixed anchor/pair decisions, exact ranked
mixtures, sparse call counts, inactive specialists, empty and single-expert
registries, arbitrary token-leading shapes, no token dropping, configurable
temperature and combination policy, COM-001/COM-002/COM-003 weights,
score/hidden/expert gradients, structural validation before execution,
candidate output contracts, affine-router and full-MLP integration, public
result invariants, compatibility-constrained sparse execution, registry-index
integration, policy-result validation before execution, both/one/neither
availability, second-ranked survivor promotion, effective route arity,
pair-shaped overflow metadata, and preservation of all INF-001 tests.

The initial implementation groups each candidate with a separate mask scan.
Optimized batched dispatch, token-aligned auxiliary MLP arguments, capacity
estimation, fallback beyond the selected pair, live-model checkpoint hash
verification, route serialization,
automatic install hooks, model-level configuration, multi-device
materialization, top-2 telemetry, `RoutedMLP`
top-2 integration, and package-driven top-2 attachment remain unsupported.
