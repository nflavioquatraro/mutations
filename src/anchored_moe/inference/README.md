# Inference

## Scope

The inference subpackage owns token dispatch, output reconstruction,
privacy-safe in-memory route telemetry, and explicit registry-to-model runtime
synchronization. INF-001 connects already calibrated router scores, anchored
top-1 selection, BAL-001 capacity fallback, the immutable anchor, and expert
module objects for one routed layer. PKG-007 safely materializes those objects
from active local packages.

It does not install packages implicitly, estimate serving capacity, combine
top-2 experts, manage distributed all-to-all execution, or expose a complete
`AnchoredMoEForCausalLM`. Those remain separate registry, capacity,
combination, scaling, and model-integration features.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| INF-001 | `AnchoredTop1Dispatcher`, `RoutedMLP.configure_anchored_top1` | `dispatch.py`, `modeling.py` | Validated |
| EVAL-002 | `RouteTelemetry`, `RouteRecord`, `RouteSummary` | `telemetry.py` | Validated |
| PKG-007 | `synchronize_anchored_top1_registry` | `composition.py`, `modeling.py` | Validated |
| TRN-001 | `RoutedMLP.configure_forced_expert` | `modeling.py` | Validated |

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

Dispatch:

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

`RoutedMLP` remains in exact `forced_anchor` mode until
`configure_anchored_top1()` is called explicitly. An empty configured expert
set constructs an empty score dimension and dispatches every token to the
anchor.

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
8. Build all layer runtimes and validate dtype, device, ordering, and model
   layer coverage before replacing any existing routed runtime.
9. Return a `RegistryCompositionReport` containing registry identity and
   applied expert order.

## Assumptions and invariants

- There is exactly one effective route per token and no token dropping.
- Anchor candidate zero is permanent.
- Experts execute in affine-router append order.
- Inactive experts are not executed.
- Overflow executes the anchor, not another specialist.
- The feature extractor and router heads are frozen at inference attachment.
- Existing expert gradient flags are preserved.
- Anchor and expert outputs must share trailing shape, dtype, and device.
- Input token order is restored deterministically.
- Telemetry is opt-in, bounded, detached, copied to CPU, and contains no model
  inputs or activations.
- Additional MLP arguments are forwarded unchanged. Token-aligned auxiliary
  tensors requiring sparse indexing are not supported by INF-001.
- Package-driven attachment, distributed dispatch, and top-2 execution are not
  separate concerns; PKG-007 implements local explicit attachment while
  distributed dispatch and top-2 remain excluded.
- Registry installation and unloading never mutate a model. Synchronization is
  a distinct, explicit API call.
- Active package order and serialized router-head order determine runtime
  append order.
- Binding keys and router keys must match exactly.
- Missing bindings, duplicate expert IDs, unsupported strategies, layer
  mismatches, or incompatible tensor schemas fail before wrapper mutation.
- Synchronizing an empty registry removes loaded experts and restores exact
  forced-anchor calls.
- Forced-expert mode has exactly one expert, no router heads, and no capacity
  or selection decision.

For \(N=\prod_i D_i\), router selection is \(O(NK)\), grouping is \(O(NK)\) in
the initial implementation, and each selected token executes exactly one MLP.

## Hugging Face integration

The Qwen3 backend installs `RoutedMLP` around selected decoder MLPs. INF-001
configures such a wrapper explicitly after expert and router objects are
constructed. PKG-007 uses the same model-family-neutral backend interface to
discover wrappers and synchronize an active local registry. The wrapper
preserves the Qwen3 MLP tensor return contract, so surrounding Hugging Face
forward and generation paths require no changes.

`AnchoredMoEConfig` still rejects model-level `anchored_top1` configuration.
Package installation and complete model composition are required before that
declarative model configuration can be executable rather than silently
incomplete.

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

See `examples/verify_routed_top1.py` and
`examples/verify_package_runtime.py` for deterministic offline examples.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines anchored top-1 inference, exact fallback, stable ordering, and
  overflow-to-anchor behavior.
- [PyTorch `index_select` documentation](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html)
  defines sparse token gathering.
- [PyTorch `index_copy` documentation](https://docs.pytorch.org/docs/stable/generated/torch.index_copy.html)
  defines differentiable output reconstruction.
- [PyTorch `ModuleList` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
  motivates registering ordered expert modules without encoding expert IDs as
  Python attribute names.

INF-001 and EVAL-002 are framework infrastructure and do not derive a training
or optimization algorithm from a paper.

## Tests and known limitations

Acceptance tests cover mixed anchor/expert routes, sparse candidate call
counts, deterministic reconstruction, overflow, empty registries, gradient
flow, validation before execution, bounded telemetry, aggregate counts,
privacy-sensitive field exclusion, and exact Qwen3 hidden-state/logit equality
when a selected expert is an anchor copy. PKG-007 adds package-driven
multi-layer materialization, exact removal fallback, deterministic reinsertion,
and malformed router/binding failure before model mutation.

The initial implementation groups each candidate with a separate mask scan.
Optimized batched dispatch, token-aligned auxiliary MLP arguments, capacity
estimation, live-model checkpoint hash verification, route serialization,
automatic install hooks, model-level configuration, multi-device
materialization, and top-2 combinations remain unsupported.
