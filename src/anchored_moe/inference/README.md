# Inference

## Scope

The inference subpackage owns token dispatch, output reconstruction, and
privacy-safe in-memory route telemetry. INF-001 connects already calibrated
router scores, anchored top-1 selection, BAL-001 capacity fallback, the
immutable anchor, and installed expert module objects for one routed layer.

It does not install packages, estimate serving capacity, combine top-2
experts, manage distributed all-to-all execution, or expose a complete
`AnchoredMoEForCausalLM`. Those remain separate registry, capacity,
combination, scaling, and model-integration features.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| INF-001 | `AnchoredTop1Dispatcher`, `RoutedMLP.configure_anchored_top1` | `dispatch.py`, `modeling.py` | Validated |
| EVAL-002 | `RouteTelemetry`, `RouteRecord`, `RouteSummary` | `telemetry.py` | Validated |

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
  included.

For \(N=\prod_i D_i\), router selection is \(O(NK)\), grouping is \(O(NK)\) in
the initial implementation, and each selected token executes exactly one MLP.

## Hugging Face integration

The Qwen3 backend already installs `RoutedMLP` around selected decoder MLPs.
INF-001 configures such a wrapper explicitly after expert and router objects
are constructed. The wrapper preserves the Qwen3 MLP tensor return contract,
so the surrounding Hugging Face decoder and causal-language-model forward
paths require no changes.

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

See `examples/verify_routed_top1.py` for a deterministic offline example.

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
when a selected expert is an anchor copy.

The initial implementation groups each candidate with a separate mask scan.
Optimized batched dispatch, token-aligned auxiliary MLP arguments, capacity
estimation, package-driven runtime attachment, route serialization,
model-level configuration, and top-2 combinations remain unsupported.
