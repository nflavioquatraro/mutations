# Evaluation

## Scope

The evaluation subpackage measures architectural invariants. The current scope
includes dense equivalence, anchor-state immutability, exact route agreement,
and package removal/reinsertion lifecycle checks. Benchmark quality and
efficiency metrics are deferred.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| EVAL-001 | `assert_tensors_equivalent`, `fingerprint_module` | `equivalence.py` | Validated |
| EVAL-002 | `compare_route_decisions`, `assert_route_decisions_equal` | `routing.py` | Validated |
| EVAL-003 | `evaluate_registry_lifecycle` | `lifecycle.py` | Validated |

## Mathematical formulation

For original output \(y\) and converted output \(\hat y\):

\[
d_{\mathrm{abs}}=\max_i|\hat y_i-y_i|,
\qquad
d_{\mathrm{rel}}=\max_i
\frac{|\hat y_i-y_i|}{\max(|y_i|,\epsilon)}.
\]

Exact equivalence additionally requires `torch.equal(y_hat, y)`. Tests declare
both absolute and relative tolerances explicitly rather than hiding a global
default.

Anchor immutability is checked with a SHA-256 digest over sorted state-dict
names, shapes, dtypes, and tensor bytes.

For actual routes \(r\) and reference routes \(r^\star\) under the same stable
expert ordering, exact route agreement is

\[
\operatorname{agreement}(r,r^\star)
=
\frac{1}{N}\sum_{t=1}^{N}\mathbb{1}[r_t=r^\star_t].
\]

`exact` is true only when the agreement is one. Two empty decisions have
agreement one. Selected scores are excluded because route agreement measures
executed candidates; calibration metrics remain separate.

For installed registry state \(S\), removed state \(S^{-}\), and reinserted
state \(S^{+}\), exact lifecycle restoration requires

\[
d(S^{-})\ne d(S),
\qquad
|S^{-}|=|S|-1,
\qquad
d(S^{+})=d(S),
\qquad
P(S^{+})=P(S),
\]

while the compatibility identity remains unchanged across all three states.

## Algorithms

1. Execute a model in evaluation mode before conversion.
2. Fingerprint each selected anchor MLP.
3. Convert the selected layers.
4. Execute identical inputs without gradients.
5. Compare every hidden state and the final logits.
6. Re-fingerprint each wrapped anchor and compare digests.

Route agreement:

1. Require identical token shapes and expert ID ordering.
2. Detach and copy candidate indices to CPU.
3. Count identical per-token candidate assignments.
4. Return exactness, matching count, total count, and agreement fraction.
5. Optionally assert exactness with named diagnostics.

Registry lifecycle evaluation:

1. Capture the installed registry snapshot.
2. Unload one exact package identity and capture the removed snapshot.
3. Revalidate and reinstall identical package content.
4. Compare active counts, ordered identities, registry hashes, and
   compatibility binding.
5. Report each invariant separately and expose aggregate `exact` status.

## Assumptions and invariants

- Inputs, model mode, device, and dtype are identical between executions.
- Stochastic layers are disabled by evaluation mode.
- Comparison never mutates tensors or modules.
- A fingerprint proves tensor-state equality, not semantic compatibility.
- Route indices are comparable only under identical expert ID ordering.
- Route comparison never records token content, hidden states, or activations.
- Lifecycle evaluation consumes immutable snapshots and never mutates a
  registry.

## Hugging Face integration

The integration test uses a tiny locally initialized `Qwen3ForCausalLM` and
requests `output_hidden_states=True`. No model or tokenizer download is needed.

## Configuration and examples

```python
result = assert_tensors_equivalent(
    converted.logits,
    dense.logits,
    name="logits",
    atol=0.0,
    rtol=0.0,
)
assert result.exact

route_result = assert_route_decisions_equal(
    actual_routes,
    expected_routes,
    name="layer-3",
)

lifecycle = evaluate_registry_lifecycle(
    installed_snapshot,
    removed_snapshot,
    reinserted_snapshot,
)
assert lifecycle.exact
```

`notebooks/01_convert_dense_model.ipynb` exercises this API against logits,
every hidden state, and the selected anchor MLP fingerprints. It also reports
the post-conversion trainable-parameter count.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  specifies exact fallback and forced-anchor equivalence.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  motivates explicit tolerance declarations for future runtime variants.

No paper-derived evaluation algorithm is implemented here.

## Tests and known limitations

CPU tests require exact equality because the wrapper executes the same MLP
object with no additional arithmetic. Tolerance-based accelerator and optimized
kernel checks will be added with functional compatibility support.

Route tests cover exact and partial agreement, diagnostics, registry-order
validation, configured Qwen3 expert execution, and privacy-safe telemetry.
Lifecycle tests cover removal identity changes, exact package-count reduction,
reinserted hash/order restoration, and persistent compatibility binding.
Cross-layer trajectory metrics, first-divergence depth, route serialization,
persistent registry evaluation, and distributed aggregation remain future work.
