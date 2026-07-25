# Evaluation

## Scope

The evaluation subpackage measures architectural invariants. The current scope
is dense equivalence and anchor-state immutability; benchmark quality, routing,
and efficiency metrics are deferred.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| EVAL-001 | `assert_tensors_equivalent`, `fingerprint_module` | `equivalence.py` | Validated |

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

## Algorithms

1. Execute a model in evaluation mode before conversion.
2. Fingerprint each selected anchor MLP.
3. Convert the selected layers.
4. Execute identical inputs without gradients.
5. Compare every hidden state and the final logits.
6. Re-fingerprint each wrapped anchor and compare digests.

## Assumptions and invariants

- Inputs, model mode, device, and dtype are identical between executions.
- Stochastic layers are disabled by evaluation mode.
- Comparison never mutates tensors or modules.
- A fingerprint proves tensor-state equality, not semantic compatibility.

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
