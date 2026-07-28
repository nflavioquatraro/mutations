# Backends

## Scope

The backend subpackage locates transformer layers and MLPs, validates the
model-family contract, and replaces selected MLPs with model-neutral routed
wrappers. It does not own routing decisions, expert weights, calibration, or
package lifecycle.

## Implemented strategies

| Feature ID | Configuration | Module | Status |
|---|---|---|---|
| CORE-002 | `ModelBackend` interface | `base.py` | Validated |
| CORE-003 | `backend: qwen3` | `qwen3.py` | Validated |
| CORE-005 | `AnchoredMoEForCausalLM` conversion | `../modeling.py` | Validated |

The Qwen3 adapter supports both `Qwen3ForCausalLM` and bare `Qwen3Model`
decoder-layer paths.

## Mathematical formulation

For a selected decoder layer \(\ell\), conversion replaces the owning module
reference:

\[
\operatorname{layer}_\ell.\operatorname{mlp}=A_\ell
\quad\longrightarrow\quad
\operatorname{layer}_\ell.\operatorname{mlp}
=\operatorname{RoutedMLP}(A_\ell).
\]

The object \(A_\ell\) is moved, not copied. Its parameter tensors and buffers
therefore retain their values and identities.

## Algorithms

1. Confirm `model.config.model_type == "qwen3"`.
2. Discover decoder layers in stable depth order.
3. Resolve the configured layer selector.
4. reject pre-wrapped selected layers;
5. freeze the complete incumbent model;
6. move each selected MLP under a `RoutedMLP`;
7. return the exact converted indices.

## Assumptions and invariants

- Decoder layers expose an `mlp` module.
- The complete incumbent model is frozen during conversion.
- Conversion never clones or mutates anchor tensor values.
- Stale MLP references and double wrapping are rejected.
- Explicit layer indices are sorted and deterministic.

## Hugging Face integration

The backend consumes ordinary `transformers` Qwen3 model objects and never
patches their classes. `AnchoredMoEForCausalLM` is a separate
`PreTrainedModel` facade that owns the native anchor, resolves this adapter,
and delegates forward and generation back to the model-family implementation.
This preserves backend/model separation while making conversion available
through a Hugging Face–compatible public entry point.

## Configuration and examples

```python
from anchored_moe import AnchoredMoEConfig
from anchored_moe.backends import Qwen3Backend

config = AnchoredMoEConfig(converted_layers="upper_half")
report = Qwen3Backend().convert(model, config)
```

See `examples/verify_dense_equivalence.py` for a non-interactive local example.
`examples/verify_package_runtime.py` exercises backend resolution through
`AnchoredMoEForCausalLM`.
The guided workflows in `notebooks/00_inspect_base_model.ipynb` and
`notebooks/01_convert_dense_model.ipynb` inspect the model-family contract and
perform conversion through the same public backend API.

## References

- [Qwen3 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
  defines the supported Transformers model family and public configuration.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines the immutable-anchor and layer-conversion requirements.

No paper algorithm is implemented in this foundation backend.

## Tests and known limitations

Tests cover discovery, selected-layer replacement, stale/double conversion
protection, full-model freezing, and exact tiny-model output equivalence.
CORE-005 additionally tests local `from_pretrained` loading, native generation
equivalence, base-configuration hash rejection before conversion, and complete
parameter freezing. Optimized kernels, remote production checkpoints, and
non-Qwen3 families are not yet validated.
