# Experts

## Scope

The experts subpackage owns independently trainable transformations that can
replace an anchor MLP when a future routing policy selects them. EXP-001
implements a complete copied MLP for one converted layer.

This subpackage does not decide routes, attach experts to `RoutedMLP`, install
packages, manage capacity, or provide training loops. Those responsibilities
remain with the modeling, routing, registry, capacity, and training features.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| EXP-001 | `expert_parameterization: full_mlp` | `base.py`, `full_mlp.py` | Validated |

`Expert` defines the model-family-neutral execution, parameter-accounting, and
serialization contract. `FullMLPExpert` implements that contract by owning one
complete copied MLP.

## Mathematical formulation

For converted layer \(\ell\), let the immutable dense MLP be

\[
A_\ell:\mathbb{R}^{d_\text{model}}\rightarrow
\mathbb{R}^{d_\text{model}}.
\]

A full expert \(k\) is initialized by an exact parameter-and-buffer copy:

\[
E_{\ell,k}\leftarrow A_\ell.
\]

For hidden states
\(h\in\mathbb{R}^{B\times S\times d_\text{model}}\), direct expert execution is

\[
y=E_{\ell,k}(h),
\qquad
y\in\mathbb{R}^{B\times S\times d_\text{model}}.
\]

At initialization and in the same execution mode:

\[
E_{\ell,k}(h)=A_\ell(h).
\]

Equality of initial values does not imply shared storage. For every copied
parameter or buffer \(p\), the implementation requires

\[
\operatorname{storage}(p_{E_{\ell,k}})
\ne
\operatorname{storage}(p_{A_\ell}).
\]

The anchor has no trainable parameters after construction, while the default
expert does:

\[
\nabla_{\theta_{A_\ell}}\mathcal{L}=0,
\qquad
\nabla_{\theta_{E_{\ell,k}}}\mathcal{L}\ \text{is enabled}.
\]

## Algorithms

Anchor-copy initialization:

1. Validate that the supplied anchor is a `torch.nn.Module`.
2. Deep-copy its complete module hierarchy, parameters, and buffers.
3. set `requires_grad=False` on the supplied anchor;
4. set every copied expert parameter to the requested trainability;
5. forward calls directly through the copied MLP.

Safetensors export:

1. Require an existing parent directory and a `.safetensors` destination.
2. Reject symbolic links and accidental overwrites.
3. detach, sort, make contiguous, and move every expert tensor to CPU;
4. record the format version, `full_mlp` parameterization, and module class;
5. write beside the destination and replace it atomically.

Safetensors loading:

1. Reject symbolic links, wrong suffixes, and non-files.
2. Parse safetensors metadata without executable content.
3. require the exact format, parameterization, and module class;
4. compare state keys, shapes, and dtypes with the supplied anchor;
5. only then copy and freeze the anchor;
6. load tensors strictly into the independent expert.

## Assumptions and invariants

- The runtime can deep-copy the anchor MLP.
- The copied module preserves the anchor's device, dtype, buffers, and mode.
- Anchor and expert parameters and buffers have independent storage.
- Constructing or successfully loading an expert freezes the supplied anchor.
- Expert trainability is explicit and uniform across its parameters.
- Expert execution performs no anchor addition, mixing, routing, or reshaping.
- Serialization contains expert state only; it contains no anchor checkpoint
  and no executable Python payload.
- Tensor loading is strict across state keys, shapes, dtypes, and module class.
- Existing payloads are not overwritten unless `overwrite=True`.

## Hugging Face integration

`FullMLPExpert` accepts an ordinary Transformers MLP module, including a Qwen3
decoder layer's `mlp`. It remains model-family-neutral because it preserves the
supplied module and its forward signature instead of reconstructing a Qwen3
class from configuration.

The `.safetensors` payload is suitable for inclusion in the
`expert_weight_files` declared by an `ExpertPackageManifest`. Package
construction, base-checkpoint validation, and multi-layer tensor layout remain
separate registry concerns.

## Configuration and examples

```python
from anchored_moe import FullMLPExpert

anchor_mlp = model.model.layers[2].mlp
expert = FullMLPExpert.from_anchor(anchor_mlp)
expert.save_safetensors("expert.safetensors")

reloaded = FullMLPExpert.from_safetensors(
    anchor_mlp,
    "expert.safetensors",
    trainable=False,
)
```

See `examples/verify_full_mlp_expert.py` for an offline Qwen3 example covering
copy equality, gradient isolation, and tensor serialization.

## References

- [PyTorch `nn.Module` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  defines module state, parameter registration, gradient flags, and state-dict
  behavior used by the expert abstraction.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor payload and metadata format.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  specifies full-copy initialization and immutable-anchor behavior.

EXP-001 is infrastructure and does not implement a paper-derived algorithm.

## Tests and known limitations

Acceptance tests cover exact initial outputs, independent parameter and buffer
storage, anchor freezing, expert-only gradients, inference-only loading,
deterministic safetensors output, round trips, overwrite protection, metadata
validation, and state-schema rejection.

The expert is not yet attached to `RoutedMLP`, selected by a router, trained by
a framework trainer, or composed across layers. Those behaviors belong to
TRN-001, RTR-001/RTR-002, POL-001, and PKG-002.
