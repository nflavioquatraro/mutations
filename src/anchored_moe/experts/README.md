# Experts

## Scope

The experts subpackage owns independently trainable transformations that can
replace an anchor MLP when a routing policy selects them. EXP-001 implements a
complete copied MLP for one converted layer. EXP-002 implements a complete
PEFT/LoRA MLP over a private frozen anchor copy, reducing trainable and
serialized state while preserving the existing replacement-expert interface.

This subpackage does not decide routes, attach experts to `RoutedMLP`, install
packages, manage capacity, or provide training loops. TRN-001 consumes the
public `Expert` interface from the separate training subpackage so expert
parameterization remains independently configurable.

## Implemented strategies

| Feature ID | Public configuration | Module | Status |
|---|---|---|---|
| EXP-001 | `expert_parameterization: full_mlp` | `base.py`, `full_mlp.py` | Validated |
| EXP-002 | `expert_parameterization: peft_lora`, `LoRAExpertConfig` | `base.py`, `low_rank.py` | Validated |

`Expert` defines the model-family-neutral execution, parameter-accounting, and
serialization contract. Its `set_trainable()` extension point lets each
parameterization declare the exact subset owned by an optimizer.
`FullMLPExpert` owns one complete trainable copy. `PEFTLoRAExpert` owns a
private frozen copy with PEFT-injected adapter tensors.

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

For an EXP-002 targeted projection with frozen private base weight \(W_\ell\),
rank \(r\), factors
\(A_{\ell,k}\in\mathbb{R}^{r\times d_\text{in}}\) and
\(B_{\ell,k}\in\mathbb{R}^{d_\text{out}\times r}\), and scaling \(s\):

\[
W_{\ell,k}h=W_\ell h+sB_{\ell,k}A_{\ell,k}h.
\]

Ordinary LoRA uses \(s=\alpha/r\); the optional rank-stabilized variant uses
\(s=\alpha/\sqrt{r}\). PEFT's zero-update initialization makes
\(B_{\ell,k}=0\), so in the same mode and dtype:

\[
E_{\ell,k}(h)=A_\ell(h)
\quad\text{at initialization}.
\]

The expert still returns this complete MLP output, not
\(\Delta_{\ell,k}(h)\) alone. Only adapter factors belong to training and
serialization:

\[
\mathcal P_\text{optimizer}(E_{\ell,k})
=\{A_{\ell,k},B_{\ell,k}\},
\qquad
\nabla_{W_\ell}\mathcal L=0.
\]

The supplied live anchor and the reconstructed private base have equal initial
values but disjoint storage. The private base is omitted from the payload.

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

PEFT/LoRA construction and serialization:

1. Validate rank, alpha, dropout, rank-stabilized scaling, and a stable unique
   target-module tuple.
2. Deep-copy the complete anchor MLP before any caller-visible mutation.
3. Freeze the copy and inject one ordinary LoRA adapter through PEFT's
   low-level module API.
4. Freeze the live anchor only after copying and injection succeed.
5. Map PEFT's adapter-name-neutral checkpoint keys to exact registered adapter
   parameters and enable only that subset.
6. Save only PEFT's adapter state plus canonical configuration, source-class,
   format, and PEFT-version metadata.
7. During loading, inject into a private validation copy, require exact keys,
   shapes, and dtypes, restore through PEFT, and verify every tensor.
8. Freeze the compatible live anchor only after complete load verification.

## Assumptions and invariants

- The runtime can deep-copy the anchor MLP.
- The copied module preserves the anchor's device, dtype, buffers, and mode.
- Anchor and expert parameters and buffers have independent storage.
- Constructing or successfully loading an expert freezes the supplied anchor.
- Expert trainability is explicit. Full experts toggle every parameter;
  `PEFTLoRAExpert` always freezes its private base and toggles only adapters.
- Expert execution performs no anchor addition, mixing, routing, or reshaping.
- Serialization contains expert state only; it contains no anchor checkpoint
  and no executable Python payload.
- Tensor loading is strict across state keys, shapes, dtypes, and module class.
- Existing payloads are not overwritten unless `overwrite=True`.
- EXP-002 supports ordinary LoRA with `bias="none"`, no `modules_to_save`, and
  PEFT's zero-update initialization. DoRA, non-zero initialization schemes,
  task wrappers, and trainable biases are outside the version-1 contract.
- Target modules are configuration rather than Qwen3 constants. Missing
  targets fail before the supplied anchor is frozen.
- The saved PEFT version is reproducibility metadata. A different runtime
  version is accepted only when the strict adapter schema remains compatible.

## Hugging Face integration

`FullMLPExpert` accepts an ordinary Transformers MLP module, including a Qwen3
decoder layer's `mlp`. It remains model-family-neutral because it preserves the
supplied module and its forward signature instead of reconstructing a Qwen3
class from configuration.

`PEFTLoRAExpert` uses `peft.LoraConfig`,
`inject_adapter_in_model()`, `get_peft_model_state_dict()`, and
`set_peft_model_state_dict()`. PEFT mutates only the private copy. Qwen3 callers
normally target `gate_proj`, `up_proj`, and `down_proj`; other backends can
supply different module suffixes through `LoRAExpertConfig`.

The `.safetensors` payload is suitable for inclusion in the
`expert_weight_files` declared by an `ExpertPackageManifest`. Package
construction, base-checkpoint validation, and multi-layer tensor layout are
separate registry concerns. PKG-007 materialization and NB-003 export currently
accept only `full_mlp`; PEFT/LoRA package binding remains an explicit
follow-up. NB-003's
`export_full_mlp_expert_package()` consumes trained `FullMLPExpert` objects,
derives explicit layer/expert payload bindings, and publishes a strictly
validated expert-only package without serializing the anchor.

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

```python
from anchored_moe import LoRAExpertConfig, PEFTLoRAExpert

lora_expert = PEFTLoRAExpert.from_anchor(
    model.model.layers[2].mlp,
    config=LoRAExpertConfig(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    ),
)
lora_expert.save_safetensors("code-lora.safetensors")
```

See `examples/verify_full_mlp_expert.py` for an offline Qwen3 example covering
copy equality, gradient isolation, and tensor serialization. See
`examples/verify_lora_expert.py` for zero-update equivalence, adapter-only
optimization, immutable-anchor verification, and exact adapter reload. See
`examples/train_expert_package.py` for multi-layer training and package export.

## References

- [PyTorch `nn.Module` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  defines module state, parameter registration, gradient flags, and state-dict
  behavior used by the expert abstraction.
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
  motivates the non-executable tensor payload and metadata format.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  defines the frozen-weight low-rank update used by EXP-002. This
  implementation applies it to a private MLP copy so the expert returns a
  complete replacement output; shared-anchor deltas remain COM-004.
- [Hugging Face PEFT LoRA documentation](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
  defines adapter injection, scaling, and adapter-only state conventions.
- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  specifies full-copy initialization and immutable-anchor behavior.

EXP-001 is infrastructure and does not implement a paper-derived algorithm.
EXP-002 implements the cited LoRA parameterization with the documented
private-copy difference.

## Tests and known limitations

Acceptance tests cover exact initial outputs, independent parameter and buffer
storage, anchor freezing, expert-only gradients, inference-only loading,
deterministic safetensors output, round trips, overwrite protection, metadata
validation, and state-schema rejection.

EXP-002 acceptance additionally covers deterministic configuration, exact
zero-update Qwen3 output, private base storage, adapter-only gradients,
trainability toggling, failed-injection anchor safety, adapter-only
safetensors contents, deterministic bytes, strict schema validation, exact
reload, and real adapter-only forced-path Trainer optimization.

INF-001 can explicitly attach an expert object to one `RoutedMLP`, TRN-001 can
force one trainable expert at every converted layer, PKG-002 can
activate its validated package descriptor in a registry, and PKG-007 can
materialize explicitly bound full-MLP payloads across converted layers.
PEFT/LoRA package export/materialization, shared-anchor delta execution, mixed
parameterization packages, non-default LoRA initialization, and distributed
materialization remain future work.
