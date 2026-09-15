# Model-level adapters

## Scope

This subpackage adds a separate model-level specialization path alongside the
existing `RoutedMLP` expert architecture. ADP-001 resolves configurable decoder
projection targets. ADP-002 trains one adapter without a router. ADP-003 chooses
one adapter or the base for a complete generated response. ADP-004 exports and
reloads adapter-only deployment artifacts and supports local training resume.

It does not generate datasets, train a text classifier, replace decoder blocks,
implement token-level MoE, or extend `ExpertRegistry`'s MLP payload format.

## Implemented strategies

| Feature ID | Configuration / API | Module | Status |
|---|---|---|---|
| ADP-001 | `AdapterTargetConfig`, `AdapterTargetBackend` | `targets.py`, `model.py` | Validated |
| ADP-002 | `AdapterSpecializationTrainer`, `ExpertObjectiveConfig` | `training.py` | Validated |
| ADP-003 | `PromptRoutedAdapterGenerator`, `PromptRoutingConfig` | `routing.py` | Validated |
| ADP-004 | `AdapterSpecializationModel.save_adapter`, `from_adapter` | `model.py`, `training.py` | Validated |

## Mathematical formulation

For each selected linear projection with frozen weight
`W` of shape `(d_out, d_in)`, adapter `k` introduces trainable matrices
`A_k: (r, d_in)` and `B_k: (d_out, r)`. Its output is
`W x + (alpha / r) B_k A_k dropout(x)`. PEFT initializes `B_k` to zero,
so evaluation initially equals the base. Attention and MLP adaptations execute
inside the native decoder; normalization, residual connections, positional
information, attention masks, and within-response caches retain native semantics.

The optimization objective is causal-LM loss plus optional precomputed replay
KL and parameter-count-normalized initialization drift, as defined in
[`training/README.md`](../training/README.md). Labels and masks come from the
caller; use `-100` for padding and any nonsupervised prompt tokens. No automatic
assistant-only label construction or length reweighting is implied.

The prompt router maps a pre-response history to `(adapter_id, confidence)`.
`None` means the immutable base. A configured threshold sends low-confidence
named routes to base. One choice applies to every token in that response and
every adapted projection; this is not per-layer or per-token expert selection.

## Algorithms

1. Resolve semantic components and layer selectors into exact projection paths.
2. Validate all selected modules and override paths before adapter injection.
3. Record base/configuration hashes and caller-supplied model/tokenizer pins.
4. Attach LoRA through PEFT, preserving the original parameter identity set.
5. During training, activate one named adapter and assert exact optimizer
   ownership before optimization and on each loss computation. Other adapters
   and the complete base remain frozen. Freeze everything on success or failure.
6. Export a staged deployment directory containing `manifest.json` and
   `adapter.safetensors`; validate the same identity and target schema on reload.
7. For each response, call the external router once, apply explicit fallback,
   generate with a fresh dynamic cache, and restore previous state in `finally`.

## Assumptions and invariants

- `AdapterTargetConfig` supports `last_n`, `all`, `upper_half`, or an immutable
  tuple of explicit indices. For selectors other than `last_n`, set `last_n=None`.
  Out-of-range indices and an oversized last-N selection fail, never clip.
- Select `attention`, `mlp`, or both. Alternatively set `components=()` and
  provide exact layer-relative `module_paths`. Targets must be native linear
  projections, or existing owner-managed LoRA wrappers of those projections.
- Rank/alpha overrides use exact full resolved module paths, not user regexes.
  Target selection itself uses escaped full-path matching to avoid suffix leaks.
- The default backend describes Qwen3 causal LMs. `AdapterTargetBackend` can
  describe another compatible structural layout; other families are unvalidated.
- Attachment modifies the caller's materialized model. Existing PEFT models,
  meta weights, and routed MLP wrappers are not adopted. Allocation/injection
  errors require a fresh model; full injection rollback is not promised.
- Independent adapters share frozen base storage rather than cloning blocks.
  Only attention/MLP projection adapters train, not every parameter of a block.
- Owner operations reject overlap/nesting. Direct mutation of `.model`, its
  configuration dictionaries, or parameters bypasses these guards and is not a
  supported concurrent control interface.
- Generation accepts pre-tokenized inputs plus separately supplied router
  history. The caller must ensure they describe the same prompt. History limits
  affect classification only. No target assistant answer belongs in router input.
- No cross-response KV-cache reuse is supported. Externally supplied caches
  and persistent/static cache configurations fail rather than mix base/adapter
  states. Each call represents one complete response; call again for next turn.
- Unknown adapter IDs fail even if confidence is low. Positive thresholds require
  confidence on named routes. Base routes need no confidence.

## Hugging Face integration

`AdapterSpecializationModel.model` is a PEFT causal LM over the native model;
it is not `AnchoredMoEForCausalLM` and requires no MLP conversion. Trainer handles
ordinary loss, accumulation, scheduling, callbacks, and evaluation. The owner
holds the selected route across gradient-checkpoint recomputation, although
accelerator and checkpointing configurations still need workload validation.

`AdapterSpecializationTrainer` accepts datasets and collators through ordinary
Trainer keywords. `training_provenance` records JSON-compatible dataset/revision,
split, seed, or preprocessing identifiers supplied by the caller. It does not
verify that those declarations match the dataset contents. Replay logits must
already be aligned with the student batch, vocabulary, device, and token masks.

Trainer checkpoint folders include a `specialist/` deployment package,
`adapter_training.json`, optional `drift.safetensors`, and native Trainer state.
Resume validates package identity, target configuration, objectives, and training
provenance. It restores the original drift reference, not the resumed weights as
a new reference. Native optimizer/scheduler/RNG state uses the installed Trainer's
serialization, including potentially pickle-based local state: only resume
trusted local checkpoints. A deployment package contains no such state.

`save_adapter()` refuses overwrite and publishes through a sibling staging
directory. Adapter manifests use format version 1 and bind model/tokenizer pins,
base/config hashes, target configuration, resolved paths, training provenance,
and a SHA-256 tensor-file digest. Reload checks keys, shapes, floating/finite
values, and destination dtype; hashes provide integrity, not authenticity.

## Configuration and examples

```python
from anchored_moe import (
    AdapterBaseIdentity,
    AdapterSpecializationModel,
    AdapterTargetConfig,
)

# base is a materialized, unconverted Qwen3 causal LM loaded by the caller.
specialization = AdapterSpecializationModel.from_model(
    base,
    adapter_id="pentagon",
    identity=AdapterBaseIdentity(
        "Qwen/Qwen3-4B",
        "<pinned-model-commit>",
        "Qwen/Qwen3-4B",
        "<pinned-tokenizer-commit>",
    ),
    config=AdapterTargetConfig(
        last_n=8,
        components=("attention", "mlp"),
        rank=8,
        alpha=16,
    ),
)
```

For explicit targets, use for example `layers=(1, 3), last_n=None,
components=(), module_paths=("self_attn.q_proj", "mlp.down_proj")`.
`reports[adapter_id]` exposes the resolved layer indices and module names.
`adapter_parameters(adapter_id)` exposes the exact trainable candidate tensors.

Run `python examples/verify_prompt_adapter.py` for an offline end-to-end example.
It creates a tiny random Qwen3, trains attention and MLP adapters, saves local
checkpoints, exports/reloads a deployment package, and generates through a fixed
router. `FixedAdapterRouter` is an oracle/debug option, not a trained classifier.
External classifiers implement `PromptAdapterRouter.select(messages)`.

## References

- [LoRA](https://arxiv.org/abs/2106.09685) motivates the low-rank projection update.
  Difference: multiple independent adapters share a frozen decoder and are
  selected externally for whole responses, not jointly learned token routing.
- [PEFT model API](https://huggingface.co/docs/peft/en/package_reference/peft_model)
  supplies attachment, selection, disabled-adapter contexts, and tensor state.
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
  supplies optimization and trusted-local checkpoint state. Model serialization
  is replaced by the adapter-only versioned artifact, not a full-base save.
- [Qwen3 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
  describes the native decoder whose forward/generation behavior is retained.

Target resolution, artifact validation, route locking, and fallback are framework
infrastructure, not claims to reproduce a separate paper algorithm.

## Tests and known limitations

`test_adapter_configuration.py` and `test_prompt_adapter_training.py` cover
semantic/explicit targeting, rank overrides, invalid configuration, zero-update
equivalence, attention/MLP gradients, optimizer isolation, immutable original
weights, base fallback, exact reload, deterministic checkpoint resume including
drift, generation cleanup, history limits, cache rejection, and corrupt artifacts.

Tests use tiny random CPU Qwen3 models. GPU performance and production readiness
must not be inferred from the tracker
status. Validation used PyTorch 2.14.0+cpu, Transformers 5.17.0, and PEFT 0.20.0.
The full regression run has two unrelated Windows Hub symlink-privilege failures;
all new adapter acceptance tests pass. Non-reentrant CPU gradient checkpointing,
best-checkpoint restoration, failure cleanup after a step, and the last-eight
selector on a deeper tiny model are also exercised. Qwen3-4B quality, datasets,
GPU memory/throughput, mixed precision, quantized weights, distributed/sharded
training, streaming, shared-owner concurrent serving, automatic classifier
training, and existing ExpertRegistry integration are not validated or supplied.
Changing adapter availability does not automatically retrain or calibrate a router.
