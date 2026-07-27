# Training

## Scope

The training subpackage owns expert-training execution topology, mechanical
parameter freezing, trainable-parameter assertions, and Hugging Face Trainer
integration.

TRN-001 implements forced-path single-expert training: one independently owned
expert is attached at every converted layer, and every token executes that
expert without consulting a router. It does not own package export, router
fitting, replay preservation, mixture-aware training, or distributed expert
dispatch.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| TRN-001 | `ForcedExpertBinding`, `prepare_forced_path_training`, `ForcedPathExpertTrainer` | `expert_trainer.py` | Validated |

`ForcedPathTrainingReport` records the resolved layer topology and exact
trainable parameter names and counts. `ForcedPathTrainingError` reports unsafe
route or parameter state.

## Mathematical formulation

Let \(C\) be the converted layer set, \(A_\ell\) the immutable anchor MLP, and
\(E_{\ell,k}\) the new expert with stable package identifier \(k\). Forced-path
training replaces only the MLP execution at each converted layer:

\[
y_{\ell,t}=E_{\ell,k}(h_{\ell,t}), \qquad \ell\in C.
\]

There is no router score, anchor comparison, or sparse token grouping. Every
token \(t\) follows the same expert path at each converted layer.

For causal language modeling with token targets \(x_{t+1}\), TRN-001 uses the
ordinary model loss:

\[
\mathcal L_{\mathrm{CLM}}
=-\sum_t \log p_\theta(x_{t+1}\mid x_{\le t}).
\]

Only expert parameters participate in optimization:

\[
\mathcal P_{\mathrm{train}}
=\bigcup_{\ell\in C}\mathcal P(E_{\ell,k}),
\]

\[
\mathcal P_{\mathrm{train}}
\cap
\left(
\mathcal P(\text{backbone})
\cup
\bigcup_{\ell\in C}\mathcal P(A_\ell)
\right)
=\varnothing.
\]

The implementation compares parameter object identities, not names alone, so
shared storage or accidental unfreezing cannot silently expand the optimizer
set.

## Algorithms

Preparation:

1. Discover converted `RoutedMLP` wrappers through the selected backend.
2. Normalize bindings into deterministic layer order.
3. Require exactly one binding for every converted layer.
4. Require the same expert identifier at every layer.
5. Require every wrapper to remain in exact `forced_anchor` mode.
6. Reject parameterless experts, expert/model parameter sharing, and
   cross-layer expert parameter sharing.
7. Freeze every model parameter mechanically with `requires_grad_(False)`.
8. Re-enable gradients only for the supplied expert objects.
9. Attach each expert in `forced_expert` mode without constructing a router.
10. Compare the resulting trainable parameter identity set with the exact
    expert parameter identity set.
11. Return the resolved topology and parameter-count report.

Trainer execution:

1. Require `TrainingArguments(save_strategy="no")` so the immutable anchor is
   not silently duplicated in Trainer checkpoints.
2. Prepare the forced path and register expert modules on the model.
3. Initialize the standard Hugging Face `Trainer`, allowing Accelerate to move
   the anchor and newly registered experts under one device policy.
4. Before every `train()` invocation, revalidate wrapper mode, layer ownership,
   expert object identity, and every trainable parameter flag.
5. Delegate loss computation, gradient accumulation, clipping, optimizer
   stepping, scheduling, callbacks, and device placement to `Trainer` and
   Accelerate.

## Assumptions and invariants

- The model is converted before training.
- Every converted wrapper begins in exact `forced_anchor` mode.
- One independent expert object exists for every converted layer.
- A single stable expert ID is used across the package's layers.
- Experts at different layers share neither module objects nor parameters.
- Anchor, attention, embeddings, normalization, language-model head, routers,
  and incumbent experts remain frozen.
- Every token executes the new expert at every converted layer.
- Trainer construction and every `train()` call assert the exact trainable
  parameter identity set.
- `model_init` is rejected because bindings own concrete expert objects.
- Full-model Trainer checkpoint saving is rejected. Expert safetensors and the
  complete package manifest are exported explicitly in later package workflow
  features.
- Checkpoint resume is rejected until expert-only optimizer and scheduler state
  has a versioned serialization contract.
- TRN-001 uses the model's ordinary loss and adds no preservation, drift,
  diversity, or balancing term.

## Hugging Face integration

`ForcedPathExpertTrainer` is a small `transformers.Trainer` subclass. It accepts
ordinary datasets, data collators, metrics, callbacks, optimizers, and
`TrainingArguments`. Accelerate remains responsible for CPU/GPU placement,
mixed precision, gradient accumulation, and distributed execution.

The subclass deliberately constrains standard Trainer capabilities:

- `model_init` is incompatible with concrete expert bindings;
- checkpoint `save_strategy` must be `"no"`;
- `push_to_hub` is rejected because TRN-001 does not publish full models;
- `save_model()` and checkpoint resume are rejected to prevent full-anchor
  serialization or restoration.

After training, callers serialize each `FullMLPExpert` with
`save_safetensors()`. NB-003 will assemble those payloads, explicit PKG-007
bindings, router metadata, and provenance into a complete expert package.

## Configuration and examples

```python
bindings = [
    ForcedExpertBinding(18, "code:0", layer_18_expert),
    ForcedExpertBinding(19, "code:0", layer_19_expert),
]

trainer = ForcedPathExpertTrainer(
    model=model,
    backend=Qwen3Backend(),
    expert_bindings=bindings,
    args=TrainingArguments(
        output_dir="./expert-run",
        save_strategy="no",
        gradient_accumulation_steps=4,
    ),
    train_dataset=dataset,
)
trainer.train()
print(trainer.forced_path_report.trainable_parameter_count)
```

See `examples/verify_forced_path_training.py` for a deterministic offline
training run using a randomly initialized tiny Qwen3 model.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines forced-path expert training, immutable anchors, acceptance gates,
  and the separation between expert and router training.
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
  defines the training-loop extension point.
- [Hugging Face Accelerate documentation](https://huggingface.co/docs/accelerate/)
  defines device placement, accumulation, and distributed execution behavior.
- [PyTorch `requires_grad` documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html)
  defines parameter gradient eligibility.

TRN-001 is framework infrastructure and does not derive a new optimization
algorithm from a paper.

## Tests and known limitations

Acceptance tests cover exact converted-layer coverage, deterministic binding
order, one expert ID across layers, forced expert execution, anchor and
backbone freezing, exact trainable parameter identities, shared-parameter
rejection, invalid-state failure before attachment, Trainer invariant
rechecking, real causal-language-model optimization, expert weight updates,
and byte-exact anchor preservation.

Replay/KL preservation and weight-drift objectives remain TRN-005. Package
export remains NB-003. Multi-expert, local mini-MoE, router, mixture-aware,
checkpoint-resume, mixed-precision, multi-device, and distributed acceptance
tests remain future work.
