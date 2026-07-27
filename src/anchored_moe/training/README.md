# Training

## Scope

The training subpackage owns expert-training execution topology, mechanical
parameter freezing, trainable-parameter assertions, optional preservation and
drift objectives, and Hugging Face Trainer integration.

TRN-001 implements forced-path single-expert training: one independently owned
expert is attached at every converted layer, and every token executes that
expert without consulting a router. TRN-005 adds independently weighted replay
KL against precomputed immutable-anchor logits and parameter drift from
expert-initialization snapshots. Training does not own package export, router
fitting, replay-logit generation, mixture-aware training, or distributed
expert dispatch.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| TRN-001 | `ForcedExpertBinding`, `prepare_forced_path_training`, `ForcedPathExpertTrainer` | `expert_trainer.py` | Validated |
| TRN-005 | `ExpertObjectiveConfig`, `ReplayKLPreservation`, `ExpertWeightDrift`, `ExpertObjectiveReport` | `objectives.py`, `expert_trainer.py` | Validated |

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

For forced-path logits \(z_{\ell,t}\), precomputed immutable-anchor logits
\(z^A_{\ell,t}\), replay temperature \(T\), and non-negative token weights
\(m_t\), preservation is

\[
\mathcal L_{\mathrm{preserve}}
=
T^2
\frac{
\sum_t m_t
\operatorname{KL}
\left(
p^A_T(\cdot\mid t)
\;\|\;
p_T(\cdot\mid t)
\right)
}{
\sum_t m_t
}.
\]

The teacher distribution is detached unconditionally. A zero-sum mask returns
a differentiable zero, allowing specialization-only batches in a mixed
training stream. When an explicit replay mask is absent, the trainer uses the
ordinary attention mask and then all token positions.

For live expert parameters \(\theta_i\), their immutable trainer-construction
snapshots \(\theta_{0,i}\), and total expert scalar count
\(N=\sum_i|\theta_i|\), normalized weight drift is

\[
\mathcal L_{\mathrm{drift}}
=
\frac{1}{N}
\sum_i
\left\|
\theta_i-\theta_{0,i}
\right\|_2^2.
\]

The complete TRN-005 loss is

\[
\mathcal L_{\mathrm{expert}}
=
\mathcal L_{\mathrm{CLM}}
+
\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{preserve}}
+
\lambda_{\mathrm{drift}}\mathcal L_{\mathrm{drift}}.
\]

Both coefficients are non-negative and independent. Setting both to zero
recovers the exact TRN-001 loss path.

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

TRN-005 objective execution:

1. Validate finite non-negative objective weights and a positive replay
   temperature.
2. If drift is enabled, snapshot every bound expert parameter before Trainer
   or Accelerate can perform an optimizer step.
3. Keep configured teacher-logit and replay-mask columns during ordinary
   Trainer dataset-column pruning.
4. Remove auxiliary fields before invoking the causal LM.
5. Compute the standard specialization loss through the parent Trainer.
6. Compute replay KL in float32, or float64 for float64 student logits, using
   detached teacher logits and the resolved token mask.
7. Compute squared drift from detached CPU snapshots while preserving the
   gradient path only to live expert parameters.
8. Apply the two independent coefficients and expose detached component
   diagnostics through `last_objective_report`.

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
  complete expert-only package manifest are exported explicitly through the
  NB-003 registry export workflow.
- Checkpoint resume is rejected until expert-only optimizer and scheduler state
  has a versioned serialization contract.
- Objective snapshots are ordinary detached tensors outside model state and
  can never enter the optimizer or package serialization.
- Drift snapshots require one detached CPU copy of every trained expert
  parameter; this storage cost is explicit and can be substantial for full
  MLP experts.
- Replay teacher logits must match student shape, token ordering, vocabulary,
  checkpoint, and device; malformed or non-finite tensors fail explicitly.
- Replay masks are non-negative and match every logits dimension except
  vocabulary. Boolean and fractional masks are supported.
- Precomputed teacher logits receive no gradient.
- Drift is normalized by total expert scalar count and currently requires all
  bound expert parameters on one device.
- TRN-001's default uses the model's ordinary loss. TRN-005 is opt-in and adds
  no diversity or balancing term.

## Hugging Face integration

`ForcedPathExpertTrainer` is a small `transformers.Trainer` subclass. It accepts
ordinary datasets, data collators, metrics, callbacks, optimizers, and
`TrainingArguments`. Accelerate remains responsible for CPU/GPU placement,
mixed precision, gradient accumulation, and distributed execution.

When replay KL is enabled, the trainer extends Hugging Face's resolved dataset
signature columns with only the configured teacher and mask keys. Standard
unused-column removal therefore remains active. Teacher logits are batch data,
not a registered teacher model, so anchor parameters and optimizer ownership
remain unchanged.

The subclass deliberately constrains standard Trainer capabilities:

- `model_init` is incompatible with concrete expert bindings;
- checkpoint `save_strategy` must be `"no"`;
- `push_to_hub` is rejected because TRN-001 does not publish full models;
- `save_model()` and checkpoint resume are rejected to prevent full-anchor
  serialization or restoration.

After training, callers pass the same independently owned expert objects to
`export_full_mlp_expert_package()`. NB-003 assembles safe payloads, explicit
PKG-007 bindings, checkpoint/runtime/dataset provenance, and an expert-only
manifest. It deliberately emits no router metadata beyond the frozen router
feature specification; NB-004 owns router fitting and calibrated heads.

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
    objective_config=ExpertObjectiveConfig(
        replay_kl_weight=0.5,
        drift_weight=0.1,
        replay_temperature=2.0,
        teacher_logits_key="anchor_logits",
        replay_token_mask_key="replay_token_mask",
    ),
    args=TrainingArguments(
        output_dir="./expert-run",
        save_strategy="no",
        gradient_accumulation_steps=4,
    ),
    train_dataset=dataset,
)
trainer.train()
print(trainer.forced_path_report.trainable_parameter_count)
print(trainer.last_objective_report)
```

See `examples/verify_forced_path_training.py` for a deterministic offline
training run using a randomly initialized tiny Qwen3 model, precomputed anchor
logits, token masks, and both TRN-005 terms. See
`examples/train_expert_package.py` and
`notebooks/02_train_expert_package.ipynb` for the NB-003 export boundary.

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
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
  motivates temperature-scaled KL preservation. This implementation uses
  precomputed immutable-anchor logits, token-level masks, and a separate causal
  language-model loss.
- [Explicit Inductive Bias for Transfer Learning with Convolutional Networks](https://arxiv.org/abs/1802.01483)
  motivates start-point parameter regularization. This implementation applies
  it only to expert parameters and normalizes by scalar parameter count.

TRN-001 is framework infrastructure. TRN-005 adapts the referenced
distillation and L2-SP objectives with the differences stated above.

## Tests and known limitations

Acceptance tests cover exact converted-layer coverage, deterministic binding
order, one expert ID across layers, forced expert execution, anchor and
backbone freezing, exact trainable parameter identities, shared-parameter
rejection, invalid-state failure before attachment, Trainer invariant
rechecking, real causal-language-model optimization, expert weight updates,
and byte-exact anchor preservation.

TRN-005 tests cover analytical masked-KL agreement, teacher detachment,
zero-replay batches, invalid tensors and masks, exact drift normalization,
snapshot immutability, gradient flow, weighted loss composition, Trainer
column retention, real two-step optimization, exact optimizer ownership, and
byte-exact incumbent preservation.

NB-003 records training-dataset and base-checkpoint provenance but does not
serialize per-example teacher logits or generate them online. Precomputed
full-vocabulary logits can also be storage-intensive; compact teacher targets
are not implemented. Online teacher execution, dataset streaming,
multi-expert objectives, local mini-MoE, router fitting, mixture-aware
training, expert-only optimizer checkpoint resume, mixed-precision,
multi-device, and distributed acceptance tests remain future work.
