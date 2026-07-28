# Training

## Scope

The training subpackage owns expert-training execution topology, mechanical
parameter freezing, trainable-parameter assertions, optional preservation and
drift objectives, and Hugging Face Trainer integration.

TRN-001 implements forced-path single-expert training: one independently owned
expert is attached at every converted layer, and every token executes that
expert without consulting a router. TRN-002 composes that isolated path into
an independently trained latent expert set: two or more complete expert paths
receive their own Trainer inputs and execute sequentially against the same
immutable anchor. TRN-005 adds independently weighted replay KL against
precomputed immutable-anchor logits and parameter drift from
expert-initialization snapshots. Training does not own data partitioning,
package export, router fitting, replay-logit generation, local mini-MoE or
mixture-aware training, or process-level parallel expert dispatch.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| TRN-001 | `ForcedExpertBinding`, `prepare_forced_path_training`, `ForcedPathExpertTrainer` | `expert_trainer.py` | Validated |
| TRN-002 | `IndependentExpertTrainingRun`, `train_independent_expert_set`, immutable run/set reports | `independent.py` | Validated |
| TRN-005 | `ExpertObjectiveConfig`, `ReplayKLPreservation`, `ExpertWeightDrift`, `ExpertObjectiveReport` | `objectives.py`, `expert_trainer.py` | Validated |

`ForcedPathTrainingReport` records the resolved layer topology and exact
trainable parameter names and counts. `ForcedPathTrainingError` reports unsafe
route or parameter state. `IndependentExpertTrainingReport` records one
expert's completed optimizer steps, finite aggregate loss, forced-path
topology, and optional final objective components.

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

For a latent expert set \(K=\{1,\ldots,m\}\), \(m\ge 2\), TRN-002 performs one
isolated optimization problem per expert:

\[
\theta_k^\star
=
\arg\min_{\theta_k}
\mathcal L_k
\left(
D_k;
\theta_k,
\theta_A
\right),
\qquad k\in K,
\]

where \(D_k\) is the caller-supplied data partition, \(\theta_A\) is the fixed
anchor and shared-backbone state, and only

\[
\mathcal P_{\mathrm{train},k}
=
\bigcup_{\ell\in C}\mathcal P(E_{\ell,k})
\]

is trainable during run \(k\). For distinct experts \(i\ne j\),

\[
\mathcal P(E_{\ell,i})\cap\mathcal P(E_{\ell',j})=\varnothing
\]

by both parameter identity and underlying storage. The implementation never
forms a joint loss \(\sum_k\mathcal L_k\), never routes one batch across
multiple experts, and never allows gradients from run \(k\) to reach expert
\(j\). Declared run order is operationally deterministic but does not change
the anchor because every non-expert parameter remains frozen.

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

Independent expert-set execution:

1. Require at least two immutable `IndependentExpertTrainingRun` descriptors.
2. Normalize each run's bindings by layer while preserving declared expert
   run order.
3. Require unique expert IDs and an identical, complete converted-layer path
   for every expert.
4. Before the first Trainer exists, reject expert/model, cross-layer, and
   cross-expert parameter or storage aliasing.
5. Validate every run's no-checkpoint/no-Hub Trainer boundary up front so a
   later malformed run cannot fail after an earlier expert was optimized.
6. Attach and train exactly one expert path through TRN-001 using that run's
   dataset, seed, collator, callbacks, schedule, and optional TRN-005
   objectives.
7. Record a finite positive-step report, clear gradients, detach the expert
   path, and restore every routed layer to exact `forced_anchor` mode.
8. Repeat in declared order, then return immutable per-expert and aggregate
   reports.
9. On any exception, perform the same route reset and complete parameter
   freezing before propagating the original failure.

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
- TRN-002 requires at least two unique expert IDs with the same complete layer
  coverage.
- TRN-002 rejects parameter and storage sharing across all experts before the
  first optimizer step.
- Each TRN-002 run owns its Trainer arguments and may own a distinct dataset,
  collator, seed, schedule, and TRN-005 objective configuration.
- TRN-002 preserves caller-declared run order and reports in that same order.
- Only one expert path is registered on the model at a time. Successful and
  failed runs both finish with empty routed expert lists, exact
  `forced_anchor` mode, cleared gradients, and all model/expert parameters
  frozen.
- Failure cleanup is not expert-weight rollback. A run that fails after an
  optimizer step may leave its external expert object partially trained.
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

`train_independent_expert_set()` constructs one
`ForcedPathExpertTrainer` per run and forwards that run's ordinary Trainer
keyword arguments. The coordinator is intentionally sequential in one Python
process. Callers that require concurrent training should launch independent
models/processes with TRN-001 and reconcile their expert-only artifacts; a
shared in-process model is not concurrently mutated.

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

Two independent expert paths use the same public Trainer inputs:

```python
runs = [
    IndependentExpertTrainingRun(
        expert_bindings=code_partition_a_bindings,
        args=TrainingArguments(
            output_dir="./code-partition-a-run",
            save_strategy="no",
            seed=41,
        ),
        trainer_kwargs={"train_dataset": code_partition_a},
    ),
    IndependentExpertTrainingRun(
        expert_bindings=code_partition_b_bindings,
        args=TrainingArguments(
            output_dir="./code-partition-b-run",
            save_strategy="no",
            seed=43,
        ),
        trainer_kwargs={"train_dataset": code_partition_b},
    ),
]
report = train_independent_expert_set(
    model,
    backend=Qwen3Backend(),
    runs=runs,
)
print(report.expert_ids, report.total_global_steps)
```

See `examples/verify_forced_path_training.py` for a deterministic offline
training run using a randomly initialized tiny Qwen3 model, precomputed anchor
logits, token masks, and both TRN-005 terms.
`examples/verify_independent_expert_set.py` trains two independent data
partitions with distinct seeds and verifies expert updates, exact anchor
restoration, route cleanup, and final freezing. See
`examples/train_expert_package.py` and
`notebooks/02_train_expert_package.ipynb` for the NB-003 export boundary.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  defines forced-path and independent latent expert-set training, immutable
  anchors, acceptance gates, and the separation between expert and router
  training.
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

TRN-001 and TRN-002 are framework infrastructure and introduce no
paper-derived training algorithm. TRN-005 adapts the referenced distillation
and L2-SP objectives with the differences stated above.

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

TRN-002 tests cover immutable run configuration, deterministic binding and
report order, unique expert IDs, exact rectangular layer coverage, global
cross-expert ownership validation, distinct datasets and seeds, real
independent optimizer steps, updates to every expert, byte-exact anchor state,
empty routed expert lists after completion, complete gradient clearing and
freezing, and forced-anchor restoration after a later data-loading failure.

NB-003 records training-dataset and base-checkpoint provenance but does not
serialize per-example teacher logits or generate them online. Precomputed
full-vocabulary logits can also be storage-intensive; compact teacher targets
are not implemented. Online teacher execution, dataset streaming,
automatic data partitioning, concurrent in-process expert training,
cross-expert diversity objectives, local mini-MoE, router fitting,
mixture-aware training, transactional expert-weight rollback, expert-only
optimizer checkpoint resume, mixed-precision, multi-device, and distributed
acceptance tests remain future work.
