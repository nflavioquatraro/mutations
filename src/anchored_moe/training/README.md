# Training

## Scope

The separate [model-level adapter path](../adapters/README.md) adds
`AdapterSpecializationTrainer` for attention and MLP LoRA with prompt-level
selection. It reuses the optional objectives here but does not convert MLPs or
change the TRN-001 checkpoint restriction. `ExpertWeightDrift.snapshot_state()`
and `load_snapshot_state()` provide validated initialization-state persistence
for its resumable checkpoints.

The training subpackage owns expert-training execution topology, mechanical
parameter freezing, trainable-parameter assertions, optional preservation and
drift objectives, and Hugging Face Trainer integration.

TRN-001 implements forced-path single-expert training: one independently owned
expert is attached at every converted layer, and every token executes that
expert without consulting a router. TRN-002 composes that isolated path into
an independently trained latent expert set: two or more complete expert paths
receive their own Trainer inputs and execute sequentially against the same
immutable anchor. TRN-003 jointly trains exactly two package-local experts and
their affine router heads through a differentiable score-softmax mixture.
TRN-004 validates a completed two-expert TRN-002 topology and refines those
caller-owned experts plus fresh local heads through the existing TRN-003
runtime, retaining an immutable report that connects both stages.
BAL-003 optionally regularizes those package-local router probabilities with
a token-masked auxiliary load-balancing loss while both experts continue to
execute.
BAL-004 provides persistent gradient-free specialist selection biases updated
from completed-batch sparse assignments. It exposes raw scores separately from
selection-only biased scores, excludes the fixed-zero anchor, and safely
serializes the bias history. It does not claim to rebalance TRN-003's actual
computation because that dense local path always executes both experts.
BAL-005 provides a stateless training/prefill research backend in which every
expert selects a fixed-capacity bucket from a shared batch. It consumes
TRN-003-compatible affine router logits but deliberately stops at an auditable
sparse assignment plan: expert execution, output reconstruction, Trainer
integration, and autoregressive decoding remain separate concerns.
TRN-005 adds independently weighted replay KL against
precomputed immutable-anchor logits and parameter drift from
expert-initialization snapshots. Training does not own data partitioning,
package export, global router fitting, replay-logit generation, inference-time
capacity balancing, or process-level parallel expert dispatch.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| TRN-001 | `ForcedExpertBinding`, `prepare_forced_path_training`, `ForcedPathExpertTrainer` | `expert_trainer.py` | Validated |
| TRN-002 | `IndependentExpertTrainingRun`, `train_independent_expert_set`, immutable run/set reports | `independent.py` | Validated |
| TRN-003 | `LocalMiniMoETrainingConfig`, `LocalMiniMoEExpertBinding`, `LocalMiniMoETrainer`, immutable topology/run reports | `mini_moe.py` | Validated |
| TRN-004 | `refine_independent_expert_set`, `MixtureAwareRefinementReport`, `MixtureAwareRefinementError` | `refinement.py` | Validated |
| BAL-003 | `load_balancing_weight`, `AuxiliaryLoadBalancingLoss`, `LocalMiniMoEObjectiveReport` | `balancing.py`, `mini_moe.py` | Validated |
| BAL-004 | `LossFreeBiasConfig`, `LossFreeBiasBalancer`, `LossFreeBiasedScores`, `LossFreeBiasUpdateReport` | `loss_free.py` | Validated |
| BAL-005 | `ExpertChoiceConfig`, `ExpertChoiceRouting`, `ExpertChoiceDecision`, `ExpertChoiceRoutingError` | `expert_choice.py` | Validated |
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

Only each expert-declared optimizer subset participates in optimization:

\[
\mathcal P_{\mathrm{train}}
=\bigcup_{\ell\in C}\mathcal P_{\mathrm{optimizer}}(E_{\ell,k}),
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

For the TRN-003 local pair \(K=\{1,2\}\), affine package-owned heads produce

\[
g_{\ell,k}(h)=w_{\ell,k}^{\top}\phi_\ell(h)+b_{\ell,k},
\]

and COM-001 supplies differentiable weights

\[
\alpha_{\ell,k}(h)
=
\frac{\exp(g_{\ell,k}(h)/T_{\mathrm{mix}})}
{\sum_{j=1}^{2}\exp(g_{\ell,j}(h)/T_{\mathrm{mix}})}.
\]

Every specialization token executes both experts:

\[
y_{\ell,t}
=
\alpha_{\ell,1}(h_{\ell,t})E_{\ell,1}(h_{\ell,t})
+
\alpha_{\ell,2}(h_{\ell,t})E_{\ell,2}(h_{\ell,t}).
\]

The anchor is not a candidate inside this forced package-local mixture. It
returns with coefficient one after cleanup because every wrapper is restored
to `forced_anchor`. The trainable set is

\[
\mathcal P_{\mathrm{train}}
=
\bigcup_{\ell\in C,k\in\{1,2\}}\mathcal P_{\mathrm{optimizer}}(E_{\ell,k})
\cup
\bigcup_{\ell\in C,k\in\{1,2\}}\mathcal P(w_{\ell,k},b_{\ell,k}).
\]

Zero affine weights receive deterministic biases
\((+\epsilon,-\epsilon)\) in stable expert order. This breaks identical
anchor-copy expert symmetry without randomness; both softmax weights remain
strictly positive, so both experts receive gradients.

TRN-004 starts from the independently optimized parameters
\(\theta_1^\star,\theta_2^\star\) produced by TRN-002 and performs a second,
joint problem over a caller-supplied mixture-refinement dataset
\(D_{\mathrm{mix}}\):

\[
(\theta_1^{\mathrm{ref}},\theta_2^{\mathrm{ref}},\psi^{\mathrm{ref}})
=
\arg\min_{\theta_1,\theta_2,\psi}
\mathcal L_{\mathrm{local}}
\left(
D_{\mathrm{mix}};
\theta_1,\theta_2,\psi,\theta_A
\right),
\]

where \(\psi=\{w_{\ell,k},b_{\ell,k}\}\) denotes fresh package-local affine
heads and \(\theta_A\) remains immutable. The local loss and execution equation
are exactly TRN-003's COM-001 mixture above. The source report proves completed
training topology and optimizer progress; stable IDs and layer indices bind
that provenance to external expert objects without retaining mutable modules
inside the report.

For BAL-003, let \(m_t\ge0\) be an optional token weight, usually the attention
mask, and let \(N=2\) package-local experts. The mean router probability and
hard preferred-assignment fraction at layer \(\ell\) are

\[
P_{\ell,i}
=
\frac{\sum_t m_t\alpha_{\ell,t,i}}{\sum_t m_t},
\qquad
f_{\ell,i}
=
\frac{\sum_t m_t
\mathbb{1}[\operatorname{argmax}_j\alpha_{\ell,t,j}=i]}{\sum_t m_t}.
\]

The per-layer auxiliary loss is

\[
\mathcal L_{\mathrm{balance},\ell}
=
N\sum_{i=1}^{N}f_{\ell,i}P_{\ell,i},
\]

and the trainer averages it over converted layers:

\[
\mathcal L
=
\mathcal L_{\mathrm{CLM}}
+
\lambda_{\mathrm{balance}}
\frac{1}{|C|}
\sum_{\ell\in C}\mathcal L_{\mathrm{balance},\ell}.
\]

Hard assignment fractions are detached, while gradients flow through mean
probabilities. A balanced distribution has loss one and complete two-expert
collapse approaches two. An all-zero token mask contributes differentiable
zero. Stable first-index tie-breaking makes the measurement deterministic.
The objective changes training pressure only: both local experts still execute
for every token.

For BAL-004, let \(s_{t,i}\) be the original calibrated specialist score and
\(b_i\) a persistent, non-trainable selection bias. Sparse selection observes

\[
\widetilde{s}_{t,i}=s_{t,i}+b_i,
\]

while activation combination continues to use \(s_{t,i}\). The anchor is not
part of the bias vector and remains the exact candidate-zero score:

\[
g_{t,0}=0.
\]

After a completed batch, let \(c_i\) be the optionally weighted number of
active occurrences of specialist \(i\) and
\(\bar c=N^{-1}\sum_i c_i\). With positive update rate \(u\), the next batch
uses

\[
e_i=\bar c-c_i,
\qquad
b_i\leftarrow b_i+u\,\operatorname{sign}(e_i).
\]

Thus an overloaded specialist receives a lower future selection score and an
underloaded specialist receives a higher one. The update is under
`torch.no_grad()` and the biases are buffers, so no auxiliary-loss gradient
enters the router or experts. Candidate-zero anchor and inactive padding
occurrences contribute to no \(c_i\).

For BAL-005, let \(L\in\mathbb R^{n\times E}\) be router logits for \(n\)
flattened eligible training or prefill tokens and \(E\) experts. Specialist
affinities are normalized across experts:

\[
S_{t,e}
=
\operatorname{softmax}_e(L_{t,:}).
\]

Given explicit expert capacity \(C\le n\), expert \(e\) selects

\[
I_e
=
\operatorname{TopC}_{t}\;S_{t,e},
\qquad |I_e|=C.
\]

The load of every expert is therefore exactly \(C\), while token arity is

\[
a_t
=
\sum_{e=1}^{E}\mathbb 1[t\in I_e],
\qquad 0\le a_t\le E.
\]

An eligible token with \(a_t=0\) receives anchor fallback. Padding is excluded
before ranking and requests neither specialist nor anchor work. BAL-005 stores
the sparse index matrix \(I\in\mathbb N^{E\times C}\), its selected
graph-connected gates \(G\in[0,1]^{E\times C}\), token arities, and fallback
mask; it does not materialize the paper's
\(E\times C\times n\) one-hot permutation tensor.

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
8. Call each expert's `set_trainable(True)` extension point and require a
   non-empty optimizer subset.
9. Attach each expert in `forced_expert` mode without constructing a router.
10. Compare the resulting trainable parameter identity set with the union of
    exact expert-declared optimizer subsets.
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

Local mini-MoE execution:

1. Require exactly two unique expert IDs at every converted layer and a
   layer-major `AppendOnlyRouterHeads` rectangle with matching ownership.
2. Require temperature-one, threshold-zero affine heads whose weights and
   biases are exactly zero.
3. Reject model/expert/router/feature parameter or storage aliases before
   mutating any route.
4. Freeze the model and feature transformation, then enable only each
   expert-declared optimizer subset and the affine heads.
5. Initialize the two head biases to \(+\epsilon\) and \(-\epsilon\) in stable
   expert order.
6. Attach both experts to every converted wrapper in `local_mini_moe` mode.
7. Execute both full-token expert paths and combine them with COM-001.
8. When `load_balancing_weight` is positive, retain the graph-connected
   mixture weights from exactly one model forward, compute BAL-003 per layer
   with the attention mask, average layers, and release the observations.
9. Reassert route objects and parameter identities before the one supported
   Trainer run.
10. On success or failure, clear gradients, freeze package artifacts, release
    any retained observations, and restore every wrapper to exact
    `forced_anchor` execution.

Loss-free selection-bias execution:

1. Declare at least two unique specialists in the exact router column order.
2. Initialize one zero, non-trainable floating bias buffer per specialist.
3. Add a detached snapshot of those biases to raw scores to create the
   selection-only score view; retain the raw score tensor separately.
4. Run an existing sparse anchored selection policy with the biased view.
5. Execute and weight selected experts using the original calibrated scores.
6. After the batch completes, encode the anchor and inactive route slots as
   zero and active specialists as candidate indices `1..K`.
7. Count optionally weighted active occurrences, compute mean load and signed
   violations, and update the buffers under `torch.no_grad()`.
8. Record immutable loads, errors, maximum violation, before/after biases, and
   the completed update step.
9. Save only the bias and update-step buffers plus strict JSON metadata in
   safetensors when the training history must be resumed.

Expert-choice assignment:

1. Require at least two unique expert IDs matching the final router-logit
   dimension and a positive explicit expert capacity.
2. Validate finite floating logits, exact mask shape/device, and at least
   `expert_capacity` eligible tokens before any routing work.
3. Apply softmax across the expert dimension without changing caller logits.
4. Flatten token dimensions, discard padding positions, and stably sort each
   expert's affinities across eligible tokens.
5. Retain exactly the first `expert_capacity` positions for every expert;
   flattened token order resolves exact affinity ties.
6. Gather graph-connected selected gates and scatter sparse occurrences into
   original-shape token assignment counts.
7. Mark exactly the eligible zero-assignment tokens for immutable-anchor
   fallback and return the immutable decision.

Mixture-aware refinement:

1. Require one completed `IndependentExpertSetTrainingReport` containing
   exactly two experts.
2. Validate the caller-owned local bindings against the source report's exact
   expert order and complete layer-by-expert rectangle before model mutation.
3. Validate all forwarded Trainer keywords and reject coordinator-owned or
   unknown arguments before route attachment.
4. Delegate parameter/storage isolation, fresh-head validation, joint
   optimization, objective reporting, and failure cleanup to TRN-003.
5. Require a completed local report and return an immutable TRN-004 report
   linking both stages.
6. Never retain datasets, models, optimizers, or expert objects in the report.

TRN-005 objective execution:

1. Validate finite non-negative objective weights and a positive replay
   temperature.
2. If drift is enabled, snapshot every trainable parameter in the
   expert-declared optimizer subsets before Trainer or Accelerate can perform
   an optimizer step.
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
- TRN-003 supports exactly two experts with identical complete layer coverage.
- TRN-003 requires fresh zero affine heads; calibrated or previously trained
  heads are rejected rather than silently overwritten.
- TRN-004 requires a completed TRN-002 report with exactly two experts.
- The source report's expert order and converted layers must exactly equal the
  TRN-003 refinement topology. Binding input order is normalized and cannot
  change router-head alignment or deterministic symmetry breaking.
- TRN-004 provenance is topological. The framework cannot prove that an
  external expert object is the historical object optimized during TRN-002;
  artifact identity and weight hashes remain package-export concerns.
- TRN-004 uses fresh local heads and jointly updates both expert-declared
  optimizer subsets plus those heads. Router-only refinement is unsupported.
- Only the affine heads and expert-declared optimizer subsets train. The router
  feature transformation remains frozen.
- Both local experts execute for every specialization token. Anchor abstention,
  capacity fallback, and inference load balancing do not run inside local
  training.
- BAL-003 is disabled exactly when `load_balancing_weight` is zero; the
  original TRN-003 path then retains no mixture-probability tensor.
- When BAL-003 is enabled, every converted layer must expose one
  graph-connected pair-probability tensor for the same model forward.
- Layer losses are averaged, so the coefficient does not depend on the number
  of converted layers. Padding is excluded through a non-negative attention
  mask when one is supplied.
- BAL-003 changes no trainable parameter identity and does not serialize
  assignments, probabilities, or objective state.
- BAL-004 owns no parameters. Its specialist biases and update counter are
  persistent buffers, and score application leaves the raw tensor unchanged.
- BAL-004 excludes candidate zero from both state and load counts. The anchor
  therefore retains its exact fixed-zero score and cannot become overloaded.
- Biases affect selection only. `LossFreeBiasedScores.raw_scores` remains the
  required source for activation-combination weights.
- Bias updates consume historical completed-batch assignments. Applying
  current-sequence aggregate loads before its routes execute would violate the
  causal boundary and is unsupported.
- Specialist order is immutable within one balancer and must match router
  score columns, route candidate indices, and restored safetensors metadata.
- The initial BAL-004 primitive performs no distributed load reduction and no
  automatic Trainer-step hook.
- BAL-005 is batch-relative: every expert sees the same eligible token set and
  selects exactly `expert_capacity` distinct positions. Exact affinity ties
  preserve flattened token order.
- BAL-005 requires the explicit capacity not to exceed the eligible-token
  count. It never silently clips capacity or invents a rounding rule for a
  capacity factor.
- One token may be selected by several experts. Eligible zero-assignment
  tokens are marked for the immutable anchor; masked padding is never selected
  and never marked as anchor work.
- BAL-005 mutates neither logits nor masks. Gradients flow through selected
  softmax gates, while indices, assignment counts, and fallback masks remain
  discrete audit metadata.
- BAL-005 owns no module state and performs no serialization, expert
  execution, Trainer hook, cross-process token aggregation, or autoregressive
  route decision.
- Opposite deterministic biases break initial symmetry without changing
  affine weights.
- `LocalMiniMoETrainer` is one-shot. Successful and failed runs detach the
  mixture and freeze its external experts and router heads.
- During TRN-002, only one expert path is registered on the model at a time.
  Successful and failed runs finish with empty routed expert lists, exact
  `forced_anchor` mode, cleared gradients, and all model/expert parameters
  frozen.
- Failure cleanup is not expert-weight rollback. A run that fails after an
  optimizer step may leave its external expert object partially trained.
- Anchor, attention, embeddings, normalization, language-model head, global or
  incumbent routers, and incumbent experts remain frozen.
- Every TRN-001/TRN-002 token executes the new expert at every converted layer;
  every TRN-003 token executes both package-local experts.
- Trainer construction and every `train()` call assert the exact trainable
  parameter identity set.
- Full copied experts declare every owned parameter. EXP-002
  `PEFTLoRAExpert` declares only LoRA tensors; its reconstructed private base
  remains frozen throughout forced-path preparation and training.
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

`LocalMiniMoETrainer` is a second small `transformers.Trainer` subclass. It
registers both experts, the shared frozen feature extractor, and the affine
head bank before Accelerate device preparation. The standard model loss,
optimizer, scheduler, accumulation, clipping, callbacks, and data pipeline are
unchanged. Its `compute_loss()` override adds BAL-003 only when configured and
records detached loss components without adding model state. Cleanup happens
in `train()`'s `finally` block, so data-loading and optimizer failures cannot
leave a routed model active.

`LossFreeBiasBalancer` is a buffer-only `torch.nn.Module` that can move with a
Hugging Face model or Accelerate-prepared component, but it is deliberately
not registered in `LocalMiniMoETrainer`. Sparse training loops call
`apply_for_selection()` before routing and `update()` only after the completed
batch. Its standalone safetensors file permits explicit checkpoint ownership
without serializing the anchor, experts, optimizer, or datasets.

`ExpertChoiceRouting` consumes ordinary batched PyTorch tensors. A Hugging Face
data collator's boolean attention mask can be passed as `token_mask`; false
positions are excluded before expert ranking. The backend accepts logits from
`AppendOnlyRouterHeads`, including the affine heads used by TRN-003, but is not
registered by `LocalMiniMoETrainer` and does not replace its differentiable
dense two-expert execution. Accelerate/distributed callers must establish a
shared token group themselves; this version performs no collective operation.

`refine_independent_expert_set()` validates the completed TRN-002 report and
ordinary Trainer keyword mapping before constructing `LocalMiniMoETrainer`.
It introduces no new model route, optimizer, checkpoint, Accelerate, or
dataset behavior; TRN-003 remains the single joint-training boundary.

When replay KL is enabled, the trainer extends Hugging Face's resolved dataset
signature columns with only the configured teacher and mask keys. Standard
unused-column removal therefore remains active. Teacher logits are batch data,
not a registered teacher model, so anchor parameters and optimizer ownership
remain unchanged.

The Trainer does not special-case PEFT parameter names. It calls the public
`Expert.set_trainable()` contract and derives the resulting identities, so
adapter ownership remains in the experts subpackage while TRN-001 retains
mechanical optimizer validation.

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

Two local experts and their fresh affine heads train jointly:

```python
trainer = LocalMiniMoETrainer(
    model=model,
    backend=Qwen3Backend(),
    expert_bindings=local_pair_bindings,
    router_feature_extractor=IdentityRouterFeatureExtractor(hidden_size),
    router_heads=fresh_two_expert_heads,
    config=LocalMiniMoETrainingConfig(
        load_balancing_weight=0.01,
        mix_temperature=1.0,
        router_bias_epsilon=1e-3,
    ),
    args=TrainingArguments(
        output_dir="./local-mini-moe-run",
        save_strategy="no",
    ),
    train_dataset=dataset,
)
trainer.train()
print(trainer.training_report.final_objectives)
```

An independently trained pair can then be refined jointly without rebuilding
its training engine:

```python
refinement = refine_independent_expert_set(
    model,
    backend=Qwen3Backend(),
    source_training=independent_report,
    expert_bindings=local_bindings_for_the_same_two_experts,
    router_feature_extractor=IdentityRouterFeatureExtractor(hidden_size),
    router_heads=fresh_two_expert_heads,
    config=LocalMiniMoETrainingConfig(load_balancing_weight=0.01),
    args=TrainingArguments(
        output_dir="./mixture-refinement-run",
        save_strategy="no",
    ),
    trainer_kwargs={"train_dataset": mixture_refinement_dataset},
)
print(refinement.expert_ids, refinement.total_global_steps)
```

Loss-free biases compose with an existing sparse anchored selection policy:

```python
balancer = LossFreeBiasBalancer(
    ("code:0", "code:1"),
    config=LossFreeBiasConfig(update_rate=1e-3),
)
score_views = balancer.apply_for_selection(raw_router_scores)
routes = AnchoredTop1Selection()(
    score_views.selection_scores,
    expert_ids=score_views.expert_ids,
)
# Execute the completed batch. Any output weights use score_views.raw_scores.
report = balancer.update(routes.selected_indices)
balancer.save_safetensors("./loss-free-bias.safetensors")
```

Expert-choice routing plans fixed training/prefill buckets without executing
the experts:

```python
routing = ExpertChoiceRouting(
    ExpertChoiceConfig(expert_capacity=32)
)
decision = routing(
    local_router_logits,
    expert_ids=("code:0", "code:1"),
    token_mask=attention_mask.bool(),
)
print(decision.selected_token_indices)
print(decision.token_assignment_counts)
print(decision.anchor_fallback_mask)
```

See `examples/verify_forced_path_training.py` for a deterministic offline
training run using a randomly initialized tiny Qwen3 model, precomputed anchor
logits, token masks, and both TRN-005 terms.
`examples/verify_lora_expert.py` exercises the EXP-002 adapter-only parameter
boundary consumed by the same Trainer.
`examples/verify_independent_expert_set.py` trains two independent data
partitions with distinct seeds and verifies expert updates, exact anchor
restoration, route cleanup, and final freezing.
`examples/verify_local_mini_moe.py` jointly trains two local experts and their
router with BAL-003 enabled, then verifies finite objective composition, exact
optimizer ownership, exact anchor restoration, and final freezing.
`examples/verify_loss_free_bias_balancing.py` applies BAL-004 to deterministic
sparse anchored top-1 routes, reduces a nine-to-one historical assignment
imbalance to five-to-five, proves raw-score immutability and zero trainable
parameters, and round-trips the buffer-only state.
`examples/verify_expert_choice_routing.py` applies BAL-005 to a masked batch,
proves exact equal expert capacity, variable zero/one/multiple-expert token
arity, stable sparse indices, anchor fallback, and selected-gate gradients.
`examples/verify_mixture_aware_refinement.py` trains two paths independently,
refines the same pair jointly, and verifies provenance linkage, second-stage
expert updates, trained local heads, exact anchor restoration, and final
freezing. See
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
- [Hugging Face PEFT LoRA documentation](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
  defines the adapter-only trainable state exposed by EXP-002 experts.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
  motivates temperature-scaled KL preservation. This implementation uses
  precomputed immutable-anchor logits, token-level masks, and a separate causal
  language-model loss.
- [Explicit Inductive Bias for Transfer Learning with Convolutional Networks](https://arxiv.org/abs/1802.01483)
  motivates start-point parameter regularization. This implementation applies
  it only to expert parameters and normalizes by scalar parameter count.
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
  motivates the product of preferred-assignment fractions and mean router
  probabilities. BAL-003 applies it per layer to a dense two-expert local
  mixture, supports token masks, and averages layers; hard assignments measure
  load but do not control execution.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
  motivates additive expert-wise selection biases updated from historical
  load without auxiliary-loss gradients. BAL-004 adds anchored candidate-zero
  exclusion, explicit weighted assignment inputs, strict score-view
  separation, immutable reports, and standalone safetensors state.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  documents the use of auxiliary-loss-free bias balancing at scale. This
  framework implementation is a composable local primitive and does not claim
  the report's distributed training or throughput results.
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
  motivates fixed expert buckets, token-axis top-\(k\), and variable expert
  counts per token. BAL-005 uses explicit integer capacity, sparse indices,
  padding exclusion, and anchor fallback; it omits execution, the paper's dense
  permutation tensor, capacity-factor rounding, distributed training, capped
  token arity, and autoregressive serving.

TRN-001 through TRN-004 are framework infrastructure and introduce no
paper-derived training algorithm. TRN-003 implements the package-local
score-softmax strategy specified by the repository architecture blueprint;
TRN-004 composes its public runtime after completed TRN-002 training.
TRN-005 adapts the referenced distillation and L2-SP objectives with the
differences stated above. BAL-003 adapts the Switch auxiliary objective with
the dense-execution, masking, and layer-averaging differences stated above.
BAL-004 adapts the loss-free sign update with anchored candidate semantics,
explicit completed-batch assignments, and no automatic distributed or Trainer
integration. BAL-005 adapts expert-choice assignment to an immutable-anchor
training/prefill planner with explicit capacity and no execution backend.

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

EXP-002 integration tests additionally prove that a real Trainer optimizer
contains exactly the LoRA adapter tensors, updates at least one adapter, and
leaves both the live anchor and reconstructed private base byte-exact.

TRN-002 tests cover immutable run configuration, deterministic binding and
report order, unique expert IDs, exact rectangular layer coverage, global
cross-expert ownership validation, distinct datasets and seeds, real
independent optimizer steps, updates to every expert, byte-exact anchor state,
empty routed expert lists after completion, complete gradient clearing and
freezing, and forced-anchor restoration after a later data-loading failure.

TRN-003 tests cover strict configuration round trips, exact two-expert
rectangles, fresh-router enforcement, deterministic opposite biases, COM-001
activation math, gradients to both experts and router heads, exact optimizer
ownership, real two-step causal-language-model optimization, immutable
reports, byte-exact anchor restoration, complete gradient clearing and
freezing, and failure-safe cleanup after data loading raises.

TRN-004 tests cover exact source/report topology, stable expert order,
two-expert enforcement, complete binding rectangles, pre-mutation rejection
of malformed Trainer inputs, real TRN-002 independent optimization followed by
real joint refinement, updates to both already-trained experts and fresh local
heads, immutable linked reports, byte-exact anchor preservation, exact
forced-anchor cleanup, complete gradient clearing, and final freezing.

BAL-003 tests cover analytical balanced and collapsed cases, token masking,
zero-token differentiable fallback, invalid probability rejection, router
gradient flow, weighted Trainer loss composition, unchanged optimizer
ownership, detached objective reports, observation cleanup, byte-exact anchor
preservation, and the executable local mini-MoE example.

BAL-004 tests cover exact raw/selection score separation, gradient-free buffer
state, anchor and padding exclusion, deterministic signed updates, balanced
and zero-load behavior, weighted pair-shaped assignments, strict invalid-input
rejection, safetensors round trips and format rejection, resumed state, sparse
anchored top-1 convergence, and its executable offline example.

BAL-005 tests cover exact per-expert bucket size, variable token arity,
multi-expert token selection, anchor fallback, padding exclusion, stable tie
resolution, configuration round trips, strict capacity/shape/value rejection,
decision self-validation, input immutability, selected-gate gradients,
TRN-003-compatible affine-head scores, and its executable offline example.

NB-003 records training-dataset and base-checkpoint provenance but does not
serialize per-example teacher logits or generate them online. Precomputed
full-vocabulary logits can also be storage-intensive; compact teacher targets
are not implemented. Online teacher execution, dataset streaming,
automatic data partitioning, concurrent in-process expert training,
cross-expert diversity objectives, local expert counts other than two, local
capacity, inference-time balancing, TRN-005 objectives inside TRN-003,
router-only refinement, refinement with previously trained router heads,
transactional expert-weight rollback, expert-only optimizer checkpoint
resume, gradient-checkpointing acceptance for captured router probabilities,
mixed-precision, multi-device, and distributed acceptance tests remain future
work. BAL-003 measures preferred load but does not save computation because
TRN-003 always executes both local experts. For the same reason, BAL-004 does
not rebalance TRN-003 computation: meaningful load control requires \(K<N\)
sparse training. Automatic route-decision adapters, per-layer balancer banks,
gradient-accumulation windows, distributed all-reduce, decay or proportional
update variants, bias clipping, and production load/throughput studies remain
future work. BAL-005 is not wired into TRN-003, `RoutedMLP`, or an expert
dispatcher. Capacity-factor resolution, capped experts per token, sparse
activation combination, model/package configuration, generation, distributed
token groups, accelerator performance, and production training remain future
work.
