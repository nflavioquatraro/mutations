# Evaluation

## Scope

The evaluation subpackage measures architectural invariants. The current scope
includes dense equivalence, anchor-state immutability, exact route agreement,
package removal/reinsertion lifecycle checks, and top-1/top-2 quality
comparison against explicit counterfactual candidates. EVAL-005 adds
warmup-aware operation benchmarking, exact resident module-state footprint,
optional CUDA peak allocation, and route-derived active-MLP accounting.
EVAL-006 exhaustively evaluates installation orders and registry subsets under
an explicit combinatorial trial bound, with optional caller-owned behavior
fingerprints. Production benchmark orchestration remains outside this
subpackage.

## Implemented strategies

| Feature ID | Public API | Module | Status |
|---|---|---|---|
| EVAL-001 | `assert_tensors_equivalent`, `fingerprint_module` | `equivalence.py` | Validated |
| EVAL-002 | `compare_route_decisions`, `assert_route_decisions_equal` | `routing.py` | Validated |
| EVAL-003 | `evaluate_registry_lifecycle` | `lifecycle.py` | Validated |
| EVAL-004 | `compare_top1_top2_to_oracle` | `quality.py` | Validated |
| EVAL-005 | `benchmark_operation`, `measure_active_mlp_metrics`, `measure_module_state_footprint` | `efficiency.py` | Validated |
| EVAL-006 | `evaluate_registry_compositions` | `registry_composition.py` | Validated |

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

For selected top-1 loss \(L_t^{(1)}\), selected top-2 loss \(L_t^{(2)}\),
best explicitly evaluated single-specialist loss \(L_t^{S}\), and best sampled
pair loss \(L_t^{P}\), EVAL-004 reports

\[
G_t^{(2)}=L_t^{(1)}-L_t^{(2)},
\qquad
G_t^{P}=L_t^{S}-L_t^{P}.
\]

Positive \(G_t^{(2)}\) means the selected second expert helped; positive
\(G_t^{P}\) means some sampled pair can beat every individual specialist.
Router regret for policy \(\pi\) uses CAL-007:

\[
R_t^\pi=L_t^\pi-\min_{c\in\mathcal C}L_{t,c}.
\]

The destructive-mixture rate is measured only over tokens that execute two
specialists:

\[
\frac{
\sum_t \mathbb{1}[k_t=2]\mathbb{1}[L_t^{(2)}>L_t^{(1)}]
}{
\sum_t \mathbb{1}[k_t=2]
}.
\]

It is defined as zero when no token executes a pair.

For \(M\) measured runs, \(W\) declared work items per run, and synchronized
wall-clock durations \(\ell_i>0\), EVAL-005 reports

\[
\bar{\ell}=\frac{1}{M}\sum_{i=1}^{M}\ell_i,
\qquad
Q=\frac{MW}{\sum_{i=1}^{M}\ell_i}.
\]

Median latency and the deterministic nearest-rank 95th percentile retain the
same raw duration sample. Throughput \(Q\) is meaningful only when callers use
the same work-unit definition across strategies.

For token \(t\), anchored top-1 has one active MLP. Pair-shaped top-2 has pair
indicator \(p_t\), where one means two specialists execute and zero means the
anchor executes:

\[
C_{\mathrm{top1}}=N,
\qquad
C_{\mathrm{top2}}=\sum_t(1+p_t).
\]

For adaptive route arity \(k_t\in\{0,1,2\}\), with zero denoting the anchor:

\[
C_{\mathrm{adaptive}}
=
\sum_t\left(\mathbb{1}[k_t=0]+k_t\right).
\]

Mean active MLPs per token is \(C/N\), defined as zero for an empty decision.
Resident module-state bytes sum unique parameter and buffer storages. Shared
storage is counted once; optimizer state, activations, allocator cache, Python
objects, and serialized files are excluded.

For \(N\) packages, EVAL-006 evaluates the empty registry and every
installation order of every non-empty subset. The exact trial count is the
number of partial permutations:

\[
C_N
=
\sum_{k=0}^{N}P(N,k)
=
\sum_{k=0}^{N}\frac{N!}{(N-k)!}.
\]

For one canonical package subset \(S\), let \(d_\pi(S)\) be the ordered
registry digest after installation order \(\pi\), and let \(b_\pi(S)\) be an
optional caller-supplied behavior digest. Registry-order sensitivity and
behavior-order invariance are:

\[
O_{\mathrm{registry}}(S)
=
\mathbb{1}\left[
\left|\{d_\pi(S)\}_{\pi\in\Pi(S)}\right|>1
\right],
\]

\[
I_{\mathrm{behavior}}(S)
=
\mathbb{1}\left[
\left|\{b_\pi(S)\}_{\pi\in\Pi(S)}\right|=1
\right].
\]

Registry hashes intentionally include active package order, so
\(O_{\mathrm{registry}}\) is descriptive rather than a failure. A behavior
fingerprint can instead ignore order or encode model routes, logits, or a
metric artifact, depending on the experiment.

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

Top-1/top-2 oracle comparison:

1. Require identical token shape and expert registry order.
2. map top-1 candidate indices directly to anchor or single columns;
3. map anchor-vetoed top-2 routes to the anchor;
4. canonicalize active pair identity and require its sparse loss column;
5. evaluate each policy against the CAL-007 global oracle;
6. compute best-single, best-pair, selected-pair gain, oracle-pair gain, and
   destructive-mixture statistics.

Efficiency benchmarking:

1. Validate the run counts, work-unit count, and benchmark device.
2. Execute configured warmup runs and synchronize asynchronous device work.
3. Optionally reset CUDA peak-allocation statistics after warmup.
4. Synchronize immediately before and after every measured operation.
5. Retain every positive wall-clock duration and calculate aggregate latency
   and throughput properties.
6. Count unique supplied parameter and buffer storages after execution but
   outside timed regions so lazy state is represented.

Active-MLP accounting:

1. Read only the immutable route decision.
2. Count anchor-executed tokens using the policy's anchor mask.
3. Count active specialist slots in stable expert order.
4. Sum anchor and specialist token executions.
5. Count non-empty anchor and expert groups as ideal sparse module calls.

Registry composition evaluation:

1. Validate package descriptor types, unique immutable versions, the behavior
   callback, and the positive trial bound.
2. Canonicalize packages by portable identity and calculate \(C_N\).
3. Reject an oversized experiment before package preflight or observation.
4. Install the complete set once in a temporary registry to preflight strict
   file validation and anchor/router-feature compatibility.
5. Enumerate subsets by size and canonical combination order.
6. Install every subset permutation into a fresh registry and capture its
   immutable snapshot.
7. Optionally call the explicit behavior observer, require a lowercase SHA-256
   result, and prove the observer did not mutate the supplied registry.
8. Group trials by canonical package set and report distinct registry and
   behavior counts.

## Assumptions and invariants

- Inputs, model mode, device, and dtype are identical between executions.
- Stochastic layers are disabled by evaluation mode.
- Comparison never mutates tensors or modules.
- A fingerprint proves tensor-state equality, not semantic compatibility.
- Route indices are comparable only under identical expert ID ordering.
- Route comparison never records token content, hidden states, or activations.
- Lifecycle evaluation consumes immutable snapshots and never mutates a
  registry.
- Counterfactual columns share one finite loss definition and exact token
  alignment.
- Top-1, top-2, and loss-table expert IDs use one stable registry order.
- EVAL-004 never assumes an unevaluated pair loss. Sparse coverage failures are
  explicit errors.
- Pair identities are order-independent for lookup, while CAL-007 retains
  deterministic registry order in the table.
- Anchor-vetoed top-2 tokens execute and are evaluated as the exact anchor.
- Aggregate second-expert and destructive-mixture metrics include only tokens
  that actually execute two specialists.
- Benchmark setup cost is included only when the caller places it inside the
  zero-argument operation.
- Device synchronization brackets every measured run; module mode, gradient
  mode, autocast, seeds, and thread counts remain caller-controlled.
- Work items are explicit and positive. EVAL-005 does not infer whether they
  represent tokens, sequences, or examples.
- Top-2 anchor-veto metadata counts one anchor execution and no specialist
  execution, even when the inactive ranked slot contains an expert candidate.
- Active-MLP metrics count mathematical token executions separately from
  sparse module calls.
- CUDA peak allocation is allocator-global for the configured device after the
  warmup boundary. CPU and MPS runtime peaks are reported as unavailable.
- Shared state storage is counted once across parent/child or repeated module
  references.
- Package input order does not affect EVAL-006 enumeration or its report.
- The empty package subset is always evaluated exactly once.
- Every non-empty subset contains exactly \(k!\) installation trials.
- The default 2,048-trial bound permits six packages and rejects seven before
  enumeration.
- Complete-set preflight runs before any behavior observer, so incompatible,
  non-strict, changed, or duplicate-version packages cannot produce a partial
  behavioral report.
- Every installation order uses a fresh registry. Historical slots from one
  trial cannot leak into another.
- The optional behavior observer is caller-owned and may inspect or explicitly
  synchronize external state, but it must not mutate the registry passed to
  it.
- Behavior fingerprints record only SHA-256 digests. EVAL-006 never records
  prompts, hidden states, logits, routes, or metric payloads itself.

## Hugging Face integration

The integration test uses a tiny locally initialized `Qwen3ForCausalLM` and
requests `output_hidden_states=True`. No model or tokenizer download is needed.

EVAL-004 is an offline PyTorch transform over policy decisions and losses
measured elsewhere. It does not require a Hugging Face model, but accepts
outputs from deterministic Dataset transforms or model evaluation loops.

EVAL-005 benchmarks any zero-argument operation, including a Hugging Face
`forward()` or `generate()` closure and the standalone sparse dispatchers. The
caller must put the model in the intended evaluation/precision context and set
`work_items_per_run` consistently. CUDA synchronization and allocator peak
measurement use PyTorch runtime APIs without changing model state.

EVAL-006 is model-neutral. A behavior callback may explicitly synchronize an
`AnchoredMoEForCausalLM`, execute fixed Hugging Face inputs, and return a digest
of routes or logits. The evaluator itself never loads a model, synchronizes a
registry, recalibrates a router, or retains behavioral tensors.

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

comparison = compare_top1_top2_to_oracle(
    counterfactual_losses,
    top1_decision,
    top2_decision,
)
print(comparison.mean_top1_regret)
print(comparison.mean_top2_regret)
print(comparison.destructive_mixture_fraction)

config = EfficiencyBenchmarkConfig(
    warmup_runs=2,
    measured_runs=20,
    work_items_per_run=input_ids.numel(),
    device=str(input_ids.device),
)
runtime = benchmark_operation(
    lambda: model(input_ids=input_ids),
    config=config,
    modules=model,
)
work = measure_active_mlp_metrics(route_decision)
print(runtime.throughput_items_per_second)
print(runtime.module_state.total_bytes)
print(work.mean_active_mlps_per_token)

composition = evaluate_registry_compositions(
    (code_package, math_package),
    behavior_fingerprint=lambda registry: route_digest(registry),
    maximum_trials=2_048,
)
print(composition.subset_count)
print(composition.trial_count)
print(composition.behavior_order_sensitive_subset_count)
```

`notebooks/01_convert_dense_model.ipynb` exercises this API against logits,
every hidden state, and the selected anchor MLP fingerprints. It also reports
the post-conversion trainable-parameter count.
`examples/verify_top1_top2_oracle.py` exercises sparse pair coverage,
deterministic oracle selection, router regret, and destructive mixing without
model downloads.
`examples/evaluate_registry_strategies.py` compares top-1 and top-2
synchronized latency, throughput, state footprint, active MLP work, and
counterfactual regret with a deterministic offline fixture.
`examples/verify_registry_lifecycle.py` evaluates removal/reinsertion and then
all five partial permutations of two packages using an order-invariant
package-set behavior digest.
`notebooks/05_evaluate_registry.ipynb` provides the corresponding thin public
API workflow.

## References

- The repository
  [architecture blueprint](../../../anchored-plugin-moe-architecture.md)
  specifies exact fallback, forced-anchor equivalence, counterfactual router
  regret, and the required top-1/top-2 quality metrics.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  motivates explicit tolerance declarations for future runtime variants.

No paper-derived evaluation algorithm is implemented here. EVAL-004 directly
operationalizes metrics defined by the repository architecture. EVAL-005 uses
standard synchronized wall-clock and allocator accounting rather than a
paper-derived benchmark method. EVAL-006 directly implements the blueprint's
required installation-order and every-subset lifecycle experiments using
standard finite permutations and SHA-256 digests.

## Tests and known limitations

CPU tests require exact equality because the wrapper executes the same MLP
object with no additional arithmetic. Tolerance-based accelerator and optimized
kernel checks will be added with functional compatibility support.

Route tests cover exact and partial agreement, diagnostics, registry-order
validation, configured Qwen3 expert execution, and privacy-safe telemetry.
Lifecycle tests cover removal identity changes, exact package-count reduction,
reinserted hash/order restoration, and persistent compatibility binding.
EVAL-004 tests cover exact policy-to-candidate mapping, anchor veto, reversed
pair ranking, sparse-pair coverage, registry and token-shape alignment,
best-single and best-pair losses, top-1/top-2 regret, second-expert gain,
destructive-mixture rate, policy integration, and executable example behavior.
EVAL-005 tests cover unique shared storage, non-resident meta tensors, lazy
state initialization, frozen/trainable byte accounting, top-1/top-2/adaptive
execution counts, empty routes, warmup exclusion, repetition counts, CPU
timing aggregates, public serialization mappings, dispatcher integration, and
invalid inputs. Timing assertions validate contracts and positivity, never
machine-specific performance thresholds.
EVAL-006 tests cover the empty registry, canonical input-order independence,
exhaustive subset/permutation counts, registry-hash sensitivity, invariant and
sensitive behavior observers, JSON-compatible reports, callback mutation,
invalid digests, duplicate versions, incompatible complete-set preflight,
factorial budget rejection before observation, live package revalidation, the
documentation contract, and the executable lifecycle example.
Cross-layer trajectory metrics, first-divergence depth, route serialization,
persistent registry evaluation, model-integrated counterfactual execution,
confidence intervals, CPU/MPS runtime peak memory, energy measurement,
distributed aggregation, production hardware baselines, and statistical
benchmark comparison remain future work. EVAL-006 deliberately does not define
which behavior artifact should be fingerprinted, compare metric magnitudes,
persist reports, parallelize trials, or attempt exhaustive evaluation beyond
the caller-approved bound.
