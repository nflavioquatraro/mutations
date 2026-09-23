# Shared-prefix response branches

## Scope

This additive path executes a frozen prefix once for a request, evaluates a
request router, and resumes exactly one response suffix from that same boundary.
The router may read pooled boundary features or execute its own private suffix
before pooling. It is separate from MLP-level `RoutedMLP` conversion and from the
whole-model PEFT adapter owner. Public imports are available from
`anchored_moe.branching` and the package root.

The first backend is the native JAX/Flax Needle 3 **reference model**, with
incremental KV-cached generation. Standard `.cact` export is available for each
response branch. The stock native engine runs a complete model per branch;
request routing and prefix sharing are not part of its published ABI. No model
weights are downloaded implicitly. Joint expert mixtures and automatic dataset
generation are not supplied.

## Implemented strategies

| Feature ID | Configuration / API | Module | Status |
|---|---|---|---|
| BRN-001 | `BranchBackend`, `PrefixState`, `BranchedModelConfig` | `backend.py`, `configuration.py`, `model.py` | Validated |
| BRN-002 | `NeedleBranchBackend`, `from_checkpoint` | `needle.py` | Validated |
| BRN-003 | `FrozenSuffix`, `LoRASuffix`, `CopiedSuffix`, `BranchConfig` | `parameters.py` | Validated |
| BRN-004 | `RequestScorer`, `RequestSelectionPolicy`, `LinearRequestScorer`, `calibrate_utility_selection` | `routing.py` | Validated |
| BRN-005 | `BranchTrainer`, `BranchTrainingStrategy`, `BranchObjective`, `JaxAdamStrategy` | `training.py` | Validated |
| BRN-006 | `save_branch_package`, `load_branch_package`, `evaluate_branch_utilities`, `benchmark_branch_request` | `package.py`, `evaluation.py` | Validated |
| BRN-007 | `generate_tokens(use_cache=...)`, `NeedleDecodeState`, `benchmark_branch_generation` | `backend.py`, `needle_cache.py`, `model.py`, `evaluation.py` | Validated |
| BRN-008 | `CactExportConfig`, `export_needle_branch`, `export_needle_bundle` | `native_export.py` | Validated |

## Mathematical formulation

For input tokens x, prefix P and suffixes S_k, the response model is
`S_k(P(x))`. The frozen original suffix is an explicit fallback candidate.
Needle's split representation is a full residual-lane stream with shape
`[1, sequence, lanes, width]`, plus masks, rotary positions and frozen engram
key/value arrays. A pooled `[width]` vector is suitable for routing, but is not
sufficient to resume a suffix.

`features = mean(P(x), sequence and lane axes)` gives the linear-router input.
With a router suffix, features are the mean over sequence positions of its
final-normalized output. Both exclude future response tokens at inference.
Scoring and selection are separate: `s = features @ W + b`, with
`W: [width, candidates]`, `b: [candidates]`.

Utility heads predict `loss(base) - loss(specialist)`. Base evidence is fixed at
zero, and ties favor base. Probability heads instead apply temperature softmax
over all responses, including base. A utility score is not a probability.
`ThresholdSelection` tests the argmax candidate with strict `score > threshold`;
rejection returns base without searching lower-ranked candidates.

LoRA uses native Needle matrix orientation `[input, output]`:
`W_effective = W_base + alpha / rank * A @ B`. A is seeded, B starts at zero.
Copied suffixes own all block and layer-specific mHC tensors in their suffix.
Engram memory and projections, embeddings, final norm and the tied LM head remain
frozen and shared in every strategy. No dense MLP replacement is assumed.

Specialist training minimizes masked shifted causal cross-entropy, with optional
temperature-scaled KL from precomputed anchor logits and parameter-normalized
squared drift from the beginning of that training call. Router fitting uses
independent ridge regression, utility MSE or categorical cross-entropy. Adam
receives only the explicitly owned suffix/head parameter tree.

## Algorithms

1. Pin the checkpoint, configuration, tokenizer identity, dependency versions and
   split point. Initialize independent branch payloads without changing the base.
2. Encode one unpadded request and execute the shared prefix once.
3. Run the scorer over prefix features or a separate router suffix. Select one
   response branch; the router never overwrites the saved prefix state.
4. Resume that response suffix, applying the original frozen output projection.
5. Lock the route for the complete response. Prefill the selected suffix once,
   then append one token per compiled KV update. The router is never rerun.
   `use_cache=False` retains full-history recomputation as a parity oracle.
6. Train specialists independently with forced branches; train the router on
   pre-response examples. Cache frozen prefix states for each training call.
7. Fit held-out thresholds and bind them to the complete composition hash. Any
   branch or scorer update invalidates that binding; routed inference rejects it.
8. Export a staged, strictly reloaded JSON/safetensors package, then publish it.

`evaluate_branch_utilities()` performs offline counterfactual evaluation with
one prefix per request and every response suffix. Its caller-supplied scalar
loss may express task accuracy or another metric. Do not accidentally feed a
teacher-forced target answer into the request router's training inputs.
`benchmark_branch_request()` synchronizes via a host logits read and measures
warm prefill plus routing plus response-suffix time. It excludes checkpoint load,
tokenization and generation, and is not an optimized-device throughput claim.
`benchmark_branch_generation()` measures the entire generated response, including
prefill, routing and cache setup. Warmup excludes compilation for the tested
shape/capacity. It compares the JAX cached and uncached paths, not native CQ4 speed.

The cache stores post-norm/rotary keys and values in native grouped-query head
geometry. Global layers allocate the reserved token capacity. Local layers use
`min(sliding_window, capacity)` ring slots with absolute-position validity masks.
Raw Q/K/V convolution tails are retained separately. Frozen engrams use a bounded
token history of `4 * max(engram_orders)`, which covers the n-gram lookup and all
four dilated value-convolution taps. Native mHC and Hadamard modules still run;
only attention is intercepted to substitute incremental execution.

Native export materializes the selected suffix in a full original parameter tree,
then calls the pinned upstream CQ4 writer with 128-element groups. It validates
the tokenizer, rereads geometry/tensors and rejects nonfinite quantization output
before atomic publication. Original embedding/confidence/router probe heads are
omitted because their behavior has not been validated for the modified/quantized
model. These probe heads are separate from this package's affine request scorer.

## Assumptions and invariants

- `shared_prefix_layers` is a layer count, not an index. It must leave at least
  one prefix and one suffix layer. The initial implementation uses a contiguous
  split of the materialized model, not Needle's interleaved depth-ladder selection.
- Requests are integer `[1, sequence]` arrays with no padding. Tokenization and
  schema/system-prompt rendering are caller responsibilities. Reusing a state
  with different tokens, positions or prefix weights is not supported.
- The Needle backend retains original layer indices, local/global attention
  masks, convolution context and mHC lane offsets. Shared engram computation is
  frozen and can be reused by both router and response suffixes.
- Native JAX arrays and branch snapshots are immutable. Public parameter reads
  return defensive copies. An owner lock rejects nested/concurrent operations.
  Direct private-field mutation is unsupported.
- Decode snapshots retain private effective suffix weights and immutable cache
  pytrees. Forking a snapshot cannot alter another response. Overflow, padding
  tokens and incompatible backend snapshots are rejected. All updates process one
  new token; prompt residual and full engram arrays are released after prefill.
  Different reserved capacities can require separate JIT compilations.
- Training optimizes candidate trees and publishes only after complete validation.
  Exceptions leave the original owner usable and unchanged. Base tensors never
  enter the optimizer parameter tree. Shared-prefix training is unsupported.
- Calibration limits false activation independently for each head on observed
  nonbeneficial rows. It is not a bound on the combined router or future traffic.
  Fitting/calibration IDs known to the owner cannot overlap; content-level split
  verification is the caller's responsibility. Manual thresholds explicitly have
  no calibration guarantee.
- Base equivalence is numerical relative to the unsplit reference, not guaranteed
  bit equality across different scan/compiler execution orders. Zero-update
  branches equal the split base exactly in the tested numerical configuration.
- Native quantized execution is not reproduced: the backend uses `quant=False`.
  Runtime versions are part of package identity. Calibrating or benchmarking the
  floating reference does not validate a quantized deployment.
- Native `.cact` files use CQ4 weights and the checkpoint's declared deployment KV
  settings. Their quantized numerics differ from this floating backend. Bundle
  metadata marks source routing calibration as requiring native-outcome evaluation.
  A bundle repeats the prefix in each native response model and records external
  routing explicitly; it is not a new native branch-container format.
- Custom scorers implement `score(features)`. For calibration fingerprints they
  also supply `identity()` containing all version-affecting state. Custom
  selectors implement `select(scores, fallback)`. Only builtin scorer/selector
  formats currently serialize; unsupported custom formats fail explicitly.

## Hugging Face integration

Install optional training dependencies with `pip install -e ".[needle]"`.
Normal package import does not load JAX, Flax, Optax or Needle. The backend pins
`cactus-needle==3.0.4` because it deliberately calls the upstream private scan
body; upgrading requires parity validation before relaxing that pin.

`NeedleBranchBackend.from_checkpoint()` reads a caller-downloaded format-v2
`.safetensors` master checkpoint with configuration metadata. Pickle and `.cact`
files are rejected. No Hugging Face Trainer or PEFT wrapper is imposed on Flax.
Tensor digests independently identify the actual loaded base; model revision and
tokenizer IDs are caller-supplied provenance, not automatic tokenizer validation.

Deployment packages contain `manifest.json` and `branches.safetensors`, including
owned branch tensors, affine scorer, selection settings and data IDs. The pinned
base must be supplied again on reload. These reference packages contain no
optimizer state or native-engine files. Packages refuse overwrite and reject mismatched topology, runtime,
base, tensor keys, shapes, dtypes, nonfinite values or stale calibration.
Integrity hashes are not signatures or publisher authentication.

## Configuration and examples

```python
import numpy as np
from anchored_moe.branching import (
    BranchConfig,
    BranchedModelConfig,
    BranchedModel,
    NeedleBranchBackend,
    LinearRequestScorer,
    ThresholdSelection,
)

backend = NeedleBranchBackend.from_checkpoint(
    "checkpoints/needle3.safetensors",
    shared_prefix_layers=16,
    model_id="Cactus-Compute/needle3",
    revision="<pinned-commit>",
    tokenizer_id="<pinned-tokenizer-identity>",
)
config = BranchedModelConfig(
    shared_prefix_layers=16,
    response_branches=(
        BranchConfig("base", "frozen"),
        BranchConfig("calendar", "lora", rank=8),
    ),
    router_branch=BranchConfig("router", "lora", rank=8),
)
scorer = LinearRequestScorer(
    ("calendar",),
    np.zeros((backend.feature_size, 1), np.float32),
    [0.0],
)
model = BranchedModel(
    backend,
    config,
    scorer,
    ThresholdSelection((("calendar", 0.0),)),
)
# Zero utility defaults to base until specialist/router fitting and calibration.
# Pass router_branch=None to use a linear head directly on prefix features.
# Use parameterization="copy" for an independently trained complete suffix.
```

`BranchTrainer.train_specialist()` accepts `ResponseTrainingExample` objects
with `-100` labels on prompt positions. `train_router()` accepts
`RouterTrainingExample` objects and can fit the head alone with
`train_suffix=False`. Custom optimization strategies implement
`optimize(parameters, loss_fn)` and return a candidate tree plus a training
report. Custom objectives supply validation and differentiable scalar losses.

Cached decoding is the default when the backend supports it:

```python
result = model.generate_tokens(input_ids, max_new_tokens=64, use_cache=True)
assert result.prefix_evaluations == 1
# use_cache=False selects the original full-history parity/debugging path.
```

Export a native response or the complete response set using the original packaged
tokenizer. Neither function fetches a checkpoint, tokenizer or engine:

```python
from anchored_moe.branching import export_needle_branch, export_needle_bundle

artifact = export_needle_branch(
    model,
    "calendar",
    "deployment/calendar.cact",
    tokenizer="checkpoints/needle3.cact",
)
bundle = export_needle_bundle(
    model,
    "deployment/all-branches",
    tokenizer="checkpoints/needle3.cact",
)
```

Use `artifact.path` with the official engine's `--model` option or
`needle.Needle(weights=str(artifact.path), tools=...)`. Native engines are installed
separately. The bundle contains one `.cact` per response, `manifest.json`, and
`routing.safetensors` with the affine head and optional router suffix. That router
still requires the source JAX backend/base. The manifest records this distinction;
the native engine cannot resume its prefix from a JAX router state.
`model.route_request(input_ids)` exposes the external routing decision without
executing a response suffix. An application can select the corresponding native
agent from its branch-ID mapping and call `complete()` on that agent. Render the
router's input consistently with its fitting data and the deployed request/tools.

Run `python examples/verify_needle_branches.py` for an offline tiny-model workflow:
native equivalence, independent specialist/router training, synthetic held-out
calibration, exact save/reload behavior and synchronized latency samples.
Run `python examples/verify_needle_deployment.py` for cached/reference generation
parity, two native response exports and warm end-to-end decoding measurements.

## References

- [Needle architecture](https://github.com/cactus-compute/needle/blob/main/needle/model/architecture.py):
  upstream layer execution and mHC/engram semantics. The backend uses the pinned
  package implementation rather than copying a transformer approximation.
- [LoRA](https://arxiv.org/abs/2106.09685): low-rank projection updates. Difference:
  updates are restricted to the selected suffix; routing is request-level and
  trained independently. This is not a reproduction of token-level MoE training.
- [Optax](https://optax.readthedocs.io/): functional optimizer transformations.
- [safetensors](https://huggingface.co/docs/safetensors/): non-executable tensor payloads.
- [Needle exporter](https://github.com/cactus-compute/needle/blob/main/needle/model/export.py):
  standard native geometry, tensor order, tokenizer and CQ4 encoding.
- [Native format](https://www.cactuscompute.com/blog/cact-format) and
  [porting guide](https://www.cactuscompute.com/blog/porting-needle): native model
  contract. This implementation exports full response paths, without extending
  the engine ABI or claiming native sharing across branches.

Split ownership, package contracts, counterfactual utilities and threshold binding
are framework infrastructure, not claims to reproduce a separate paper algorithm.

## Tests and known limitations

`tests/unit/test_branching.py` validates configuration, score semantics, ties,
thresholds, stale calibration, prefix execution counts, isolated updates, owner
cleanup, generation route locking and strict artifact reload/tamper rejection.
`tests/integration/test_needle_branches.py` tests actual tiny native Needle layers,
including engram sites on both sides, QKV convolution, local/global attention,
non-lane-aligned boundaries, LoRA/full-copy training, router-suffix training and
reference greedy generation. Further cases cover master checkpoint loading,
head-only fitting with preserved temperature, bfloat16 attention without engrams,
asymmetric attention heads, padding suppression and generation context limits.
Optional tests skip when dependencies are absent.

Additional tests compare cached logits after every appended token, including
nonzero QKV/engram taps, short and long prompts, local-ring wraparound, bfloat16
attention, unequal head dimensions, forked snapshots and specialized suffixes.
Export tests check unchanged prefix records, LoRA/full-copy materialization,
tokenizer validation, failure cleanup and overwrite rejection. The optional
native smoke test uses `ANCHORED_MOE_NEEDLE_LIBRARY` pointing to a caller-installed
library and exercises `needle_load`, `needle_init` and `needle_complete` in a
separate process. It never downloads an engine during tests. Native kernel
geometry limits still apply: the width-16 JAX toy fixture is rejected by the
pinned engine; native acceptance uses width 128, 64-dimensional heads and four
residual lanes. Format roundtrip alone is not a native execution guarantee.

All 38 branch tests pass across full regression and final focused reruns with
the pinned Needle package on CPU. The original offline
example checks float32 native parity (maximum absolute logit error approximately
`4.84e-8`), isolated fitting and exact package reload. Lint and wheel build pass.
Full regression: 707 passed, two existing Hub transport symlink tests failed for
Windows privileges unavailable on this validation host. An extended-path pytest
temporary directory avoids the host's separate legacy path-length limit. Those
Hub tests are outside the branch implementation.

The deployment example observed roughly 0.16 seconds cached versus 1.07 seconds
uncached for six generated tokens after warmup (about 6.6x on this host). This
compares the new compiled cache path with the existing uncached backend on a
small random CPU model; it is not a production-model speed guarantee. The native
smoke test passed with the official Windows x86_64 engine 3.0.1, downloaded from
`Cactus-Compute/needle3` revision `b274efcb211a9eef48c9a88da4b43bd569696a39`.
Native model loading, prompt prefill and short token decoding are covered;
real-weight quality and device performance still require evaluation.

These are CPU correctness tests on random small models, not real Needle 3 quality
or performance evaluations. No native shared-prefix branch ABI, grammar/confidence
integration in the JAX decoder, quantization-aware branch training, automatic
cross-response cache reuse, padded batches, distributed training, optimizer resume,
mixed response branches in one batch, joint branch training, runtime package
installation/removal or learned engram updates are implemented. `generate_tokens`
is a research decoder with an optional logits-processor hook; it does not promise
schema-valid tool calls without a caller-supplied constraint processor.
