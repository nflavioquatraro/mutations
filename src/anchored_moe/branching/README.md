# Shared-prefix response branches

## Scope

This additive path executes a frozen prefix once for a request, evaluates a
request router, and resumes exactly one response suffix from that same boundary.
The router may read pooled boundary features or execute its own private suffix
before pooling. It is separate from MLP-level `RoutedMLP` conversion and from the
whole-model PEFT adapter owner. Public imports are available from
`anchored_moe.branching` and the package root.

The first backend is the native JAX/Flax Needle 3 **reference model**. No model
weights are downloaded implicitly. Native `.cact` deployment, optimized cached
decoding, joint expert mixtures and automatic dataset generation are not supplied.

## Implemented strategies

| Feature ID | Configuration / API | Module | Status |
|---|---|---|---|
| BRN-001 | `BranchBackend`, `PrefixState`, `BranchedModelConfig` | `backend.py`, `configuration.py`, `model.py` | Validated |
| BRN-002 | `NeedleBranchBackend`, `from_checkpoint` | `needle.py` | Validated |
| BRN-003 | `FrozenSuffix`, `LoRASuffix`, `CopiedSuffix`, `BranchConfig` | `parameters.py` | Validated |
| BRN-004 | `RequestScorer`, `RequestSelectionPolicy`, `LinearRequestScorer`, `calibrate_utility_selection` | `routing.py` | Validated |
| BRN-005 | `BranchTrainer`, `BranchTrainingStrategy`, `BranchObjective`, `JaxAdamStrategy` | `training.py` | Validated |
| BRN-006 | `save_branch_package`, `load_branch_package`, `evaluate_branch_utilities`, `benchmark_branch_request` | `package.py`, `evaluation.py` | Validated |

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
5. Lock the route for the complete response. Reference greedy decoding recomputes
   the full growing history for later tokens, but never reruns the router.
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
  floating reference does not validate a future quantized deployment.
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
base must be supplied again on reload. No optimizer state or native-engine files
are bundled. Packages refuse overwrite and reject mismatched topology, runtime,
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

Run `python examples/verify_needle_branches.py` for an offline tiny-model workflow:
native equivalence, independent specialist/router training, synthetic held-out
calibration, exact save/reload behavior and synchronized latency samples.

## References

- [Needle architecture](https://github.com/cactus-compute/needle/blob/main/needle/model/architecture.py):
  upstream layer execution and mHC/engram semantics. The backend uses the pinned
  package implementation rather than copying a transformer approximation.
- [LoRA](https://arxiv.org/abs/2106.09685): low-rank projection updates. Difference:
  updates are restricted to the selected suffix; routing is request-level and
  trained independently. This is not a reproduction of token-level MoE training.
- [Optax](https://optax.readthedocs.io/): functional optimizer transformations.
- [safetensors](https://huggingface.co/docs/safetensors/): non-executable tensor payloads.

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

All 28 branch tests pass with the pinned Needle package on CPU. The offline
example checks float32 native parity (maximum absolute logit error approximately
`4.84e-8`), isolated fitting and exact package reload. Lint and wheel build pass.
Full regression: 698 passed, two existing Hub transport symlink tests failed for
Windows privileges unavailable on this validation host. An extended-path pytest
temporary directory avoids the host's separate legacy path-length limit. Those
Hub tests are outside the branch implementation.

These are CPU correctness tests on random small models, not real Needle 3 quality
or performance evaluations. No native `.cact` branch export, grammar/confidence
integration, quantization-aware branch training, incremental KV-cache execution,
cross-response cache reuse, padded batches, distributed training, optimizer resume,
mixed response branches in one batch, joint branch training, runtime package
installation/removal or learned engram updates are implemented. `generate_tokens`
is a research decoder with an optional logits-processor hook; it does not promise
schema-valid tool calls without a caller-supplied constraint processor.
