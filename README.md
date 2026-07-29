# mutations
# Anchored Plug-in MoE

Anchored Plug-in MoE is a research framework for installing independently
trained specialization packages into an immutable Hugging Face causal language
model. The dense model remains an exact fallback while expert representation,
routing, combination, calibration, and package lifecycle remain independently
configurable.

The current implementation covers the modular top-1 foundation and the first
standalone weighted top-2 strategy primitives:

- declarative `AnchoredMoEConfig` configuration;
- a model-family-neutral backend contract;
- Qwen3 MLP discovery and replacement;
- a forced-anchor routed MLP;
- hidden-state, logit, and anchor-state equivalence utilities;
- a versioned expert-package manifest with canonical JSON and SHA-256
  validation;
- full copied-MLP experts with independent trainable storage and strict,
  deterministic safetensors serialization;
- standalone PEFT/LoRA experts with private frozen anchor copies, exact
  zero-update behavior, adapter-only training, and deterministic adapter-only
  safetensors serialization;
- frozen identity router features over pre-MLP hidden states;
- independent append-only affine router heads with calibrated scores and
  strict, deterministic safetensors serialization;
- a frozen registry-level logical router that compiles independent heads into
  layer-local append-only matrices while preserving incumbent scores exactly;
- deterministic router-only global recalibration that jointly optimizes every
  installed affine score column against anchor-relative utility replay while
  leaving the source router, anchor, and experts unchanged;
- deterministic anchored hard-route distillation from an immutable teacher
  score bank into a storage-independent frozen logical deployment router, with
  hashed assignments and exact replay-agreement reports;
- standalone safetensors registry-router versions with exact snapshot,
  feature-space, report, expert-order, and tensor-digest validation plus
  explicit stale-safe activation;
- anchored hard top-1 selection with fixed-zero anchor evidence and stable
  anchor-first tie-breaking;
- deterministic anchor-veto top-2 selection with temperature-scaled
  score-softmax, equal-weight, and normalized-positive-evidence activation
  combination policies;
- deterministic soft-oracle fitting of a separate frozen affine mixture-weight
  head, with immutable reports, standalone safetensors serialization, and
  unchanged top-2 selection;
- adaptive anchor/top-1/top-2 selection with explicit auditable route arity
  and deterministic mixture-threshold behavior;
- explicit always-on-anchor execution with sparse top-1 or adaptive top-2
  PEFT/LoRA deltas, exact dense fallback, and capacity-aware degradation;
- confidence-gated dynamic-\(k\) selection with stable specialist ranking,
  explicit per-token compute bounds, and auditable variable route arity;
- immutable package-declared expert compatibility groups and deterministic
  global top-2 selection that never mixes unrelated or undeclared experts;
- standalone sparse anchor-veto or compatibility-constrained top-2 dispatch
  with ranked specialist execution, weighted activation reconstruction, and
  auditable mixture weights;
- capacity-aware top-2 degradation that retains both available specialists,
  promotes one selected survivor with coefficient one, or returns overflow to
  the immutable anchor without searching beyond the selected pair;
- versioned router feature caches with deterministic row alignment, provenance,
  strict safetensors validation, and embedded tensor integrity digests;
- feature-bank-bound domain and individual-utility label datasets with
  deterministic class and expert ordering;
- deterministic independent ridge fitting from utility targets into frozen
  affine router heads;
- immutable per-head temperature scaling and held-out threshold selection
  under an explicit false-activation budget;
- deterministic candidate-only stability-margin refinement against an
  immutable incumbent router, with explicit old/general and useful new-domain
  partitions plus post-refinement route metrics;
- immutable sparse counterfactual route-loss tables with deterministic
  anchor-first oracle selection and per-token router regret;
- deterministic bounded or explicitly exhaustive expert-pair plans with
  anchor utility, improvement-over-best-single targets, stable best-pair
  selection, and direct counterfactual-table construction;
- top-1/top-2 counterfactual comparison with best-single and best-pair loss,
  second-expert gain, and destructive-mixture metrics;
- synchronized warmup-aware latency and throughput benchmarks, exact unique
  module-state footprint, optional CUDA peak allocation, and route-derived
  active-MLP accounting;
- deterministic bounded evaluation of the empty registry, every package
  subset, and every installation order, with optional read-only behavior
  fingerprints and explicit order-sensitivity reports;
- deterministic router-data scaling reports comparing strict append-only,
  router-only recalibration, and a matched local partial MoE across nested
  replay subsets and held-out specialization-interior, boundary, and mixed
  strata;
- opt-in sparse anchored top-1 execution for configured routed MLPs;
- deterministic overflow-to-anchor fallback with no token dropping;
- bounded privacy-safe route telemetry and exact route-agreement metrics;
- strictly validated local expert-package descriptors with atomic in-memory
  install, unload, stable ordering, compatibility binding, and registry hashes;
- strict Hugging Face Hub expert-package download with manifest-first immutable
  revision pinning, content-addressed regular-file caching, and explicit staged
  upload of only manifest-tracked package files;
- slot-preserving immutable package upgrades with exact, stale-safe rollback
  and no implicit model synchronization;
- explicit versioned `(layer, expert_id)` payload bindings and safe local
  package materialization into converted anchored top-1 layers;
- explicit registry-to-model synchronization with exact anchor restoration on
  unload and deterministic routed behavior after reinsertion;
- forced-path single-expert training with complete converted-layer coverage,
  mechanical backbone freezing, exact trainable-parameter assertions, and a
  constrained Hugging Face Trainer/Accelerate integration;
- independent latent expert-set training with per-expert datasets, seeds, and
  Trainer configuration, global storage-isolation checks, immutable reports,
  and failure-safe anchor-route restoration;
- deterministic two-expert local mini-MoE training with package-owned affine
  router heads, score-softmax activation mixing, exact optimizer isolation,
  optional token-masked auxiliary load balancing, auditable objective reports,
  and success/failure-safe anchor-route restoration;
- provenance-validated mixture-aware refinement that links a completed
  two-expert independent-training report to joint expert-and-router updates
  through the same local mini-MoE runtime;
- optional token-masked replay KL against precomputed immutable-anchor logits
  and normalized expert-initialization weight-drift regularization;
- staged expert-only package export with deterministic payload bindings,
  derived metadata, strict reload, and failure cleanup;
- immutable calibrated-router package derivation with disjoint data
  provenance, exact expert-byte preservation, and strict runtime attachment;
- a Hugging Face `AnchoredMoEForCausalLM` facade with base-checkpoint loading,
  exact native forward and generation delegation, backend-driven conversion,
  and an explicit install/upgrade/rollback/synchronize/unload lifecycle;
- immutable local `save_pretrained()`/`from_pretrained()` composition
  checkpoints that reference the pinned anchor, bundle byte-identical active
  packages, reject stale runtime state, and restore exact routed behavior;
- executable offline notebooks for base-model inspection, exact
  dense-equivalence verification, forced-path expert training/export, custom
  package composition, top-1/top-2 quality/efficiency evaluation, and an exact
  export/reload/remove/reinsert/upgrade/rollback lifecycle.

Automatic model mutation during installation, capacity estimation, fallback
beyond a selected pair, online replay-teacher execution,
local mixtures larger than two experts, router-only mixture refinement,
model-integrated and
package-driven top-2 execution, persistent registry storage, remote composition
checkpoint transfer, and anchor-weight bundling are intentionally not
implemented yet. Learned
mixture-head package binding and model-integrated oracle weight sweeps also
remain explicit future work. Composition
checkpoints retain the active registry snapshot but not inactive historical
slots or rollback records. Pair-aware top-2 global recalibration, embedding
registry-router histories in composition checkpoints, and automatic router
activation during registry changes remain separate tracked features.
Teacher-score distribution distillation, top-2/adaptive distillation,
cross-layer route hysteresis, arbitrary-\(k\) capacity/combination/dispatch,
and frozen-route expert training also remain separate work.

## Development setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Run the small, randomly initialized Qwen3 equivalence example without
downloading model weights:

```bash
python examples/verify_dense_equivalence.py
python examples/inspect_expert_package.py
python examples/verify_full_mlp_expert.py
python examples/verify_lora_expert.py
python examples/verify_always_on_anchor_deltas.py
python examples/verify_affine_router.py
python examples/verify_logical_router.py
python examples/verify_router_distillation.py
python examples/verify_anchored_top1.py
python examples/verify_anchor_veto_top2.py
python examples/verify_compatibility_top2.py
python examples/verify_adaptive_selection.py
python examples/verify_dynamic_k_selection.py
python examples/verify_combination_baselines.py
python examples/verify_learned_mixture_weights.py
python examples/verify_top2_dispatch.py
python examples/verify_router_feature_bank.py
python examples/verify_router_label_banks.py
python examples/verify_independent_ridge.py
python examples/verify_temperature_thresholds.py
python examples/verify_stability_margins.py
python examples/verify_pair_utilities.py
python examples/verify_top1_top2_oracle.py
python examples/evaluate_registry_strategies.py
python examples/evaluate_router_data_scaling.py
python examples/verify_routed_top1.py
python examples/verify_registry_lifecycle.py
python examples/verify_package_hub.py
python examples/verify_package_runtime.py
python examples/export_reload_lifecycle.py
python examples/verify_forced_path_training.py
python examples/verify_independent_expert_set.py
python examples/verify_local_mini_moe.py
python examples/verify_mixture_aware_refinement.py
python examples/train_expert_package.py
python examples/fit_router.py
```

The equivalent guided workflows are available in:

- [`notebooks/00_inspect_base_model.ipynb`](./notebooks/00_inspect_base_model.ipynb)
  for a resolved configuration, model inspection, and reproducibility hashes;
- [`notebooks/01_convert_dense_model.ipynb`](./notebooks/01_convert_dense_model.ipynb)
  for conversion, freezing assertions, and exact hidden-state and logit checks;
- [`notebooks/02_train_expert_package.ipynb`](./notebooks/02_train_expert_package.ipynb)
  for forced-path training, immutable-anchor checks, and safe expert-only
  package export;
- [`notebooks/03_fit_and_calibrate_router.ipynb`](./notebooks/03_fit_and_calibrate_router.ipynb)
  for deterministic ridge fitting, held-out temperature/threshold calibration,
  immutable router-package derivation, and runtime materialization;
- [`notebooks/04_compose_custom_model.ipynb`](./notebooks/04_compose_custom_model.ipynb)
  for explicit package installation, synchronization, unloading, and
  deterministic reinsertion through `AnchoredMoEForCausalLM`;
- [`notebooks/05_evaluate_registry.ipynb`](./notebooks/05_evaluate_registry.ipynb)
  for an offline top-1/top-2 comparison of counterfactual quality, latency,
  throughput, state footprint, and active-MLP work;
- [`notebooks/06_export_and_reload.ipynb`](./notebooks/06_export_and_reload.ipynb)
  for composition export/reload, exact anchor fallback after removal,
  deterministic reinsertion, and explicitly synchronized package upgrade and
  rollback.

The notebooks default to deterministic local Qwen3 smoke models and require no
network access.

## Design documents

The documentation contract tracks resolved configuration and exact fallback
behavior (CORE-001 and CORE-004), every executable notebook workflow (NB-001,
NB-002, NB-003, NB-004, NB-005, NB-006, and NB-007), and theory-README
coverage (DOC-002). DOC-003 keeps those feature IDs, public configuration
names, implementation modules, equations, citations, and tracker statuses
synchronized through automated structural checks plus explicit semantic
review.

- [Architecture blueprint](./anchored-plugin-moe-architecture.md)
- [Feature implementation plan](./feature-implementation-plan.md)
- [Source documentation standard](./docs/source-documentation-standard.md)
- [Theory README template](./docs/theory-readme-template.md)

Generated model weights and datasets must not be committed to the repository.
