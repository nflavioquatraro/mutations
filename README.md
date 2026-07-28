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
- frozen identity router features over pre-MLP hidden states;
- independent append-only affine router heads with calibrated scores and
  strict, deterministic safetensors serialization;
- a frozen registry-level logical router that compiles independent heads into
  layer-local append-only matrices while preserving incumbent scores exactly;
- anchored hard top-1 selection with fixed-zero anchor evidence and stable
  anchor-first tie-breaking;
- deterministic anchor-veto top-2 selection with temperature-scaled
  score-softmax, equal-weight, and normalized-positive-evidence activation
  combination policies;
- adaptive anchor/top-1/top-2 selection with explicit auditable route arity
  and deterministic mixture-threshold behavior;
- immutable package-declared expert compatibility groups and deterministic
  global top-2 selection that never mixes unrelated or undeclared experts;
- standalone sparse anchor-veto or compatibility-constrained top-2 dispatch
  with ranked specialist execution, weighted activation reconstruction, and
  auditable mixture weights;
- versioned router feature caches with deterministic row alignment, provenance,
  strict safetensors validation, and embedded tensor integrity digests;
- feature-bank-bound domain and individual-utility label datasets with
  deterministic class and expert ordering;
- deterministic independent ridge fitting from utility targets into frozen
  affine router heads;
- immutable per-head temperature scaling and held-out threshold selection
  under an explicit false-activation budget;
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
- opt-in sparse anchored top-1 execution for configured routed MLPs;
- deterministic overflow-to-anchor fallback with no token dropping;
- bounded privacy-safe route telemetry and exact route-agreement metrics;
- strictly validated local expert-package descriptors with atomic in-memory
  install, unload, stable ordering, compatibility binding, and registry hashes;
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
- optional token-masked replay KL against precomputed immutable-anchor logits
  and normalized expert-initialization weight-drift regularization;
- staged expert-only package export with deterministic payload bindings,
  derived metadata, strict reload, and failure cleanup;
- immutable calibrated-router package derivation with disjoint data
  provenance, exact expert-byte preservation, and strict runtime attachment;
- a Hugging Face `AnchoredMoEForCausalLM` facade with base-checkpoint loading,
  exact native forward and generation delegation, backend-driven conversion,
  and an explicit install/upgrade/rollback/synchronize/unload lifecycle;
- executable offline notebooks for base-model inspection, exact
  dense-equivalence verification, forced-path expert training/export, and
  custom package composition and top-1/top-2 quality/efficiency evaluation.

Automatic model mutation during installation, online replay-teacher execution,
local mini-MoE and mixture-aware joint training, model-integrated and
package-driven top-2 execution, persistent registry storage, Hub transfer, and
full-model serialization are intentionally not implemented yet.

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
python examples/verify_affine_router.py
python examples/verify_logical_router.py
python examples/verify_anchored_top1.py
python examples/verify_anchor_veto_top2.py
python examples/verify_compatibility_top2.py
python examples/verify_adaptive_selection.py
python examples/verify_combination_baselines.py
python examples/verify_top2_dispatch.py
python examples/verify_router_feature_bank.py
python examples/verify_router_label_banks.py
python examples/verify_independent_ridge.py
python examples/verify_temperature_thresholds.py
python examples/verify_pair_utilities.py
python examples/verify_top1_top2_oracle.py
python examples/evaluate_registry_strategies.py
python examples/verify_routed_top1.py
python examples/verify_registry_lifecycle.py
python examples/verify_package_runtime.py
python examples/verify_forced_path_training.py
python examples/verify_independent_expert_set.py
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
  throughput, state footprint, and active-MLP work.

The notebooks default to deterministic local Qwen3 smoke models and require no
network access.

## Design documents

- [Architecture blueprint](./anchored-plugin-moe-architecture.md)
- [Feature implementation plan](./feature-implementation-plan.md)
- [Source documentation standard](./docs/source-documentation-standard.md)
- [Theory README template](./docs/theory-readme-template.md)

Generated model weights and datasets must not be committed to the repository.
