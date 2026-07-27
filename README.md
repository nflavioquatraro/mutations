# mutations
# Anchored Plug-in MoE

Anchored Plug-in MoE is a research framework for installing independently
trained specialization packages into an immutable Hugging Face causal language
model. The dense model remains an exact fallback while expert representation,
routing, combination, calibration, and package lifecycle remain independently
configurable.

The current implementation covers the first dense-equivalence foundation:

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
- anchored hard top-1 selection with fixed-zero anchor evidence and stable
  anchor-first tie-breaking;
- versioned router feature caches with deterministic row alignment, provenance,
  strict safetensors validation, and embedded tensor integrity digests;
- feature-bank-bound domain and individual-utility label datasets with
  deterministic class and expert ordering;
- deterministic independent ridge fitting from utility targets into frozen
  affine router heads;
- immutable per-head temperature scaling and held-out threshold selection
  under an explicit false-activation budget;
- opt-in sparse anchored top-1 execution for configured routed MLPs;
- deterministic overflow-to-anchor fallback with no token dropping;
- bounded privacy-safe route telemetry and exact route-agreement metrics;
- strictly validated local expert-package descriptors with atomic in-memory
  install, unload, stable ordering, compatibility binding, and registry hashes;
- explicit versioned `(layer, expert_id)` payload bindings and safe local
  package materialization into converted anchored top-1 layers;
- explicit registry-to-model synchronization with exact anchor restoration on
  unload and deterministic routed behavior after reinsertion;
- forced-path single-expert training with complete converted-layer coverage,
  mechanical backbone freezing, exact trainable-parameter assertions, and a
  constrained Hugging Face Trainer/Accelerate integration;
- optional token-masked replay KL against precomputed immutable-anchor logits
  and normalized expert-initialization weight-drift regularization;
- staged expert-only package export with deterministic payload bindings,
  derived metadata, strict reload, and failure cleanup;
- immutable calibrated-router package derivation with disjoint data
  provenance, exact expert-byte preservation, and strict runtime attachment;
- a Hugging Face `AnchoredMoEForCausalLM` facade with base-checkpoint loading,
  exact native forward and generation delegation, backend-driven conversion,
  and an explicit install/synchronize/unload lifecycle;
- executable offline notebooks for base-model inspection, exact
  dense-equivalence verification, forced-path expert training/export, and
  custom package composition.

Automatic model mutation during installation, online replay-teacher execution,
multi-expert training, top-2 execution, persistent registry storage, Hub
transfer, and full-model serialization are intentionally not implemented yet.

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
python examples/verify_anchored_top1.py
python examples/verify_router_feature_bank.py
python examples/verify_router_label_banks.py
python examples/verify_independent_ridge.py
python examples/verify_temperature_thresholds.py
python examples/verify_routed_top1.py
python examples/verify_registry_lifecycle.py
python examples/verify_package_runtime.py
python examples/verify_forced_path_training.py
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
  deterministic reinsertion through `AnchoredMoEForCausalLM`.

The notebooks default to deterministic local Qwen3 smoke models and require no
network access.

## Design documents

- [Architecture blueprint](./anchored-plugin-moe-architecture.md)
- [Feature implementation plan](./feature-implementation-plan.md)
- [Source documentation standard](./docs/source-documentation-standard.md)
- [Theory README template](./docs/theory-readme-template.md)

Generated model weights and datasets must not be committed to the repository.
