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
- executable offline notebooks for base-model inspection and exact
  dense-equivalence verification.

Routed expert attachment and dispatch, model-integrated selection, router
fitting and calibration, package installation, training, and full-model
serialization are intentionally not implemented yet.

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
```

The equivalent guided workflows are available in:

- [`notebooks/00_inspect_base_model.ipynb`](./notebooks/00_inspect_base_model.ipynb)
  for a resolved configuration, model inspection, and reproducibility hashes;
- [`notebooks/01_convert_dense_model.ipynb`](./notebooks/01_convert_dense_model.ipynb)
  for conversion, freezing assertions, and exact hidden-state and logit checks.

Both notebooks default to deterministic local Qwen3 smoke models and require no
network access.

## Design documents

- [Architecture blueprint](./anchored-plugin-moe-architecture.md)
- [Feature implementation plan](./feature-implementation-plan.md)
- [Source documentation standard](./docs/source-documentation-standard.md)
- [Theory README template](./docs/theory-readme-template.md)

Generated model weights and datasets must not be committed to the repository.
