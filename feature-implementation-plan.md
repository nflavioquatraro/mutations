# Anchored Plug-in MoE — Feature Implementation Plan

## 1. Purpose

This document is the implementation tracker for the Anchored Plug-in MoE framework.

It has four roles:

1. define the intended Hugging Face–based software architecture;
2. establish stable interfaces between interchangeable strategies;
3. track feature implementation and validation;
4. keep notebooks concise, composable, and suitable for end users.

Every implementation agent should update the relevant feature row when work is completed or materially changed.

The architecture and research rationale are defined in [anchored-plugin-moe-architecture.md](./anchored-plugin-moe-architecture.md).

---

## 2. Development principles

1. **Hugging Face native.** Use familiar `transformers`, `datasets`, `accelerate`, `peft`, `safetensors`, and Hub conventions.
2. **Strategy composition.** Expert representation, training, routing, combination, calibration, and registry behavior must be independently configurable.
3. **Thin notebooks.** Notebooks demonstrate public APIs; reusable logic belongs in the Python package.
4. **Immutable anchor.** The base checkpoint is never silently modified.
5. **Versioned packages.** Expert packages and registry-specific routers are independently serializable.
6. **Deterministic tests.** Routing and installation behavior use stable seeds and tie-breaking.
7. **Model-family abstraction.** Qwen3 is the first backend, not a permanent hard-coded dependency.
8. **No hidden recalibration.** Installing a package must not retrain or alter routing unless explicitly requested.
9. **Safe serialization.** Prefer `safetensors`; avoid pickle-based package formats.
10. **Tracker discipline.** A feature is not complete until its acceptance tests pass and this document is updated.
11. **Commented implementations.** Research code must explain its mathematical intent, invariants, and non-obvious decisions.
12. **Traceable references.** Implementations derived from papers must cite the paper in both source documentation and the relevant theory README.
13. **Local theory documentation.** Every feature subpackage must contain a `README.md` explaining its strategies, mathematics, assumptions, and references.

---

## 3. Proposed repository structure

This is the target layout. It is documented here but should be created incrementally as features are implemented.

```text
anchored-moe/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── anchored_moe/
│       ├── __init__.py
│       ├── configuration.py
│       ├── modeling.py
│       ├── auto.py
│       ├── backends/
│       │   ├── README.md
│       │   ├── base.py
│       │   └── qwen3.py
│       ├── experts/
│       │   ├── README.md
│       │   ├── base.py
│       │   ├── full_mlp.py
│       │   ├── low_rank.py
│       │   ├── micro.py
│       │   └── package.py
│       ├── routers/
│       │   ├── README.md
│       │   ├── base.py
│       │   ├── heads.py
│       │   ├── hierarchical.py
│       │   ├── dynamic_k.py
│       │   └── stabilized.py
│       ├── policies/
│       │   ├── README.md
│       │   ├── selection.py
│       │   ├── combination.py
│       │   ├── capacity.py
│       │   └── compatibility.py
│       ├── training/
│       │   ├── README.md
│       │   ├── expert_trainer.py
│       │   ├── mini_moe_trainer.py
│       │   ├── router_trainer.py
│       │   └── objectives.py
│       ├── calibration/
│       │   ├── README.md
│       │   ├── features.py
│       │   ├── labels.py
│       │   ├── ridge.py
│       │   ├── temperature.py
│       │   └── oracle.py
│       ├── registry/
│       │   ├── README.md
│       │   ├── manifest.py
│       │   ├── registry.py
│       │   ├── compatibility.py
│       │   └── lifecycle.py
│       ├── inference/
│       │   ├── README.md
│       │   ├── dispatch.py
│       │   ├── batching.py
│       │   └── telemetry.py
│       ├── evaluation/
│       │   ├── README.md
│       │   ├── equivalence.py
│       │   ├── routing.py
│       │   ├── quality.py
│       │   └── efficiency.py
│       └── utils/
│           ├── README.md
│           ├── hashing.py
│           ├── reproducibility.py
│           └── logging.py
├── configs/
│   ├── models/
│   ├── experts/
│   ├── routers/
│   ├── policies/
│   └── experiments/
├── notebooks/
│   ├── 00_inspect_base_model.ipynb
│   ├── 01_convert_dense_model.ipynb
│   ├── 02_train_expert_package.ipynb
│   ├── 03_fit_and_calibrate_router.ipynb
│   ├── 04_compose_custom_model.ipynb
│   ├── 05_evaluate_registry.ipynb
│   └── 06_export_and_reload.ipynb
├── examples/
│   ├── train_expert.py
│   ├── fit_router.py
│   ├── compose_model.py
│   └── evaluate_model.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── equivalence/
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── package-format.md
│   ├── routing-policies.md
│   └── evaluation.md
└── artifacts/
    └── README.md
```

Generated model weights and datasets should not be committed under `artifacts/`.

---

## 4. Hugging Face integration guidelines

### 4.1 Model and configuration

- Provide an `AnchoredMoEConfig` compatible with `transformers.PretrainedConfig`.
- Expose an `AnchoredMoEForCausalLM` compatible with `PreTrainedModel`.
- Preserve the base model's generation interface and `GenerationMixin` behavior.
- Support `save_pretrained()` and `from_pretrained()`.
- Register custom classes with `AutoConfig` and `AutoModelForCausalLM` when the implementation is stable.
- Store strategy names and parameters in configuration objects, not notebook code.
- Keep base-model conversion logic behind a backend adapter.

### 4.2 Training

- Use `Trainer` or a small compatible subclass for ordinary expert training.
- Use `accelerate` for device placement, mixed precision, accumulation, and distributed execution.
- Keep router feature extraction and counterfactual labeling available as standalone dataset transforms.
- Support `datasets.Dataset` and streaming datasets where possible.
- Freeze parameters mechanically and assert the trainable-parameter set before every run.

### 4.3 Parameter-efficient experts

- Use `peft` conventions for LoRA-based or other adapter experts.
- Keep full-MLP expert support independent of PEFT.
- Export adapters separately from the anchor.
- Record target modules, rank, scaling, and PEFT version in the package manifest.

### 4.4 Serialization and Hub compatibility

- Store tensors using `safetensors`.
- Store manifests and strategy configurations as JSON or YAML.
- Include base checkpoint revision and hashes.
- Make expert packages independently uploadable and downloadable.
- Keep registry configurations small enough to version in Git or on the Hub.
- Never duplicate the anchor weights inside every package unless explicitly requested.

### 4.5 Reproducibility

- Record Python, PyTorch, Transformers, PEFT, Accelerate, and kernel versions.
- Record dataset identifiers, revisions, filters, and splits.
- Save router reference-bank and threshold-selection metadata.
- Emit a resolved experiment configuration beside every checkpoint.

### 4.6 Code comments, docstrings, and references

All implementation code must be well documented.

- Public classes, functions, configuration fields, and extension points require docstrings.
- Mathematical code must define the meaning and expected shapes of symbols such as \(h\), \(g_k\), \(\alpha_k\), capacity, and routing masks.
- Comments should explain intent, invariants, numerical choices, and departures from a referenced algorithm—not restate obvious syntax.
- Non-obvious tensor reshaping, token dispatch, tie-breaking, overflow, and gradient-freezing behavior require nearby comments.
- Every paper-derived implementation must include a `References` section in its module or primary class docstring.
- References should include the paper title and a stable arXiv, DOI, or publisher URL.
- If the implementation modifies a paper's method, document the difference explicitly.
- Configuration names should map clearly to terminology used in the theoretical README and cited paper.

Example:

```python
class StableRouter(...):
    """Distilled router used after route discovery.

    The deployed router is frozen so that expert assignments do not fluctuate
    while the associated experts are trained or evaluated.

    References:
        Dai et al., "StableMoE: Stable Routing Strategy for Mixture of Experts."
        https://arxiv.org/abs/2204.08396

    Differences from the paper:
        Package heads are frozen independently instead of freezing one jointly
        trained global router.
    """
```

### 4.7 Subpackage theory README contract

Every feature-bearing subfolder under `src/anchored_moe/` must contain a `README.md`.

The README is part of the implementation. When an agent adds or changes a feature, it must update the README in the same change.

Each README should follow this structure:

```markdown
# <Subpackage name>

## Scope
What the subpackage owns and deliberately does not own.

## Implemented strategies
Feature IDs, configuration names, implementation modules, and status.

## Mathematical formulation
Definitions, equations, tensor shapes, and routing or optimization objectives.

## Algorithms
Step-by-step training and inference behavior.

## Assumptions and invariants
Frozen parameters, compatibility requirements, numerical assumptions,
fallback behavior, and complexity.

## Hugging Face integration
Relevant Transformers, PEFT, Accelerate, Datasets, or Hub interfaces.

## Configuration and examples
Minimal public configuration and API examples.

## References
Paper titles with stable arXiv, DOI, or publisher links, plus an explanation
of which implementation detail each paper motivated.

## Tests and known limitations
Acceptance tests, unsupported cases, and open research questions.
```

For infrastructure-oriented subpackages such as `utils/`, replace mathematical formulation with precise behavioral contracts where mathematics is not relevant.

The architecture blueprint can be linked for shared definitions, but local READMEs must contain enough theory to understand and review the implementation without reverse-engineering its source code.

---

## 5. Core composable interfaces

The exact class names may evolve, but the responsibilities should remain separate.

### `ModelBackend`

Responsible for:

- locating transformer layers and MLP modules;
- reading hidden and intermediate dimensions;
- replacing MLPs with routed wrappers;
- preserving model-specific forward signatures;
- validating compatibility.

### `Expert`

Responsible for:

- executing one expert transformation;
- exposing capacity and parameter metadata;
- serialization and loading;
- declaring compatible combination groups.

### `RouterHead`

Responsible for:

- producing expert scores from frozen router features;
- temperature and threshold calibration;
- append-only serialization.

### `SelectionPolicy`

Responsible for choosing:

- anchor;
- top-1;
- top-2;
- dynamic-\(k\);
- a hierarchical package and local experts.

### `CombinationPolicy`

Responsible for:

- hard expert output;
- activation-space weighted combination;
- anchor-plus-delta combination;
- parameter-space merging.

### `CapacityPolicy`

Responsible for:

- per-expert capacity;
- overflow behavior;
- top-2 to top-1 degradation;
- load-balancing biases.

### `ExpertRegistry`

Responsible for:

- installation and removal;
- active package ordering;
- registry hashes;
- router-version association;
- atomic upgrades and rollback.

### `CalibrationStrategy`

Responsible for:

- feature collection;
- target construction;
- ridge or gradient fitting;
- threshold selection;
- counterfactual oracle evaluation.

---

## 6. Intended user experience

The public API should allow configuration-driven composition:

```python
from anchored_moe import (
    AnchoredMoEForCausalLM,
    ExpertPackage,
    RegistryConfig,
)

model = AnchoredMoEForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    converted_layers="upper_half",
)

model.install(
    ExpertPackage.from_pretrained("organization/code-experts"),
)
model.install(
    ExpertPackage.from_pretrained("organization/math-experts"),
)

model.configure_registry(
    RegistryConfig(
        selection="anchor_veto_top2",
        combination="score_softmax",
        overflow="anchor",
    )
)

output = model.generate(**inputs)
model.save_pretrained("./custom-post-trained-model")
```

The final API should feel declarative: notebooks choose and compose strategies rather than implement them.

---

## 7. Feature status conventions

| Status | Meaning |
|---|---|
| Not started | No implementation has begun |
| In progress | An agent is actively implementing the feature |
| Blocked | Work cannot continue; the reason is recorded |
| Implemented | Code exists but full acceptance criteria have not passed |
| Validated | Required tests and documentation have passed |
| Deferred | Intentionally postponed |

Agents should not mark a feature `Validated` unless:

1. implementation exists behind the intended interface;
2. unit and required integration tests pass;
3. source code has the required comments and docstrings;
4. paper-derived code reports its references and any deviations;
5. the relevant subpackage `README.md` contains its theoretical and mathematical explanation;
6. public configuration is documented;
7. at least one example or notebook exercises it;
8. compatibility and serialization behavior are tested where applicable.

---

## 8. Feature implementation tracker

Update `Status`, `Last update`, and `Implementation notes` as work progresses. Add a commit or pull-request reference when available.

| ID | Priority | Area | Feature or strategy | Dependencies | Status | Last update | Implementation notes |
|---|---:|---|---|---|---|---|---|
| CORE-001 | P0 | Configuration | `AnchoredMoEConfig` and resolved strategy configuration | — | Not started | — | — |
| CORE-002 | P0 | Backend | Generic `ModelBackend` interface | CORE-001 | Not started | — | — |
| CORE-003 | P0 | Backend | Qwen3 MLP discovery and replacement backend | CORE-002 | Not started | — | — |
| CORE-004 | P0 | Modeling | Routed MLP wrapper with forced-anchor mode | CORE-003 | Not started | — | — |
| CORE-005 | P0 | Modeling | `AnchoredMoEForCausalLM` load, forward, and generation compatibility | CORE-004 | Not started | — | — |
| CORE-006 | P0 | Serialization | `save_pretrained()` and `from_pretrained()` round trip | CORE-005, PKG-001 | Not started | — | — |
| EXP-001 | P0 | Experts | Full copied-MLP expert | CORE-004 | Not started | — | — |
| EXP-002 | P1 | Experts | Anchor-relative low-rank or LoRA expert | CORE-004 | Not started | — | Prefer PEFT-compatible format |
| EXP-003 | P2 | Experts | Structured or sparse expert delta | EXP-002 | Not started | — | — |
| EXP-004 | P3 | Experts | Fine-grained micro-expert representation | EXP-001 | Not started | — | DeepSeekMoE-inspired |
| EXP-005 | P3 | Experts | Product-key expert retrieval | EXP-004, RTR-006 | Not started | — | PEER-inspired |
| TRN-001 | P0 | Training | Forced-path single-expert trainer | EXP-001 | Not started | — | — |
| TRN-002 | P1 | Training | Independent latent expert-set training | TRN-001 | Not started | — | — |
| TRN-003 | P1 | Training | Local mini-MoE training | TRN-002, POL-002 | Not started | — | — |
| TRN-004 | P2 | Training | Mixture-aware local refinement | TRN-002, COM-001 | Not started | — | — |
| TRN-005 | P1 | Training | Replay preservation and weight-drift objectives | TRN-001 | Not started | — | — |
| RTR-001 | P0 | Router | Frozen router feature extraction | CORE-004 | Not started | — | — |
| RTR-002 | P0 | Router | Independent append-only affine heads | RTR-001 | Not started | — | — |
| RTR-003 | P1 | Router | One logical router with append-only output columns | RTR-002 | Not started | — | — |
| RTR-004 | P1 | Router | Router-only global recalibration | RTR-003, CAL-002 | Not started | — | Registry-versioned |
| RTR-005 | P2 | Router | Hierarchical package and local expert routing | RTR-003 | Not started | — | — |
| RTR-006 | P2 | Router | Prompt-level candidate shortlisting | PKG-002 | Not started | — | — |
| RTR-007 | P1 | Router | Router distillation and frozen deployment router | RTR-003, CAL-004 | Not started | — | StableMoE-inspired |
| RTR-008 | P2 | Router | Route hysteresis or cross-layer stickiness | RTR-003 | Not started | — | Experimental |
| POL-001 | P0 | Selection | Anchored hard top-1 | RTR-002 | Not started | — | — |
| POL-002 | P0 | Selection | Anchor-veto weighted top-2 | RTR-002, COM-001 | Not started | — | — |
| POL-003 | P1 | Selection | Adaptive anchor/top-1/top-2 | POL-001, POL-002 | Not started | — | — |
| POL-004 | P2 | Selection | Hierarchical package top-1 and local top-2 | RTR-005, POL-002 | Not started | — | — |
| POL-005 | P2 | Selection | Global compatibility-constrained top-2 | PKG-003, POL-002 | Not started | — | — |
| POL-006 | P2 | Selection | Dynamic-\(k\) selection | POL-003 | Not started | — | Confidence and budget variants |
| COM-001 | P0 | Combination | Score-softmax activation combination | POL-002 | Not started | — | — |
| COM-002 | P1 | Combination | Equal and normalized-positive-evidence baselines | COM-001 | Not started | — | — |
| COM-003 | P1 | Combination | Learned mixture-weight head | COM-001, CAL-003 | Not started | — | Separate selection and mixing scores |
| COM-004 | P2 | Combination | Always-on anchor plus routed deltas | EXP-002, POL-003 | Not started | — | Shared-anchor strategy |
| COM-005 | P3 | Combination | Parameter-space expert merging | EXP-002, COM-003 | Not started | — | SMEAR/X-LoRA-inspired |
| CAL-001 | P0 | Calibration | Router feature cache and reference-bank format | RTR-001 | Not started | — | — |
| CAL-002 | P0 | Calibration | Domain-label and utility-label datasets | CAL-001 | Not started | — | — |
| CAL-003 | P1 | Calibration | Pair-utility and best-pair sampling | CAL-002, COM-001 | Not started | — | Avoid exhaustive \(O(K^2)\) by default |
| CAL-004 | P0 | Calibration | Independent ridge fitting | CAL-002, RTR-002 | Not started | — | — |
| CAL-005 | P0 | Calibration | Temperature and threshold fitting | CAL-004 | Not started | — | — |
| CAL-006 | P1 | Calibration | Stability-margin refinement | CAL-005 | Not started | — | — |
| CAL-007 | P1 | Calibration | Counterfactual route oracle and router regret | CAL-002 | Not started | — | — |
| BAL-001 | P0 | Capacity | Overflow-to-anchor policy | CORE-004 | Not started | — | No token dropping |
| BAL-002 | P1 | Capacity | Capacity-aware top-2 to top-1 degradation | BAL-001, POL-003 | Not started | — | — |
| BAL-003 | P2 | Balancing | Auxiliary load-balancing objective | TRN-003 | Not started | — | — |
| BAL-004 | P2 | Balancing | Bias-based auxiliary-loss-free balancing | TRN-003 | Not started | — | DeepSeek-V3-inspired |
| BAL-005 | P3 | Balancing | Expert-choice routing research backend | TRN-003 | Not started | — | Training/prefill first |
| PKG-001 | P0 | Packaging | Expert manifest schema and hash validation | CORE-001 | Not started | — | — |
| PKG-002 | P0 | Registry | Install, unload, and active-registry hash | PKG-001 | Not started | — | — |
| PKG-003 | P1 | Registry | Compatibility groups for expert mixtures | PKG-001 | Not started | — | — |
| PKG-004 | P1 | Registry | Atomic package upgrades and rollback | PKG-002 | Not started | — | — |
| PKG-005 | P1 | Registry | Registry-specific router versions | PKG-002, RTR-004 | Not started | — | — |
| PKG-006 | P2 | Hub | Expert package Hub upload and download | PKG-001 | Not started | — | — |
| EVAL-001 | P0 | Evaluation | Forced-anchor hidden-state and logit equivalence | CORE-004 | Not started | — | — |
| EVAL-002 | P0 | Evaluation | Route logging and exact route agreement | POL-001 | Not started | — | No sensitive content |
| EVAL-003 | P0 | Evaluation | Installation, removal, and reinsertion tests | PKG-002, EVAL-002 | Not started | — | — |
| EVAL-004 | P1 | Evaluation | Top-1/top-2 oracle comparison and regret | CAL-007, POL-002 | Not started | — | — |
| EVAL-005 | P1 | Evaluation | Latency, throughput, memory, and active-MLP metrics | CORE-005 | Not started | — | — |
| EVAL-006 | P1 | Evaluation | Installation-order and registry-subset evaluation | PKG-002 | Not started | — | — |
| EVAL-007 | P2 | Evaluation | Router-data scaling experiment versus partial MoE | RTR-004, TRN-003 | Not started | — | Core research hypothesis |
| NB-001 | P0 | Notebooks | Inspect and pin base model | CORE-003 | Not started | — | `00_inspect_base_model.ipynb` |
| NB-002 | P0 | Notebooks | Convert model and verify anchor equivalence | EVAL-001 | Not started | — | `01_convert_dense_model.ipynb` |
| NB-003 | P0 | Notebooks | Train and export an expert package | TRN-001, PKG-001 | Not started | — | `02_train_expert_package.ipynb` |
| NB-004 | P0 | Notebooks | Fit and calibrate router | CAL-005 | Not started | — | `03_fit_and_calibrate_router.ipynb` |
| NB-005 | P0 | Notebooks | Compose a custom model from packages | PKG-002, POL-001 | Not started | — | `04_compose_custom_model.ipynb` |
| NB-006 | P1 | Notebooks | Evaluate registry strategies | EVAL-004, EVAL-005 | Not started | — | `05_evaluate_registry.ipynb` |
| NB-007 | P1 | Notebooks | Export, reload, remove, and upgrade packages | CORE-006, PKG-004 | Not started | — | `06_export_and_reload.ipynb` |
| DOC-001 | P0 | Documentation | Source comment, docstring, and paper-reference standard | — | Not started | — | Required for every implemented feature |
| DOC-002 | P0 | Documentation | Theory README template and README in every source subpackage | DOC-001 | Not started | — | Updated in the same change as feature code |
| DOC-003 | P1 | Documentation | Synchronize feature IDs, configuration names, code, equations, and citations | DOC-002 | Not started | — | Include in documentation checks |

---

## 9. Notebook guidelines

Each notebook should:

- begin with one resolved experiment configuration;
- call public package APIs rather than private helpers;
- contain no reusable class definitions;
- avoid hard-coded local paths;
- support a small smoke-test mode;
- save artifacts through `save_pretrained()` or package APIs;
- display trainable parameter counts and frozen-parameter assertions;
- finish with a concise validation summary;
- link to the corresponding feature IDs in this tracker.

Notebook cells should follow a predictable sequence:

```text
Install/import
    ↓
Resolve configuration
    ↓
Load data and model
    ↓
Run framework operation
    ↓
Validate invariants
    ↓
Save or publish artifact
```

The examples under `examples/` should provide non-interactive equivalents for automation and reproducibility.

---

## 10. Configuration composition

Experiment configurations should compose independent strategy blocks:

```yaml
base_model:
  model_id: Qwen/Qwen3-4B
  revision: pinned-revision
  converted_layers:
    strategy: upper_half

expert:
  representation: full_mlp
  experts_per_package: 2

training:
  strategy: independent
  preservation_objective: kl

router:
  architecture: append_only_heads
  fitting: utility_ridge_margin

selection:
  policy: anchor_veto_top2
  anchor_score: 0.0

combination:
  policy: score_softmax
  temperature: 1.0

capacity:
  overflow: anchor

registry:
  recalibration: none
```

Changing one block should not require modifying another implementation module.

---

## 11. Milestones

### M0 — Project foundation

Complete:

- CORE-001 through CORE-004;
- PKG-001;
- EVAL-001;
- DOC-001 and DOC-002;
- NB-001 and NB-002.

Exit criterion: an empty registry reproduces the dense model within declared tolerance.

### M1 — Modular top-1 baseline

Complete:

- EXP-001;
- TRN-001 and TRN-005;
- RTR-001 and RTR-002;
- POL-001;
- CAL-001, CAL-002, CAL-004, CAL-005;
- BAL-001;
- PKG-002;
- NB-003 through NB-005.

Exit criterion: one expert package can be trained, installed, selected, removed, and reinserted.

### M2 — Weighted top-2

Complete:

- TRN-002;
- POL-002 and POL-003;
- COM-001 and COM-002;
- CAL-003 and CAL-007;
- EVAL-004.

Exit criterion: top-1 and top-2 strategies can be compared under the same registry and compute metrics.

### M3 — Router-only composition

Complete:

- RTR-003 and RTR-004;
- CAL-006;
- PKG-005;
- EVAL-006 and EVAL-007.

Exit criterion: small router replay subsets can be compared with strict append-only routing and a matched partial MoE.

### M4 — Advanced research strategies

Prioritize:

- RTR-007;
- POL-006;
- COM-004;
- EXP-002;
- COM-005;
- EXP-004 and EXP-005;
- BAL-004.

Exit criterion: advanced strategies use the same public composition and evaluation interfaces as M1–M3.

---

## 12. Agent update protocol

When implementing a tracked feature, an agent should:

1. change its row to `In progress`;
2. record the date and implementation branch or task reference;
3. implement only through the declared interfaces;
4. add explanatory comments and public docstrings;
5. cite the motivating papers in code and document deviations;
6. update the relevant subpackage theory `README.md`, including equations;
7. add or update tests;
8. update relevant notebook usage;
9. change the row to `Implemented` or `Validated`;
10. summarize deviations or follow-up work in `Implementation notes`.

If one implementation completes several rows, update each row separately.

If a design changes:

- preserve the stable feature ID;
- update the feature description and dependencies;
- explain the migration in `Implementation notes`;
- do not silently remove historical work.

Suggested note format:

```text
YYYY-MM-DD — Validated in <commit/PR/task>.
Tests: <test names>. Config: <public option>.
Theory: <subpackage README section>. References: <paper links>.
Follow-up: <remaining limitation or “none”>.
```

---

## 13. Definition of framework success

The framework is successful when a user can:

1. load a supported Hugging Face causal language model;
2. convert selected MLP layers without changing dense behavior;
3. train or load independently versioned expert packages;
4. choose routing, combination, calibration, and capacity strategies declaratively;
5. compose packages into a custom post-trained model;
6. evaluate quality, route stability, and cost;
7. save, reload, remove, upgrade, and share packages;
8. reproduce the complete workflow from concise notebooks.

The implementation tracker is the operational record of progress toward that outcome.
