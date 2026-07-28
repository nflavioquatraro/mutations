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
| CORE-001 | P0 | Configuration | `AnchoredMoEConfig` and resolved strategy configuration | — | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: `test_configuration.py`. Config: `AnchoredMoEConfig`, `strategy_config`, `upper_half` or explicit layers. Theory: root README and source documentation standard. References: Hugging Face `PretrainedConfig`. Follow-up: add strategies only with their runtimes. |
| CORE-002 | P0 | Backend | Generic `ModelBackend` interface | CORE-001 | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: backend and documentation unit tests. Config: `backend`. Theory: `backends/README.md`. References: no paper-derived algorithm. Follow-up: add model families behind the same interface. |
| CORE-003 | P0 | Backend | Qwen3 MLP discovery and replacement backend | CORE-002 | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: Qwen3 backend unit tests and tiny-model integration test. Config: `backend: qwen3`. Theory: `backends/README.md`. References: Hugging Face Qwen3 documentation. Follow-up: validate optimized runtime variants separately. |
| CORE-004 | P0 | Modeling | Routed MLP wrapper with forced-anchor mode | CORE-003 | Validated | 2026-07-27 | 2026-07-25 — Validated in Codex task. Tests: exact hidden-state/logit equivalence and anchor fingerprint. Config: `selection.policy: forced_anchor`. Theory: backend and evaluation READMEs. References: no paper-derived algorithm. 2026-07-27 — The exact default path remains unchanged; INF-001 adds explicit opt-in expert dispatch without weakening forced-anchor equivalence. |
| CORE-005 | P0 | Modeling | `AnchoredMoEForCausalLM` load, forward, and generation compatibility | CORE-004 | Validated | 2026-07-28 | 2026-07-27 — Validated in current Codex task. Tests: `test_anchored_causal_lm.py`, documentation contract, executable composition example, all examples, notebook execution, wheel-content and dependency checks, and full regression suite (208 passed). Config/API: `AnchoredMoEForCausalLM.from_pretrained()` loads a native Hugging Face base checkpoint; `from_anchor_model()` supports already loaded anchors; backend conversion, `forward()`, and `generate()` preserve the model-family implementation; `install()`/`unload()` are registry-only and `synchronize_registry()` is explicit. Invariants: exact CPU logits and every hidden state, exact greedy generation, complete anchor freezing, live configuration-hash validation before conversion, and no hidden synchronization. Theory: root, backend, inference, and registry READMEs. References: Hugging Face `PreTrainedModel`, `GenerationMixin`, and `AutoModelForCausalLM`; no paper-derived algorithm. 2026-07-28 — CORE-006 now extends the same loader with local composition detection while ordinary base-checkpoint loading remains unchanged. Follow-up: production remote-checkpoint and multi-device validation remain separate work. |
| CORE-006 | P0 | Serialization | `save_pretrained()` and `from_pretrained()` round trip | CORE-005, PKG-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_composition_checkpoint.py`, CORE-006 cases in `test_anchored_causal_lm.py` and `test_package_runtime_composition.py`, documentation contract, updated executable `verify_package_runtime.py`, all 26 examples, all six notebooks, wheel-content and dependency checks, and full regression suite (401 passed). Config/API: `composition_format_version: 1`; `AnchoredMoEForCausalLM.save_pretrained()` publishes an immutable local composition and `from_pretrained()` detects and restores it; `anchor_model_name_or_path` can relocate an equivalent local anchor. Runtime: the checkpoint references rather than copies the anchor, bundles strictly validated byte-identical active packages in registry order, saves anchored/generation configuration hashes, reconstructs the active registry snapshot, and explicitly restores anchored-top-1 routing. Invariants: stale registry/runtime state, existing destinations, symbolic links, untracked or modified files, reordered identities, unsafe formats, revision conflicts, and live anchor configuration drift are rejected; ordinary install/unload/upgrade/rollback remain unsynchronized. Theory: registry and inference READMEs plus the root README. References: Hugging Face configuration/generation APIs, safetensors package contracts, SHA-256, and the architecture blueprint; no paper-derived algorithm. Follow-up: inactive historical slots and rollback records, exact live anchor-weight hashing, optional anchor bundling, Hub transfer, multi-device/distributed validation, and NB-007 remain separate work. |
| EXP-001 | P0 | Experts | Full copied-MLP expert | CORE-004 | Validated | 2026-07-27 | 2026-07-25 — Validated in Codex task. Tests: `test_full_mlp_expert.py`, `test_full_mlp_package_payload.py`, documentation contracts, executable example, and full regression suite (55 passed). Config: `expert_parameterization: full_mlp`, `initialization: anchor_copy`, and explicit trainability. Theory: `experts/README.md` with equations, storage and gradient invariants, algorithms, and limitations. References: PyTorch `nn.Module`, safetensors, and the architecture blueprint; no paper-derived algorithm. 2026-07-27 — INF-001 can attach expert objects explicitly. Follow-up: package-driven attachment, forced-path training, multi-layer package layout, and installation remain separate tracked features. |
| EXP-002 | P1 | Experts | Anchor-relative low-rank or LoRA expert | CORE-004 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_low_rank_expert.py`, `test_lora_forced_path_training.py`, documentation contract, executable `verify_lora_expert.py`, all 32 examples, dependency check, wheel-content inspection, and full regression suite (465 passed). Config/API: `expert_parameterization: peft_lora`, `LoRAExpertConfig`, `PEFTLoRAExpert`, and `Expert.set_trainable()` with `peft>=0.18.1,<1`. Runtime: PEFT injects ordinary or rank-stabilized LoRA only into a private frozen anchor copy; the live anchor is frozen after failure-atomic construction, complete MLP output preserves existing dispatch, and only adapter tensors enter gradients, drift snapshots, optimizers, and deterministic safetensors payloads. Invariants: exact zero-update Qwen3 output, disjoint private storage, immutable live/private bases during training, stable target ordering, strict adapter key/shape/dtype reload, recorded PEFT version, and no anchor serialization. Theory: `experts/README.md` plus training and root integration notes. References: Hu et al. LoRA and Hugging Face PEFT; the private-copy complete-output difference is documented. Follow-up: PEFT/LoRA package export and PKG-007 materialization, shared-anchor delta execution (COM-004), mixed parameterizations, alternative initialization/adapter methods, and multi-device/distributed validation remain separate work. |
| EXP-003 | P2 | Experts | Structured or sparse expert delta | EXP-002 | Not started | — | — |
| EXP-004 | P3 | Experts | Fine-grained micro-expert representation | EXP-001 | Not started | — | DeepSeekMoE-inspired |
| EXP-005 | P3 | Experts | Product-key expert retrieval | EXP-004, RTR-006 | Not started | — | PEER-inspired |
| TRN-001 | P0 | Training | Forced-path single-expert trainer | EXP-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_forced_path_training.py`, forced-expert routed-wrapper coverage, `test_forced_path_trainer.py`, documentation contract, executable training example, all examples, wheel-content and dependency checks, and full regression suite (201 passed). Config/API: `ForcedExpertBinding`, `prepare_forced_path_training()`, and `ForcedPathExpertTrainer` with `TrainingArguments(save_strategy: no)`; `accelerate>=1.1` is a runtime dependency. Runtime: every token uses the new expert at every converted layer; the complete incumbent model is mechanically frozen; expert/model and cross-layer parameter/storage aliasing are rejected; exact trainable identities and forced routes are reasserted before every training run. Full-model saving, Hub push, model reinitialization, and checkpoint resume are rejected so expert/package serialization stays explicit. Theory: `training/README.md`, plus expert and inference README integration notes. References: Hugging Face Trainer, Accelerate, PyTorch gradient contracts, and the architecture blueprint; no paper-derived algorithm. 2026-07-27 — TRN-005 preservation and drift objectives are now validated without weakening TRN-001 isolation. Follow-up: NB-003 package export, mixed-precision/multi-device/distributed validation, and expert-only optimizer checkpointing remain separate work. |
| TRN-002 | P1 | Training | Independent latent expert-set training | TRN-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_independent_expert_training.py`, `test_independent_expert_set_trainer.py`, documentation contract, executable `verify_independent_expert_set.py`, all 24 examples, all six notebooks, deterministic wheel-content check, and full regression suite (337 passed). Config/API: `IndependentExpertTrainingRun`, `IndependentExpertTrainingReport`, `IndependentExpertSetTrainingReport`, and `train_independent_expert_set()` compose at least two complete TRN-001 paths with per-expert `TrainingArguments`, Trainer inputs, dataset partitions, seeds, and optional TRN-005 objectives. Runtime: the entire layer-by-expert rectangle, unique IDs, no-checkpoint boundary, and cross-model/cross-expert parameter and storage independence are validated before the first Trainer; one path trains at a time in declared order; every success or failure clears gradients, freezes all experts, and restores exact forced-anchor routing. Invariants: every expert covers identical converted layers; anchor state remains byte-exact; reports are frozen and ordered; cleanup does not promise transactional rollback of partially optimized expert weights. Theory: `training/README.md` with independent objectives, parameter-set equations, algorithms, Hugging Face boundary, assumptions, and limitations. References: Hugging Face Trainer/Accelerate and the architecture blueprint; orchestration is framework infrastructure, not a paper-derived algorithm. Follow-up: automatic data partitioning, concurrent process orchestration, diversity objectives, local mini-MoE training, expert-only checkpoint resume, mixed precision, multi-device, and distributed validation remain separate work. |
| TRN-003 | P1 | Training | Local mini-MoE training | TRN-002, POL-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_local_mini_moe_training.py`, TRN-003 coverage in `test_routed_mlp.py`, `test_local_mini_moe_trainer.py`, documentation contract, executable `verify_local_mini_moe.py`, all 33 examples, all seven notebooks, wheel-content/import checks, and full regression suite (471 passed). Config/API: `LocalMiniMoETrainingConfig`, `LocalMiniMoEExpertBinding`, `LocalMiniMoETrainingTopology`, `LocalMiniMoETrainingReport`, `prepare_local_mini_moe_training()`, and one-shot `LocalMiniMoETrainer` with strategy `local_mini_moe_top2`. Runtime: exactly two package-local experts execute every specialization token at every converted layer; fresh affine heads receive deterministic opposite biases, COM-001 score-softmax combines both outputs, and ordinary Hugging Face Trainer jointly optimizes only expert-declared parameters and local heads. Invariants: exact rectangular layer/expert ownership, zero uncalibrated router initialization, model/expert/router/feature parameter and storage isolation, frozen anchor/backbone/features/incumbents, exact optimizer identity, immutable reports, and success/failure-safe gradient clearing, artifact freezing, and forced-anchor restoration. Theory: `training/README.md`, with COM-001 integration in `policies/README.md` and NB-003 guidance. References: Hugging Face Trainer/Accelerate and the architecture blueprint; the two-expert fixed-pair topology and deterministic symmetry breaking are project-specific framework choices, not paper-derived algorithms. Follow-up: more than two experts, discrete local top-k, diversity objectives, TRN-005 integration, package-local router bundling, transactional rollback, checkpoint resume, mixed precision, multi-device, and distributed validation remain separate work. 2026-07-28 — BAL-003 adds an explicitly optional training loss and final objective report while preserving the zero-weight TRN-003 execution and optimizer boundary. 2026-07-28 — TRN-004 now composes completed independent training with this unchanged local runtime. |
| TRN-004 | P2 | Training | Mixture-aware local refinement | TRN-002, COM-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_mixture_aware_refinement.py`, `test_mixture_aware_refinement_pipeline.py`, documentation contract, executable `verify_mixture_aware_refinement.py`, all 35 examples, all seven notebooks through the full suite, deterministic wheel-content inspection, and full regression suite (501 passed). Config/API: `refine_independent_expert_set()`, `MixtureAwareRefinementReport`, and `MixtureAwareRefinementError` reuse `LocalMiniMoETrainingConfig` and ordinary Hugging Face `TrainingArguments`/Trainer inputs. Runtime: exactly one completed two-expert TRN-002 report is matched to caller-owned experts by stable source order and the complete converted-layer rectangle before the existing TRN-003/COM-001 path jointly updates both expert-declared optimizer subsets and fresh affine heads. Invariants: malformed provenance and Trainer keywords fail before route attachment; delegated TRN-003 retains exact parameter/storage isolation, byte-exact anchor state, optional BAL-003 composition, immutable linked reports, gradient cleanup, artifact freezing, and forced-anchor restoration on success or failure. Theory: `training/README.md` with the sequential optimization objective, algorithms, provenance boundary, Hugging Face integration, configuration, and limitations. References: Hugging Face Trainer/Accelerate and the architecture blueprint; TRN-004 is framework orchestration, not a paper-derived algorithm. Follow-up: more than two experts, router-only refinement, previously trained router heads, cryptographic expert-artifact provenance, TRN-005 composition, transactional rollback, checkpoint resume, mixed precision, multi-device, and distributed validation remain separate work. |
| TRN-005 | P1 | Training | Replay preservation and weight-drift objectives | TRN-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_expert_objectives.py`, TRN-005 cases in `test_forced_path_trainer.py`, documentation contract, executable training example, all examples, wheel-content and dependency checks, and full regression suite (217 passed). Config/API: `ExpertObjectiveConfig` independently controls `replay_kl_weight`, `drift_weight`, `replay_temperature`, and stable teacher/mask batch keys; `ReplayKLPreservation`, `ExpertWeightDrift`, and `ExpertObjectiveReport` are public. Runtime: `ForcedPathExpertTrainer.compute_loss()` preserves the standard specialization loss, retains configured auxiliary dataset columns, detaches precomputed teacher logits, supports non-negative token masks and zero-replay batches, snapshots expert initialization outside model state, normalizes drift by scalar parameter count, and leaves the exact expert-only optimizer identity unchanged. Theory: `training/README.md` with equations, algorithms, storage costs, assumptions, and limitations. References: Hinton et al. knowledge distillation and Li et al. L2-SP, with immutable-anchor, token-mask, expert-only, and normalization differences documented. Follow-up: NB-003 teacher-logit provenance/export, compact teacher targets, online teachers, mixed precision, multi-device, and distributed validation remain separate work. |
| RTR-001 | P0 | Router | Frozen router feature extraction | CORE-004 | Validated | 2026-07-27 | 2026-07-27 — Validated in Codex task. Tests: `test_router_features.py`, router documentation contract, executable example, and full regression suite (77 passed). Config: `feature_type: identity`, `input: pre_mlp_hidden_state`, and explicit `output_size`. Theory: `routers/README.md` with feature equations, tensor shapes, gradient behavior, and invariants. References: PyTorch modules and the architecture blueprint; no paper-derived algorithm. Follow-up: learned frozen projections and CAL-001 feature caching remain separate features. |
| RTR-002 | P0 | Router | Independent append-only affine heads | RTR-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in Codex task. Tests: `test_router_heads.py`, `test_affine_router_package_payload.py`, documentation contract, executable example, deterministic wheel-content check, and full regression suite (77 passed). Config: `router_architecture: independent_affine_heads`, per-head temperature and threshold, and explicit trainability. Theory: `routers/README.md` with calibrated-score equations, append-order semantics, serialization, and limitations. References: PyTorch `nn.Linear`, safetensors, and the architecture blueprint; no paper-derived algorithm. POL-001, CAL-004, and PKG-002 are now validated. Follow-up: PKG-007 package payload binding and attachment remains separate. |
| RTR-003 | P1 | Router | One logical router with append-only output columns | RTR-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_logical_router.py`, `test_logical_router_routed_mlp.py`, package-runtime composition integration, documentation contract, executable `verify_logical_router.py`, all 25 examples, all six notebooks, wheel-content and dependency checks, and full regression suite (344 passed). Config/API: `RouterScoreBank`, `LogicalAppendOnlyRouter`, `compile_logical_router()`, and runtime architecture `logical_append_only_columns`; package manifests continue to use `independent_affine_heads`. Runtime: package-owned RTR-002 heads compile into copied layer-local weight, bias, temperature, and threshold buffers; registry synchronization installs the logical router through the shared score-bank interface. Invariants: no source-storage aliases or trainable parameters; exact score parity, stable global and layer-local ordering, preserved feature gradients, and bit-exact incumbent scores after append through shape-stable column evaluation. Theory: `routers/README.md` plus inference and root integration notes. References: PyTorch affine operations and module buffers plus the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: fused matrix scoring is deferred unless it can retain the exact append invariant; RTR-004 global recalibration and PKG-005 registry-router persistence remain separate work. |
| RTR-004 | P1 | Router | Router-only global recalibration | RTR-003, CAL-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_global_router_recalibration.py`, `test_registry_router_recalibration.py`, logical-router and documentation contracts, executable `verify_registry_router_versions.py`, all 28 examples, all seven notebooks, deterministic wheel-content check, and full regression suite (409 passed). Config/API: `recalibrate_global_router()`, `GlobalRouterRecalibrationReport`, and `LayerGlobalRouterRecalibrationReport` with explicit registry/compatibility hashes, fixed learning rate and step count, and normalized proximal drift. Runtime: every layer derives stable anchor-first best-route classes from anchor-relative individual utilities and jointly optimizes all affine weights/biases through seed-free full-batch CPU float64 Adam; temperature and threshold stay fixed; the best source/iterate is accepted only when deployed-dtype total loss does not worsen. Invariants: exact feature/layer/expert/dtype alignment, source and bank immutability, storage-independent frozen logical output, no model/expert mutation, and no activation side effect. Theory: calibration and router READMEs plus root integration notes. References: architecture blueprint router-only recalibration mode; the anchored classification target, proximal drift, deterministic optimizer, and deployed acceptance gate are documented project-specific choices, not paper-derived. Follow-up: domain-label objectives, arbitrary-incumbent targets, pair-aware top-2 objectives, automatic replay partitions, accelerator/distributed fitting, and confidence calibration remain separate work. |
| RTR-005 | P2 | Router | Hierarchical package and local expert routing | RTR-003 | Not started | — | — |
| RTR-006 | P2 | Router | Prompt-level candidate shortlisting | PKG-002 | Not started | — | — |
| RTR-007 | P1 | Router | Router distillation and frozen deployment router | RTR-003, CAL-004 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_router_distillation.py`, documentation contract, executable `verify_router_distillation.py`, all 29 examples, deterministic wheel-content check, and full regression suite (414 passed). Config/API: `HardRouteDistillationConfig`, `LayerRouterDistillationReport`, `RouterDistillationReport`, and `distill_frozen_router()` with strategy `hard_route_distillation`. Runtime: an immutable evaluation-mode `RouterScoreBank` supplies anchor-first hard assignments over an exact CAL-001 feature bank; zero-initialized affine students fit independently by layer through seed-free full-batch CPU float64 Adam, retain the best finite regularized iterate, pass a deployed-dtype acceptance gate, and compile into a storage-independent buffer-only RTR-003 router. Invariants: fixed-zero anchor tie priority, exact teacher layer/expert order, finite CPU and dtype alignment, teacher/feature/model/expert immutability, no trainable deployment parameters, deterministic repeated state, hashed scores and assignments, strict report round trips, and no hidden registry activation. Theory: `routers/README.md` with equations, algorithms, assumptions, Hugging Face boundary, and limitations. References: Dai et al., “StableMoE: Stable Routing Strategy for Mixture of Experts” (ACL 2022). Differences: this feature distills offline fixed-zero-anchor top-1 assignments from an arbitrary frozen composed teacher; it does not reproduce StableMoE's jointly learned balanced teacher or later expert-training stage. Follow-up: teacher-score distribution distillation, top-2/adaptive targets, hysteresis, frozen-route expert training, streaming/accelerator fitting, and report-bearing persistence remain separate work. |
| RTR-008 | P2 | Router | Route hysteresis or cross-layer stickiness | RTR-003 | Not started | — | Experimental |
| POL-001 | P0 | Selection | Anchored hard top-1 | RTR-002 | Validated | 2026-07-27 | 2026-07-27 — Validated in Codex task. Tests: `test_selection.py`, `test_affine_router_top1.py`, package-policy metadata integration, documentation contract, executable example, deterministic wheel-content check, and full regression suite (93 passed). Config: standalone `selection.policy: anchored_top1` with fixed `anchor_score: 0.0`; INF-001 now provides explicit routed-layer integration while model-level configuration remains rejected until package composition exists. Theory: `policies/README.md` with candidate-index semantics, equations, tensor shapes, anchor-first and append-order tie-breaking, and limitations. References: PyTorch `argmax`/`gather` and the architecture blueprint; no paper-derived algorithm. Follow-up: package-driven model configuration remains separate work. |
| POL-002 | P0 | Selection | Anchor-veto weighted top-2 | RTR-002, COM-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_top2_selection.py`, `test_affine_router_top2.py`, combination integration, documentation contract, executable `verify_anchor_veto_top2.py`, wheel-content and dependency checks, and full regression suite (249 passed). Config/API: standalone `selection.policy: anchor_veto_top2` through `AnchorVetoTop2Selection` and pair-shaped `Top2RouteDecision`; model-level configuration remains explicitly rejected. Runtime: stable descending ranking preserves anchor-first and append-order ties; the anchor vetoes unless two specialists are strictly positive; empty and single-expert registries remain anchor-only. Theory: `policies/README.md` with equations, shapes, algorithms, invariants, and limitations. References: PyTorch stable `argsort` and the architecture blueprint; no paper-derived algorithm. Follow-up: package attachment, compatibility groups, capacity degradation, telemetry, and model-level configuration remain separate work. 2026-07-27 — INF-002 now consumes POL-002 through standalone sparse top-2 dispatch. |
| POL-003 | P1 | Selection | Adaptive anchor/top-1/top-2 | POL-001, POL-002 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_adaptive_selection.py`, `test_affine_router_adaptive.py`, documentation contract, executable `verify_adaptive_selection.py`, all examples, wheel-content and dependency checks, and full regression suite (285 passed). Config/API: standalone `selection.policy: adaptive_top1_top2` through `AdaptiveTop1Top2Selection(mix_threshold)` and auditable pair-ranked `AdaptiveRouteDecision`; model-level configuration remains explicitly rejected. Runtime: route arity zero/one/two represents anchor/top-1/top-2; stable specialist-only descending ranking preserves append-order ties; the anchor wins when the best score is non-positive; an exact finite non-negative mixture-threshold tie remains top-1; top-2 requires the second score to be strictly above threshold; empty and single-expert registries use inactive zero padding. Invariants: ranked metadata, route arity, mode masks, and active per-expert masks preserve token shape, dtype/device semantics, input immutability, and deterministic ordering; malformed active padding and duplicate rankings are rejected. Theory: `policies/README.md` with equations, tensor shapes, algorithms, assumptions, Hugging Face boundary, and limitations. References: PyTorch stable `argsort` and the architecture blueprint; no paper-derived algorithm. Follow-up: adaptive sparse execution, BAL-002 capacity degradation, compatibility checks, telemetry, package attachment, and model-level configuration remain separate work. |
| POL-004 | P2 | Selection | Hierarchical package top-1 and local top-2 | RTR-005, POL-002 | Not started | — | — |
| POL-005 | P2 | Selection | Global compatibility-constrained top-2 | PKG-003, POL-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_compatibility_top2_selection.py`, `test_registry_compatibility_top2.py`, documentation contract, executable `verify_compatibility_top2.py`, all examples, deterministic wheel-content and dependency checks, and full regression suite (317 passed). Config/API: standalone `selection.policy: global_compatibility_top2` through `CompatibilityConstrainedTop2Selection` bound to an immutable PKG-003 index. Runtime: only distinct same-group specialists with two strictly positive calibrated scores are eligible; pairs are ranked lexicographically by descending member scores; append order resolves exact score and pair ties; missing groups are non-mixable; no eligible pair returns the strongest specialist beside the fixed-zero anchor for exact fallback. Theory: `policies/README.md` with equations, algorithms, complexity, invariants, Hugging Face boundary, and limitations. References: PyTorch stable `argsort`/`where` and the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: package-driven top-2 attachment, capacity degradation, telemetry, and model-level configuration remain separate work. 2026-07-28 — INF-002 now consumes POL-005 through the public `Top2SelectionPolicy` contract; registry-to-dispatch integration, exact anchor fallback, sparse call counts, and gradient preservation passed in the 319-test regression suite. |
| POL-006 | P2 | Selection | Dynamic-\(k\) selection | POL-003 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_dynamic_k_selection.py`, `test_affine_router_dynamic_k.py`, documentation contract, executable `verify_dynamic_k_selection.py`, all 30 examples, dependency check, compile check, and full regression suite (436 passed). Config/API: `DynamicKRouteDecision` and `DynamicKSelection(max_experts, activation_threshold)` with strategy `dynamic_k`. Runtime: stable specialist-only ranking, strict calibrated-score confidence gating, anchor arity zero, explicit per-token maximum-expert compute bound, inactive padding for undersized registries, and auditable arity/per-expert masks. Invariants: deterministic append-order ties, finite non-negative thresholds, positive integer budgets, retained-score dtype/device/gradient preservation, input immutability, distinct real candidates, and no active padding. Theory: `policies/README.md` with equations, algorithms, assumptions, Hugging Face boundary, and limitations. References: Huang et al. dynamic routing and Li et al. DA-MoE; the fixed-zero anchored threshold and deterministic per-token cap are documented project-specific differences. Follow-up: arbitrary-\(k\) compatibility, combination, capacity, sparse dispatch, telemetry, package attachment, learned difficulty/allocation, and batch-wide budgets remain separate work. |
| COM-001 | P0 | Combination | Score-softmax activation combination | POL-002 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_combination.py`, `test_affine_router_top2.py`, POL-002 integration, documentation contract, executable `verify_anchor_veto_top2.py`, wheel-content and dependency checks, and full regression suite (249 passed). Config/API: standalone `combination.policy: score_softmax` through `ScoreSoftmaxCombination(mix_temperature)` and auditable `ActivationCombinationResult`; model-level configuration remains explicitly rejected. Runtime: finite positive temperature, numerically stable pair softmax, arbitrary token/output shapes, exact dtype/device preservation, non-negative sum-to-one weights, input immutability, and gradients through scores and activations. Theory: `policies/README.md` with equations, tensor shapes, algorithm, numerical contract, and limitations. References: PyTorch `softmax` and the architecture blueprint; no paper-derived algorithm. Follow-up: pair compatibility, capacity handling, package attachment, and model-level configuration remain separate work. 2026-07-27 — INF-002 now consumes COM-001 after ranked sparse expert execution. |
| COM-002 | P1 | Combination | Equal and normalized-positive-evidence baselines | COM-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: COM-002 cases in `test_combination.py` and `test_top2_dispatch.py`, documentation contract, executable `verify_combination_baselines.py`, all examples, wheel-content and dependency checks, and full regression suite (269 passed). Config/API: standalone `combination.policy: equal_weight` through `EqualWeightCombination` and `combination.policy: normalized_positive_evidence` through `NormalizedPositiveEvidenceCombination`; model-level configuration remains explicitly rejected. Runtime: both policies share the pair-shaped `CombinationPolicy` contract; equal weights are exactly one half with no score dependency; positive evidence clamps scores at zero, rejects zero total evidence, and preserves gradients through positive scores; inputs, dtype, device, token order, output shape, and auditable sum-to-one weights are preserved. INF-002 now accepts the generic combination interface while retaining COM-001 as its default. Theory: `policies/README.md` and `inference/README.md` with equations, algorithms, gradient behavior, invariants, and limitations. References: PyTorch `clamp` and the architecture blueprint; no paper-derived algorithm. Follow-up: declarative model configuration, package attachment, pair compatibility, capacity handling, and telemetry remain separate work. |
| COM-003 | P1 | Combination | Learned mixture-weight head | COM-001, CAL-003 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_learned_mixture.py`, `test_learned_mixture_dispatch.py`, documentation contract, executable `verify_learned_mixture_weights.py`, all 31 examples, all seven notebooks through the regression contract, deterministic wheel-content check, and full regression suite (450 passed). Config/API: `MixtureWeightFittingConfig`, `MixtureWeightFittingReport`, `fit_learned_mixture_head()`, `LearnedMixtureWeightHead`, and `LearnedMixtureWeightCombination` with policy `learned_mixture_weight`. Runtime: caller-supplied normalized oracle weights align to exact CAL-003 plan-local pair indices; seed-free full-batch CPU float64 Adam fits a two-score affine mixing logit with soft-target binary cross entropy and normalized coefficient decay; the neutral half-weight initialization remains acceptance-eligible; deployment uses storage-independent frozen buffers, preserves selected pair order, and integrates through the unchanged INF-002 `CombinationPolicy` boundary. Serialization: strict atomic standalone safetensors with finite coefficient and metadata validation. Invariants: selection scores decide the pair while separate learned logits decide interpolation; inputs are immutable; reports contain canonical tensor digests and counts but no prompts or score values; top-2 route decisions remain exact. Theory: policies, calibration, inference, and root READMEs. References: PyTorch Adam, binary cross entropy, softmax, safetensors, SHA-256, and the architecture blueprint; the affine objective and explicit oracle-target boundary are project-specific rather than paper-derived. Follow-up: model-integrated counterfactual weight sweeps, pair-identity or hidden-state features, package-manifest and configuration-driven attachment, accelerator/distributed fitting, and COM-005 remain separate work. |
| COM-004 | P2 | Combination | Always-on anchor plus routed deltas | EXP-002, POL-003 | Not started | — | Shared-anchor strategy |
| COM-005 | P3 | Combination | Parameter-space expert merging | EXP-002, COM-003 | Not started | — | SMEAR/X-LoRA-inspired |
| INF-001 | P0 | Inference | Anchored top-1 token dispatch and output reconstruction | EXP-001, RTR-002, POL-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_dispatch.py`, `test_routed_mlp.py`, `test_routed_top1_qwen3.py`, documentation contract, executable example, all examples, wheel-content and dependency checks, and full regression suite (172 passed). Config: explicit `RoutedMLP.configure_anchored_top1()` with stable router-head order and optional telemetry. Theory: `inference/README.md` with dispatch equations, tensor shapes, sparse grouping, reconstruction, gradients, and limitations. References: PyTorch `index_select`, `index_copy`, `ModuleList`, and the architecture blueprint; no paper-derived algorithm. Follow-up: package-driven attachment, token-aligned auxiliary arguments, optimized grouping, and model-level composition remain separate features. 2026-07-27 — INF-002 adds standalone top-2 dispatch without changing the validated INF-001 path. |
| INF-002 | P0 | Inference | Pair-shaped top-2 token dispatch, activation combination, and output reconstruction | EXP-001, RTR-002, POL-002, POL-005, COM-001 | Validated | 2026-07-28 | 2026-07-27 — Validated in current Codex task. Tests: `test_top2_dispatch.py`, `test_top2_dispatch_full_mlp.py`, POL-002/COM-001 integration, preserved `test_dispatch.py`, documentation contract, executable `verify_top2_dispatch.py`, all examples, wheel-content and dependency checks, and full regression suite (258 passed). Config/API: standalone `AnchoredTop2Dispatcher` composed from `AnchorVetoTop2Selection` and `ScoreSoftmaxCombination`; `Top2DispatchResult` exposes ranked routes and full-shape mixture weights. Runtime: vetoed tokens execute one sparse anchor batch; approved tokens execute each selected specialist once in append order, reconstruct first/second rank with differentiable `index_copy`, combine through COM-001, and restore original token order without dropping tokens; empty and single-expert registries remain exact anchor execution. Invariants: inactive experts are not executed; anchor tokens have zero specialist weights; approved weights are finite, non-negative, and sum to one; scores, activations, outputs, and weights share dtype/device; score and expert gradients are preserved. Theory: `inference/README.md` with equations, tensor shapes, algorithms, complexity, invariants, and limitations. References: PyTorch `index_select`, `index_copy`, `softmax`, and the architecture blueprint; no paper-derived algorithm. Follow-up: `RoutedMLP` and package synchronization, capacity degradation, telemetry, token-aligned auxiliary arguments, optimized grouping, multi-device/distributed validation, and model-level configuration remain separate work. 2026-07-27 — COM-002 generalized the dispatcher to the `CombinationPolicy` interface, retained score-softmax as the default, and validated equal-weight and normalized-positive-evidence execution in the 269-test regression suite. 2026-07-28 — Validated compatibility-constrained execution in current Codex task. Tests: `test_top2_dispatch.py`, `test_registry_compatibility_top2.py`, documentation contract, executable `verify_compatibility_top2.py`, all 22 examples, deterministic wheel-content and dependency checks, and full regression suite (319 passed). Config/API: public `Top2SelectionPolicy` narrows pair-shaped selection; `AnchoredTop2Dispatcher` accepts POL-002 or POL-005 and validates result IDs, token shape, dtype, and device before module execution. Runtime: incompatible high-score specialists never execute; eligible compatible pairs preserve ranked sparse execution and score/expert gradients; no eligible pair executes the exact anchor. Theory: inference, policies, registry, and root READMEs. References: existing PyTorch dispatch operations and the architecture blueprint; framework infrastructure, not paper-derived. |
| CAL-001 | P0 | Calibration | Router feature cache and reference-bank format | RTR-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_router_feature_bank.py`, `test_router_feature_cache.py`, documentation contract, executable example, wheel-content and dependency checks, and full regression suite (106 passed). Config: `router_feature_bank_format_version: 1`. Theory: `calibration/README.md` with aligned design-matrix equations, deterministic collection, safe serialization, provenance, integrity, and limitations. References: safetensors, Hugging Face Datasets, FIPS 180-4, and the architecture blueprint; no paper-derived algorithm. Follow-up: CAL-002 labels, model hooks, fitting, threshold selection, sharding, and streaming remain separate features. |
| CAL-002 | P0 | Calibration | Domain-label and utility-label datasets | CAL-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_router_label_banks.py`, `test_router_label_datasets.py`, documentation contract, executable example, wheel-content and dependency checks, and full regression suite (122 passed). Config: `router_label_bank_format_version: 1`. Theory: `calibration/README.md` with categorical-domain and signed individual-utility equations, feature-bank binding, deterministic ordering, safe serialization, and limitations. References: safetensors, Hugging Face Datasets, FIPS 180-4, and the architecture blueprint; no paper-derived algorithm. Follow-up: CAL-003 pair utility, CAL-004 fitting, CAL-007 counterfactual execution, model hooks, sharding, and streaming remain separate features. |
| CAL-003 | P1 | Calibration | Pair-utility and best-pair sampling | CAL-002, COM-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_pair_utilities.py`, `test_pair_utility_oracle_pipeline.py`, documentation contract, executable `verify_pair_utilities.py`, all examples, and full regression suite (306 passed). Config/API: `PairSamplingPlan`, `plan_expert_pairs(max_pairs=64, seed=0)`, `PairUtilityResult`, and `compute_pair_utilities()`, with `max_pairs=None` as explicit exhaustive mode. Runtime: unbiased SHA-256 stream sampling of unique flat pair positions without bounded-mode \(O(K^2)\) materialization; canonical registry-ordered columns; JSON-compatible plan metadata; defensive detached CPU loss storage; anchor utility; improvement over best single; strict mixture margin; stable best-pair ties; and direct CAL-007 construction. Integration: COM-001 pair outputs flow through measured loss, CAL-003, CAL-007, and EVAL-004. Theory: `calibration/README.md` with equations, algorithms, complexity, invariants, Hugging Face boundary, and limitations. References: FIPS 180-4 and the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: model-integrated counterfactual execution, learned mixture weights, persistent pair-bank serialization, sharding, and production-scale sampling studies remain separate work. |
| CAL-004 | P0 | Calibration | Independent ridge fitting | CAL-002, RTR-002 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_ridge.py`, `test_independent_ridge_top1.py`, documentation contract, fitted-head serialization, executable example, wheel-content and dependency checks, all examples, and full regression suite (134 passed). Config: `fitting: independent_ridge`, finite positive `regularization`, unregularized intercept, CPU float64 solve, and frozen feature-dtype heads. Theory: `calibration/README.md` with centered multi-output equations, ordering, numerical invariants, algorithm, and limitations. References: Hoerl and Kennard ridge regression paper and the architecture blueprint, with implementation differences documented. Follow-up: CAL-005 temperature and threshold fitting; domain-label targets, weighted fitting, and incremental fitting remain separate work. |
| CAL-005 | P0 | Calibration | Temperature and threshold fitting | CAL-004 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_temperature.py`, `test_temperature_threshold_pipeline.py`, documentation contract, calibrated-head serialization, executable example, wheel-content and dependency checks, all examples, and full regression suite (154 passed). Config: `calibration: temperature_threshold`, explicit disjoint temperature/threshold row-ID partitions, inclusive `false_activation_rate`, bounded deterministic inverse-temperature search, strict deployed-dtype threshold semantics, and immutable frozen output heads. Theory: `calibration/README.md` with binary positive-utility equations, held-out partitioning, false-activation-constrained threshold selection, tie-breaking, fallbacks, complexity, reports, and limitations. References: Guo et al. temperature scaling paper and the architecture blueprint; binary per-head and threshold-policy differences are documented. Follow-up: CAL-006 stability-margin refinement; automatic splitting, joint multiclass calibration, model hooks, and dispatch remain separate work. |
| CAL-006 | P1 | Calibration | Stability-margin refinement | CAL-005 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_stability_margins.py`, `test_stability_margin_pipeline.py`, documentation contract, executable `verify_stability_margins.py`, all 26 examples, all six notebooks, final wheel API/content/dependency checks, and full regression suite (392 passed). Config/API: `refine_stability_margins()`, `StabilityMarginRefinementReport`, and per-head `HeadStabilityMarginReport` with explicit disjoint stability/win row IDs, non-negative margins, positive hinge weights, proximal drift, fixed learning rate, and fixed step count. Runtime: each frozen CAL-005 candidate affine head is refined independently against `max(anchor, incumbent scores)` through seed-free full-batch CPU float64 Adam; only positive-utility win rows participate; the best initialization/iterate is copied to source dtype/device and retained only when the deployed objective improves. Invariants: incumbent, candidate, banks, score order, temperature, and threshold remain unchanged; candidate/incumbent IDs are disjoint; `None` is exact anchor-only operation; RTR-002 and RTR-003 incumbents are supported; outputs are frozen, storage-independent, safely serializable RTR-002 heads; post-refinement metrics supersede CAL-005 metrics for changed affine heads. Theory: `calibration/README.md`, plus router and root integration notes. References: architecture blueprint margin equations; deterministic optimizer, proximal drift, candidate-only ownership, and deployed report are documented project-specific additions, not paper-derived. Follow-up: CAL-005 false-activation budgets are not re-enforced after refinement; automatic partitions, package derivation, registry-wide RTR-004 recalibration, and PKG-005 router-version persistence remain separate work. |
| CAL-007 | P1 | Calibration | Counterfactual route oracle and router regret | CAL-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_counterfactual_oracle.py`, `test_top1_top2_oracle_pipeline.py`, documentation contract, executable `verify_top1_top2_oracle.py`, all examples, and full regression suite (294 passed). Config/API: `CounterfactualRoute`, immutable `CounterfactualRouteLosses`, `CounterfactualOracleResult`, and `evaluate_counterfactual_oracle()`. Runtime: canonical anchor-first, complete-single, sparse-pair columns; stable first-minimum ties; defensive detached CPU storage; exact selected-route lookup; and non-negative per-token regret. Theory: `calibration/README.md` with candidate, oracle, regret, ordering, complexity, algorithms, invariants, and limitations. References: architecture blueprint; framework infrastructure, not paper-derived. Follow-up: CAL-003 pair sampling and serialized pair utilities, model-integrated counterfactual execution, sharding, and persistent loss tables remain separate work. |
| BAL-001 | P0 | Capacity | Overflow-to-anchor policy | CORE-004 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_capacity.py`, dispatch and routed-wrapper overflow integration, documentation contract, executable example, all examples, and full regression suite (172 passed). Config: `capacity.overflow: anchor` through `OverflowToAnchorCapacity`; explicit boolean availability has token shape plus stable expert dimension. Theory: `policies/README.md` and `inference/README.md` with effective-route equations, shape contracts, and no-token-dropping invariant. References: PyTorch `gather` and the architecture blueprint; no paper-derived algorithm. Follow-up: capacity estimation, second-choice fallback, and top-2-to-top-1 degradation remain separate features. |
| BAL-002 | P1 | Capacity | Capacity-aware top-2 to top-1 degradation | BAL-001, POL-003 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_capacity.py`, `test_top2_dispatch.py`, full-MLP and compatibility integration, documentation contract, updated executable `verify_top2_dispatch.py`, all 25 examples, all six notebooks, deterministic wheel-content and dependency checks, and full regression suite (373 passed). Config/API: standalone `capacity.degradation: top2_to_top1` through `Top2ToTop1Capacity`, auditable `Top2CapacityDecision`, and `AnchoredTop2Dispatcher(expert_availability=...)`; model-level configuration remains explicitly unsupported. Runtime: an approved pair retains both available specialists, promotes either lone survivor to effective slot zero with coefficient one, or executes the anchor when neither survives; anchor-vetoed routes remain anchor routes and never report overflow. Invariants: no token dropping, no search beyond the selected pair, unavailable experts never execute, exact input dtype/device and stable expert order, pair-shaped overflow metadata, pre-execution composed-policy validation, preserved top-2 score gradients, and no capacity estimation. Theory: `policies/README.md` and `inference/README.md` with equations, shapes, algorithms, Hugging Face boundary, assumptions, and limitations. References: PyTorch `gather`, `where`, `index_select`, and `index_copy` plus the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: capacity estimation, fallback beyond the selected pair, adaptive dispatcher integration, telemetry, package-driven top-2 attachment, `RoutedMLP`, model-level configuration, and multi-device/distributed validation remain separate work. |
| BAL-003 | P2 | Balancing | Auxiliary load-balancing objective | TRN-003 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_auxiliary_load_balancing.py`, BAL-003 cases in `test_local_mini_moe_training.py` and `test_local_mini_moe_trainer.py`, documentation contract, executable `verify_local_mini_moe.py`, all 33 examples, all seven notebooks, deterministic wheel-content inspection, and full regression suite (484 passed). Config/API: optional non-negative `LocalMiniMoETrainingConfig.load_balancing_weight`, `AuxiliaryLoadBalancingLoss`, `AuxiliaryLoadBalancingError`, `LocalMiniMoEObjectiveReport`, and final objective reporting. Runtime: graph-connected COM-001 probabilities produce the token-masked Switch-style assignment-fraction/mean-probability objective independently per converted layer; layer losses are averaged and observations are released after the same forward. Invariants: zero weight retains no probability graph and preserves the original TRN-003 loss; both experts still execute; stable argmax ties, all-zero masks, exact optimizer identity, byte-exact anchor state, one-shot cleanup, and no objective serialization are enforced. Theory: `training/README.md` with equations, algorithm, assumptions, configuration, references, deviations, tests, and limitations. References: Switch Transformers and the architecture blueprint; unlike Switch, hard assignments measure load but do not control dense local execution, token masks are supported, and layer losses are averaged. Follow-up: more than two experts, sparse dispatch, diversity objectives, gradient-checkpointing acceptance, mixed precision, multi-device, distributed training, and inference-time balancing remain separate work. |
| BAL-004 | P2 | Balancing | Bias-based auxiliary-loss-free balancing | TRN-003 | Not started | — | DeepSeek-V3-inspired |
| BAL-005 | P3 | Balancing | Expert-choice routing research backend | TRN-003 | Not started | — | Training/prefill first |
| PKG-001 | P0 | Packaging | Expert manifest schema and hash validation | CORE-001 | Validated | 2026-07-27 | 2026-07-25 — Validated in Codex task. Tests: manifest, hashing, documentation, and full regression suite (39 passed); wheel contents and dependency check passed. Config: `package_format_version: 1`, canonical JSON, safetensors payloads, exact base/tokenizer pins, runtime and dataset provenance. Theory: registry and utils READMEs. References: Safetensors, Hugging Face Hub revisions, FIPS 180-4, and RFC 8785 with documented project-specific differences. 2026-07-27 — PKG-002 now provides strict local descriptors and registry lifecycle. Follow-up: runtime payload bindings, upgrades, rollback, and Hub transfer remain separate tracked features. |
| PKG-002 | P0 | Registry | Install, unload, and active-registry hash | PKG-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_registry.py`, strict package fixture coverage, documentation contract, executable lifecycle example, all examples, deterministic wheel-content and dependency checks, and full regression suite (182 passed). Config: `registry_format_version: 1`, strict `ExpertPackage.from_directory()`, atomic `ExpertRegistry.install()`/`unload()`, permanent anchor/feature compatibility binding, historical slot ordering, and portable active-registry SHA-256. Theory: `registry/README.md` with identity equations, algorithms, security checks, locking, invariants, and limitations. References: safetensors, Hugging Face revisions, FIPS 180-4, RFC 8785 with documented differences, Python `RLock`, and the architecture blueprint; no paper-derived algorithm. Follow-up: PKG-007 payload binding/model attachment, persistence, upgrades, rollback, Hub transfer, and recalibration remain separate work. |
| PKG-003 | P1 | Registry | Compatibility groups for expert mixtures | PKG-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_mixture_compatibility.py`, `test_registry_compatibility_top2.py`, POL-005 integration, documentation contract, executable `verify_compatibility_top2.py`, all examples, deterministic wheel-content and dependency checks, and full regression suite (317 passed). Config/API: manifest `mixture_compatibility_group`, immutable `MixtureCompatibilityIndex`, and `ExpertRegistry.mixture_compatibility(layer_index=...)`. Runtime: compatibility is keyed by explicit payload-bound expert IDs; two distinct experts mix only when both declare the same non-empty group; missing groups never form an implicit default; candidate order realigns by ID but candidate sets must match exactly; legacy binding-free packages, layer gaps, duplicate cross-package IDs, and concurrent registry changes fail explicitly. Theory: `registry/README.md` with compatibility equations, algorithms, lifecycle boundary, invariants, and limitations. References: safetensors/package identity contracts and the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: package-driven top-2 attachment remains separate work. 2026-07-28 — The immutable layer index now drives standalone POL-005/INF-002 sparse execution; install/unload behavior, compatible mixing, and exact post-unload anchor fallback passed in the 319-test regression suite. |
| PKG-004 | P1 | Registry | Atomic package upgrades and rollback | PKG-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: PKG-004 cases in `test_registry.py`, model synchronization coverage in `test_package_runtime_composition.py`, documentation contract, executable `verify_registry_lifecycle.py`, all 25 examples, all six notebooks, deterministic wheel-content check, and full regression suite (367 passed). Config/API: `RegistryUpgradeChange`, `ExpertRegistry.upgrade()`/`.rollback()`, and matching `AnchoredMoEForCausalLM` delegates. Runtime: a strict live-validated replacement must preserve specialization and compatibility while changing immutable version in the exact active slot; rollback requires the exact old package and upgrade record and rejects any intervening registry state. Invariants: active count and unrelated order remain fixed; all failure records and payload checks precede mutation; inactive paths are not retained; registry changes never synchronize model routes implicitly; exact rollback restores the prior snapshot and registry hash. Theory: `registry/README.md` with replacement equations, algorithms, Hugging Face boundary, assumptions, and limitations. References: Python `RLock`, SHA-256, immutable package identities, and the architecture blueprint; framework infrastructure, not paper-derived. 2026-07-28 — CORE-006 now persists the active package snapshot but deliberately excludes inactive historical slots and rollback records. Follow-up: complete registry-history persistence, cross-process locking, router-version restoration, automatic synchronization, and NB-007 remain separate work. |
| PKG-005 | P1 | Registry | Registry-specific router versions | PKG-002, RTR-004 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_registry_router_versions.py`, `test_registry_router_recalibration.py`, documentation contract, executable `verify_registry_router_versions.py`, all 28 examples, all seven notebooks, deterministic wheel-content check, and full regression suite (409 passed). Config/API: `registry_router_version_format_version: 1`, `RegistryRouterVersion`, `RegistryRouterVersionError`, `save_safetensors()`, `from_safetensors()`, and explicit `activate(snapshot)`. Runtime: one standalone safe artifact stores finite CPU affine tensors plus canonical JSON containing the complete registry snapshot, compatibility binding, feature specification, RTR-004 report, global layer/expert order, temperatures, thresholds, dtypes, names, and dtype-aware tensor SHA-256 digests. Invariants: strict metadata/tensor closure, canonical reconstruction, source-storage independence, fresh frozen routers on access, exact snapshot equality before activation, atomic overwrite-protected save, symbolic-link rejection, and no install/unload/upgrade/rollback/model mutation or hidden recalibration. Theory: registry, router, and calibration READMEs plus root integration notes. References: safetensors, SHA-256, canonical JSON, and the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: authenticated signatures, composition-checkpoint embedding and router history, Hub transfer, cross-process lifecycle persistence, runtime auto-selection, multi-device state, and distributed validation remain separate work. |
| PKG-006 | P2 | Hub | Expert package Hub upload and download | PKG-001 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_package_hub.py`, registry and documentation contracts, executable `verify_package_hub.py`, all 34 examples, all seven notebooks, dependency and wheel-content checks, and full regression suite (495 passed). Config/API: `ExpertPackage.from_pretrained()` resolves local directories or Hub model repositories with revision, cache, offline, token, force-download, and endpoint options; `ExpertPackage.push_to_hub()` exposes explicit visibility, branch, pull-request, existing-repository, token, endpoint, and commit-message choices. Runtime: manifest-first download resolves one immutable 40-character commit, transfers only manifest-declared files, converts trusted Hub cache symlinks into a content-addressed regular-file package, records non-identity Hub provenance, and stages a second strictly validated upload snapshot before remote mutation. Invariants: package identity and bytes remain unchanged; moving revisions cannot mix manifest and payload commits; escaping symlinks, tampering, malformed snapshots, non-strict upload descriptors, incoherent options, and silently ignored local-path Hub arguments are rejected; registry/model state and authentication remain outside transport. Theory: `registry/README.md` plus root integration notes. References: official Hugging Face Hub download, cache, and `HfApi` documentation; framework infrastructure, not paper-derived. Follow-up: credentialed public/private/enterprise endpoint validation, large-package transfer benchmarks, remote composition checkpoints, and registry-router artifact transfer remain separate work. |
| PKG-007 | P0 | Packaging and inference | Explicit `(layer, expert_id)` payload bindings and package-driven routed-layer attachment | PKG-002, INF-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_payload_bindings.py`, routed-runtime replacement/reset cases, `test_package_runtime_composition.py`, documentation contract, executable package-runtime example, all examples, wheel-content and dependency checks, and full regression suite (193 passed). Config: `payload_binding_format_version: 1` with explicit `payload_bindings`; legacy package-format-v1 manifests remain registry-compatible but are not attachable without bindings. Runtime: `synchronize_anchored_top1_registry()` safely materializes full-MLP experts and independent affine heads in active-registry order, explicitly resets an empty registry to forced anchor, and never mutates a model during install/unload. Theory: registry, inference, expert, and router READMEs. References: safetensors, PyTorch module/state contracts, and the architecture blueprint; no paper-derived algorithm. Follow-up: exact live-checkpoint hash verification, automatic model configuration, multi-device validation, alternative expert/router strategies, and top-2 remain separate features. |
| EVAL-001 | P0 | Evaluation | Forced-anchor hidden-state and logit equivalence | CORE-004 | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: `test_qwen3_equivalence.py`; exact CPU logits and every hidden state, plus SHA-256 anchor-state checks. Config: explicit `atol=0`, `rtol=0`. Theory: `evaluation/README.md`. References: PyTorch numerical-accuracy documentation. Follow-up: add declared tolerances for validated runtime variants. |
| EVAL-002 | P0 | Evaluation | Route logging and exact route agreement | POL-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_route_telemetry.py`, `test_route_agreement.py`, routed-wrapper and Qwen3 integration, documentation contract, executable example, wheel-content and dependency checks, and full regression suite (172 passed). Config: opt-in bounded `RouteTelemetry`; exact `compare_route_decisions()` and `assert_route_decisions_equal()`. Theory: `inference/README.md` and `evaluation/README.md` with privacy exclusions, aggregate counts, and agreement equations. References: PyTorch tensor operations and the architecture blueprint; no paper-derived algorithm. Follow-up: cross-layer trajectory metrics, first-divergence depth, route serialization, and distributed aggregation remain separate features. |
| EVAL-003 | P0 | Evaluation | Installation, removal, and reinsertion tests | PKG-002, EVAL-002 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_registry_lifecycle.py`, registry unit lifecycle cases, documentation contract, executable lifecycle example, all examples, wheel-content and dependency checks, and full regression suite (182 passed). Config: read-only `evaluate_registry_lifecycle()` over immutable installed/removed/reinserted snapshots. Theory: `evaluation/README.md` and `registry/README.md` with removal, count, ordered-identity, hash-restoration, and compatibility-binding invariants. References: architecture blueprint; no paper-derived algorithm. 2026-07-27 — PKG-007 integration now covers model-level route installation, exact forced-anchor restoration after unload, deterministic routed-logit restoration after reinsertion, multi-package order, and pre-mutation rejection of mismatched or tampered payloads. Persistent and distributed lifecycle evaluation remain separate work. |
| EVAL-004 | P1 | Evaluation | Top-1/top-2 oracle comparison and regret | CAL-007, POL-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_top1_top2_oracle_comparison.py`, `test_top1_top2_oracle_pipeline.py`, documentation contract, executable `verify_top1_top2_oracle.py`, all examples, and full regression suite (294 passed). Config/API: `Top1Top2OracleComparison` and `compare_top1_top2_to_oracle()`. Runtime: exact registry/token alignment, direct top-1 mapping, anchor-veto mapping, order-independent sparse-pair lookup, best-single/best-pair loss, selected and oracle pair gain, top-1/top-2 regret, and pair-only destructive-mixture rate. Theory: `evaluation/README.md` with equations, algorithms, invariants, Hugging Face boundary, and limitations. References: architecture blueprint; no paper-derived algorithm. Follow-up: model-integrated counterfactual execution, compute-adjusted quality, confidence intervals, NB-006, and production benchmarks remain separate work. |
| EVAL-005 | P1 | Evaluation | Latency, throughput, memory, and active-MLP metrics | CORE-005 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_efficiency.py`, `test_efficiency_pipeline.py`, documentation contract, executable `evaluate_registry_strategies.py`, all 23 examples, all six notebooks, deterministic wheel-content and dependency checks, and full regression suite (332 passed). Config/API: `EfficiencyBenchmarkConfig`, `EfficiencyBenchmarkReport`, `ModuleStateFootprint`, `ActiveMLPMetrics`, `benchmark_operation()`, `measure_module_state_footprint()`, and `measure_active_mlp_metrics()`. Runtime: explicit work units; warmup exclusion; synchronized CPU/CUDA/MPS timing; retained mean, median, nearest-rank p95, and throughput; post-execution unique parameter/buffer storage accounting with trainable and non-resident meta semantics; optional CUDA allocator peak; exact top-1, pair-shaped top-2, and adaptive MLP token executions plus ideal sparse module-call counts. Invariants: inputs and routes remain immutable; shared state is counted once; anchor veto counts only the exact anchor; timing tests use no hardware thresholds; unsupported CPU/MPS runtime peaks are explicit `None`. Theory: `evaluation/README.md` with latency, throughput, compute, storage equations, algorithms, assumptions, and limitations. References: PyTorch timing, device synchronization, storage, and allocator APIs plus the architecture blueprint; framework infrastructure, not paper-derived. Follow-up: production hardware baselines, confidence intervals, CPU/MPS runtime peaks, energy, and distributed benchmarking remain separate work. |
| EVAL-006 | P1 | Evaluation | Installation-order and registry-subset evaluation | PKG-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_registry_composition_evaluation.py`, `test_registry_subset_evaluation.py`, documentation contract, executable lifecycle example, all 25 examples, all six notebooks, deterministic wheel-content and dependency checks, and full regression suite (358 passed). Config/API: `RegistryCompositionTrial`, `RegistrySubsetEvaluation`, `RegistryCompositionEvaluation`, and `evaluate_registry_compositions(maximum_trials=2048, behavior_fingerprint=...)`. Runtime: canonical package identity order; empty plus every non-empty subset; every installation permutation in fresh registries; complete-set compatibility preflight; exact partial-permutation budget rejection before observation; optional lowercase SHA-256 behavior digests with registry-mutation detection. Invariants: caller package order does not affect reports; ordered registry hashes remain descriptive; callbacks are explicit and package/model behavior is not retained; package files are never changed. Theory: `evaluation/README.md` with trial-count and order-sensitivity equations, algorithms, Hugging Face boundary, assumptions, and limitations. References: architecture blueprint, standard finite permutations, and SHA-256; framework infrastructure, not paper-derived. Follow-up: model-integrated route/logit fingerprints, persistent reports, metric-magnitude comparison, parallel trials, and production-scale subset sampling remain separate work. |
| EVAL-007 | P2 | Evaluation | Router-data scaling experiment versus partial MoE | RTR-004, TRN-003 | Not started | — | Core research hypothesis |
| NB-001 | P0 | Notebooks | Inspect and pin base model | CORE-003 | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: `test_foundation_notebooks.py` structure and offline cell execution; full regression suite (42 passed). Config: one resolved experiment mapping, exact model/tokenizer revisions, `upper_half`, and `ANCHORED_MOE_SMOKE_TEST`. Theory: backend README and architecture pinning invariants. References: Hugging Face Qwen3 configuration and model APIs. Follow-up: execute the production path against a pinned Qwen3-4B revision in a model-capable environment. |
| NB-002 | P0 | Notebooks | Convert model and verify anchor equivalence | EVAL-001 | Validated | 2026-07-25 | 2026-07-25 — Validated in Codex task. Tests: `test_foundation_notebooks.py` structure and offline cell execution; exact CPU logits, every hidden state, anchor fingerprints, and frozen-parameter assertions; full regression suite (42 passed). Config: `selection.policy: forced_anchor`, `atol=0`, `rtol=0`, and `upper_half`. Theory: backend and evaluation READMEs. References: PyTorch numerical-accuracy guidance. Follow-up: add declared tolerances only when optimized runtime variants are validated. |
| NB-003 | P0 | Notebooks | Train and export an expert package | TRN-001, PKG-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_package_export.py`, `test_foundation_notebooks.py`, manifest/binding/trainer integration, documentation contract, executable `train_expert_package.py`, all examples, deterministic wheel-content and dependency checks, and full regression suite (222 passed). Config/API: `ExpertPackageExportConfig`, `ExpertPackageExportBinding`, and `export_full_mlp_expert_package()` derive topology, dtype, safe filenames, PKG-007 bindings, and hashes from independently owned trained full-MLP experts. Workflow: `02_train_expert_package.ipynb` uses one resolved experiment mapping and only public APIs for deterministic forced-path training, expert-only staged export, strict reload, and validation. Invariants: no anchor/router/optimizer serialization, exact converted-anchor fingerprints, expert updates, complete rectangular payload ownership, storage independence, immutable destinations, deterministic package identity, and staging cleanup on failure. Theory: registry, training, expert, and root READMEs. References: Hugging Face Trainer/Accelerate, safetensors, FIPS 180-4, and the architecture blueprint; export is infrastructure, not paper-derived. Follow-up: NB-004 calibrated router artifact construction, pinned Qwen3-4B execution, replay-teacher provenance, mixed precision, multi-device, and distributed validation remain separate work. |
| NB-004 | P0 | Notebooks | Fit and calibrate router | CAL-005 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_calibrated_package_export.py`, `test_foundation_notebooks.py`, CAL-004/CAL-005 and PKG-007 integration, documentation contract, executable `fit_router.py`, all examples, deterministic wheel-content and dependency checks, and full regression suite (226 passed). Config/API: `CalibratedRouterPackageConfig` and `derive_calibrated_router_package()` consume exact fitting/calibration banks plus the frozen CAL-005 router/report pair and publish a new immutable anchored-top-1 package version. Workflow: `03_fit_and_calibrate_router.ipynb` uses one resolved experiment mapping and only public APIs for ridge fitting, explicit disjoint temperature/threshold calibration, reproducibility-bank serialization, routed-package derivation, strict reload, anchor/expert route checks, and model materialization. Invariants: source package bytes unchanged; new specialization version; globally disjoint fit/calibration row IDs; exact feature/layer/expert/head/report/dtype agreement; frozen finite heads; unchanged expert payload hashes; complete router/provenance/report hashes and metadata; no prompt embedding; deterministic derived identity. Theory: calibration, registry, and root READMEs. References: Hoerl and Kennard ridge regression, Guo et al. temperature scaling, safetensors, FIPS 180-4, and the architecture blueprint; package derivation adds no paper-derived algorithm. Follow-up: production feature/utility collection and pinned Qwen3-4B execution, counterfactual labels, sharding/streaming, mixed precision, multi-device, and distributed validation remain separate work. |
| NB-005 | P0 | Notebooks | Compose a custom model from packages | PKG-002, PKG-007, POL-001 | Validated | 2026-07-27 | 2026-07-27 — Validated in current Codex task. Tests: `test_foundation_notebooks.py`, executable `verify_package_runtime.py`, all examples, and full regression suite (208 passed). Workflow: `04_compose_custom_model.ipynb` uses only public APIs and a deterministic offline Qwen3 fixture to construct a validated bound package, prove that install/unload require explicit synchronization, restore exact dense logits after removal, and restore the exact registry hash and routed logits after reinsertion. Config: one resolved experiment mapping with smoke package identity, converted layers, input, and zero equivalence tolerances. Theory: root, inference, and registry READMEs. References: Hugging Face causal-model APIs, safetensors package contracts, SHA-256, and the architecture blueprint; no paper-derived algorithm. 2026-07-27 — NB-003/NB-004 now provide the validated expert-only to calibrated routed-package artifact chain and NB-004 proves PKG-007 materialization. Follow-up: retain this notebook's self-contained smoke fixture; add persistent cross-notebook artifact handoff only with production-path validation. |
| NB-006 | P1 | Notebooks | Evaluate registry strategies | EVAL-004, EVAL-005 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_foundation_notebooks.py`, EVAL-004/EVAL-005 integration, executable `evaluate_registry_strategies.py`, all 23 examples, deterministic wheel-content and dependency checks, and full regression suite (332 passed). Workflow: `05_evaluate_registry.ipynb` uses one resolved experiment mapping and only public APIs to compare deterministic offline top-1/top-2 counterfactual regret, synchronized latency/throughput, identical frozen module-state footprint, and exact active-MLP work. Invariants: stable registry order and work-unit definition; zero trainable parameters; no downloads or reusable notebook definitions; positive timing without machine-specific thresholds; exact top-1 work and top-2 pair accounting; concise JSON-compatible validation summary. Theory: `evaluation/README.md` and root README. References: EVAL-004/EVAL-005 infrastructure and the architecture blueprint; no paper-derived algorithm. Follow-up: production Qwen3-4B and accelerator runs, package-driven top-2 model execution, confidence intervals, and persistent benchmark artifacts remain separate work. |
| NB-007 | P1 | Notebooks | Export, reload, remove, and upgrade packages | CORE-006, PKG-004 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: `test_foundation_notebooks.py`, CORE-006/PKG-004 lifecycle and documentation cases (41 focused tests), executable `export_reload_lifecycle.py`, all 27 examples, all seven notebooks, wheel-content and dependency checks, and full regression suite (402 passed). Workflow: `06_export_and_reload.ipynb` uses one resolved experiment mapping and only public APIs to save and reload an immutable anchored-top-1 composition, remove and reinsert its package, upgrade to a new immutable version, and roll back to the exact prior version. Invariants: exact CPU logits and registry hash after reload, anchor fallback after removal, routed restoration after reinsertion, changed output and registry identity after upgrade, exact slot/hash/output restoration after rollback, complete parameter freezing, and no hidden synchronization across any registry mutation. Theory: registry and inference READMEs plus the root README. References: CORE-006/PKG-004 infrastructure and the architecture blueprint; no paper-derived algorithm. Follow-up: composition checkpoints retain only active packages, so the prior package is supplied explicitly for rollback; inactive history persistence, Hub transfer, anchor bundling, production Qwen3-4B, multi-device, and distributed validation remain separate work. |
| DOC-001 | P0 | Documentation | Source comment, docstring, and paper-reference standard | — | Validated | 2026-07-25 | 2026-07-25 — Validated for the foundation scope in Codex task. Tests: public-symbol docstring contract. Config: none. Theory: `docs/source-documentation-standard.md`. References: reference and deviation rules documented. Follow-up: apply to every later feature. |
| DOC-002 | P0 | Documentation | Theory README template and README in every source subpackage | DOC-001 | Validated | 2026-07-25 | 2026-07-25 — Validated for current feature-bearing subpackages in Codex task. Tests: README-presence contract and wheel package-data check. Config: none. Theory: template plus backend/evaluation READMEs. References: local reference sections. Follow-up: require a README with each new subpackage. |
| DOC-003 | P1 | Documentation | Synchronize feature IDs, configuration names, code, equations, and citations | DOC-002 | Validated | 2026-07-28 | 2026-07-28 — Validated in current Codex task. Tests: expanded `test_documentation.py`, focused documentation suite, all 33 examples, all seven notebooks, deterministic wheel-content inspection, and full regression suite (484 passed). Config: none. Contracts: unique tracker IDs, resolvable dependencies, valid statuses and dated notes, exact README status mapping, public API/module columns, complete theory sections, documentation coverage for active and completed IDs, unique resolvable root exports, callable docstrings, and paper-reference/deviation markers. Semantic boundary: equations, configuration meanings, citation relevance, and deviation completeness remain explicit reviewer responsibilities documented in the source standard rather than unreliable text inference. Theory: `docs/source-documentation-standard.md`, `docs/theory-readme-template.md`, root README, and local subpackage READMEs. References: repository architecture and Hugging Face/PyTorch/primary-paper references recorded by each feature; DOC-003 itself is framework infrastructure, not paper-derived. Follow-up: extend structural rules when new artifact types or configuration registries are introduced; do not substitute string matching for mathematical review. |

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
- INF-001;
- CAL-001, CAL-002, CAL-004, CAL-005;
- BAL-001;
- PKG-002;
- PKG-007;
- EVAL-002;
- NB-003 through NB-005.

Exit criterion: one expert package can be trained, installed, selected, removed, and reinserted.

### M2 — Weighted top-2

Complete:

- TRN-002;
- POL-002 and POL-003;
- COM-001 and COM-002;
- INF-002;
- CAL-003 and CAL-007;
- EVAL-004.

Exit criterion: top-1 and top-2 strategies can be compared under the same registry and compute metrics.

### M3 — Router-only composition

Completed:

- RTR-003 and RTR-004;
- CAL-006;
- PKG-005;
- EVAL-006.

Pending:

- EVAL-007, whose TRN-003 dependency is now satisfied but whose matched
  partial-MoE scaling experiment remains unimplemented.

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
