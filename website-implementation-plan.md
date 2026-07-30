# Mutations Website Implementation Plan

## 1. Purpose

This document defines the implementation plan for the Mutations website.

The first priority is a documentation experience that explains the repository
from high-level concepts down to executable examples. The documentation must
remain approachable to readers who understand language models but are not
already experts in Mixture-of-Experts systems.

The website must:

1. mirror the structure of the Python package;
2. distinguish implemented end-to-end behavior from standalone research
   primitives;
3. explain the relevant mathematics in plain language;
4. provide short, verified code examples without overwhelming the main text;
5. connect paper-derived features to primary references, including direct
   arXiv links;
6. remain synchronized with the repository's source READMEs and public APIs.

This plan complements, rather than replaces:

- `anchored-plugin-moe-architecture.md`, which defines the research design;
- `feature-implementation-plan.md`, which tracks implementation status;
- the theory README in each `src/anchored_moe/` subpackage.

---

## 2. Current website baseline

The existing `website/` project is a static Astro site with:

- a visual landing page;
- a reusable base layout and brand components;
- an animated architecture visualization;
- a documentation index and dynamic document route;
- static paper, architecture, logo, and social-preview assets.

The current documentation route is tied to four legacy Markdown imports. The
new documentation system should replace that fixed list with a package-aligned
information architecture and a maintainable content source.

---

## 3. Documentation information architecture

### 3.1 Route structure

Create one documentation subpage for every feature-bearing Python subpackage:

```text
/docs/
├── overview/
├── getting-started/
├── configuration-and-modeling/
├── backends/
├── experts/
├── routers/
├── policies/
├── training/
├── calibration/
├── registry/
├── inference/
├── evaluation/
├── utils/
├── examples/
└── references/
```

The direct subpackage mapping is:

| Website route | Source of truth | Primary subject |
|---|---|---|
| `/docs/backends/` | `src/anchored_moe/backends/README.md` | Model-family discovery and conversion |
| `/docs/experts/` | `src/anchored_moe/experts/README.md` | Expert representations and storage |
| `/docs/routers/` | `src/anchored_moe/routers/README.md` | Feature extraction and router architectures |
| `/docs/policies/` | `src/anchored_moe/policies/README.md` | Selection, combination, and capacity |
| `/docs/training/` | `src/anchored_moe/training/README.md` | Expert and router training workflows |
| `/docs/calibration/` | `src/anchored_moe/calibration/README.md` | Fitting, calibration, and route labels |
| `/docs/registry/` | `src/anchored_moe/registry/README.md` | Packages, compatibility, and lifecycle |
| `/docs/inference/` | `src/anchored_moe/inference/README.md` | Sparse dispatch and model composition |
| `/docs/evaluation/` | `src/anchored_moe/evaluation/README.md` | Correctness, quality, and efficiency |
| `/docs/utils/` | `src/anchored_moe/utils/README.md` | Hashing and shared infrastructure |

`configuration.py` and `modeling.py` are not subpackages, so they should share
the `/docs/configuration-and-modeling/` foundation page.

### 3.2 Documentation navigation

The desktop layout should have:

- a persistent left sidebar containing the page hierarchy;
- a central article column;
- an on-page table of contents on the right;
- previous/next page navigation;
- a documentation search entry point;
- a visible link back to the repository source for each page.

On small screens, the left sidebar and table of contents should collapse into
accessible disclosure menus.

The sidebar should group pages as follows:

```text
Introduction
  Overview
  Getting started
  Configuration and modeling

Core components
  Backends
  Experts
  Routers
  Policies

Workflows
  Training
  Calibration
  Registry
  Inference
  Evaluation

Learn by example
  End-to-end examples

Project
  Utilities
  References
```

---

## 4. Standard page contract

Every package documentation page should follow the same reader-oriented
structure:

1. **What this part does** — a two- or three-sentence explanation.
2. **Where it fits** — its inputs, outputs, and neighboring components.
3. **Implemented approaches** — a compact strategy comparison.
4. **How it works** — a simple conceptual walkthrough.
5. **Mathematical intuition** — notation and equations with plain-language
   explanations.
6. **Minimal usage** — one short visible example.
7. **Verified implementation details** — expandable code snippets.
8. **Current limitations** — what is standalone versus fully integrated.
9. **References** — primary papers with title, authors, purpose, and arXiv link.
10. **Related pages** — links to adjacent documentation.

Each page must clearly display one of these integration labels for every
strategy:

- **Integrated** — available through the package-to-model lifecycle.
- **Composable** — implemented and tested, but requires explicit assembly.
- **Research primitive** — validated in isolation and intended for experiments.
- **Planned** — documented but not yet implemented.

These labels prevent a validated primitive from being mistaken for an
automatically configurable production path.

---

## 5. Expandable verified code snippets

### 5.1 Presentation

Long or secondary snippets should be hidden by default using an accessible
disclosure component. The summary must describe what the reader will reveal.

Rendered Markdown should support:

````html
<details>
  <summary>Show the complete verified example</summary>

  ```python
  from anchored_moe import AnchoredMoEForCausalLM

  model = AnchoredMoEForCausalLM.from_pretrained("Qwen/Qwen3-4B")
  ```

</details>
````

The Astro implementation should eventually wrap this pattern in a reusable
component, for example:

```text
website/src/components/docs/CodeDisclosure.astro
```

The component should support:

- a concise summary;
- syntax highlighting;
- copy-to-clipboard;
- an optional source-file link;
- an optional verification badge;
- keyboard operation and visible focus states;
- a non-JavaScript fallback through native `<details>` behavior.

### 5.2 Verification rules

A snippet may be labeled **Verified** only when:

- it is imported from, generated from, or exercised by a repository example or
  test;
- its imports match the current public API;
- it runs without network access unless clearly marked otherwise;
- its expected output or invariant is stated;
- the source test or example is linked.

Prefer adapting snippets from `examples/` instead of maintaining unrelated
copies in website content. A later build-time check should validate all Python
snippets marked as executable.

Visible snippets should normally remain under 12 lines. Complete setup,
training loops, payload construction, and validation should live inside the
expandable section.

---

## 6. Mathematical explanation guidelines

Mathematics should clarify behavior, not act as a substitute for explanation.
Each equation should be introduced in three layers:

1. a one-sentence intuition;
2. the equation;
3. a plain-language interpretation of every symbol and practical consequence.

For example, anchored top-1 routing can be introduced as:

> The anchor is treated as a permanent candidate with score zero. A specialist
> runs only when its calibrated score is strictly positive and greater than
> every other specialist score.

\[
g_0(h)=0,
\qquad
r(h)=\underset{k\in\{0,\ldots,K\}}{\operatorname{argmax}}\;g_k(h).
\]

Here, \(h\) is the token representation before the MLP, \(g_k(h)\) is the
calibrated score for specialist \(k\), and candidate \(0\) is the immutable
anchor. Anchor-first tie-breaking preserves the dense model when the router
has no positive evidence for specialization.

General writing rules:

- introduce no more notation than the page needs;
- define tensor shapes near the equation;
- explain edge cases such as ties, empty registries, and capacity overflow;
- connect every equation to the corresponding class or function;
- use small numeric examples for top-1, top-2, and mixture weights;
- avoid claiming that a project-specific mechanism comes from a paper;
- explicitly explain differences from cited methods.

---

## 7. Paper references

References should appear near the feature they motivate and again in the
page's reference section. Each entry should include:

- paper title;
- authors or project;
- direct arXiv link;
- the implementation detail it motivated;
- a short note explaining how this repository differs.

Initial references include:

| Topic | Reference |
|---|---|
| LoRA experts | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* — https://arxiv.org/abs/2106.09685 |
| Fine-grained experts | Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* — https://arxiv.org/abs/2401.06066 |
| Product-key expert retrieval | He, *Mixture of A Million Experts* — https://arxiv.org/abs/2407.04153 |
| Auxiliary load balancing | Fedus, Zoph, and Shazeer, *Switch Transformers* — https://arxiv.org/abs/2101.03961 |
| Router calibration foundations | Guo et al., *On Calibration of Modern Neural Networks* — https://arxiv.org/abs/1706.04599 |

The complete references page should be generated from a structured reference
collection so titles and URLs remain consistent across pages.

---

## 8. End-to-end examples page

`/docs/examples/` should be a landing page containing several complete
workflows. Each workflow should state:

- the problem it solves;
- whether it is fully integrated or explicitly composed;
- the expert representation;
- router, selection, combination, and capacity policies;
- training and calibration requirements;
- expected active MLP work;
- package and serialization support;
- current limitations;
- links to executable repository examples and tests.

### 8.1 Initial example set

#### Example 1 — Exact anchor conversion

Purpose: establish that converting selected Qwen3 MLP layers does not change
hidden states, logits, or generation.

```text
Qwen3 anchor → RoutedMLP wrappers → forced-anchor execution
```

Status: **Integrated**.

Source starting points:

- `examples/verify_dense_equivalence.py`
- `tests/integration/test_qwen3_equivalence.py`

#### Example 2 — Packaged anchored top-1

Purpose: train, calibrate, package, install, synchronize, and execute
full-MLP specialists while retaining exact anchor fallback.

```text
Full-MLP experts
  → independent affine router heads
  → calibrated anchored top-1
  → package registry
  → explicit model synchronization
```

Status: **Integrated** and the primary end-to-end reference workflow.

Source starting points:

- `examples/train_expert_package.py`
- `examples/fit_router.py`
- `examples/verify_package_runtime.py`
- `examples/export_reload_lifecycle.py`

#### Example 3 — Weighted top-2 specialists

Purpose: select two positive specialists, execute the selected pair, and mix
their activations using score-derived weights.

```text
Affine scores
  → anchor-veto top-2
  → sparse pair dispatch
  → score-softmax combination
```

Status: **Composable**. The primitives are validated, but automatic
package-driven top-2 model attachment is not yet implemented.

Source starting points:

- `examples/verify_anchor_veto_top2.py`
- `examples/verify_top2_dispatch.py`
- `tests/integration/test_top2_dispatch_full_mlp.py`

#### Example 4 — Adaptive anchor/top-1/top-2

Purpose: vary computation per token according to specialist confidence.

```text
No positive evidence → anchor
One useful specialist → top-1
Two sufficiently useful specialists → top-2
```

Status: **Composable**.

Source starting points:

- `examples/verify_adaptive_selection.py`
- `tests/integration/test_affine_router_adaptive.py`

#### Example 5 — Always-on anchor with routed LoRA deltas

Purpose: always execute the stable anchor and add zero, one, or two selected
anchor-relative adaptations.

\[
y(h)=A(h)+\sum_{k\in S(h)}\alpha_k(h)\Delta_k(h).
\]

Status: **Composable** through an explicit `RoutedMLP` mode; package-driven
LoRA materialization remains future work.

Source starting points:

- `examples/verify_always_on_anchor_deltas.py`
- `tests/integration/test_anchor_delta_routed_mlp.py`

#### Example 6 — Two-expert local mini-MoE training

Purpose: jointly train a package-local pair of specialists and their router
while keeping the anchor and incumbent artifacts frozen.

Status: **Composable training workflow**.

Source starting points:

- `examples/verify_local_mini_moe.py`
- `examples/verify_mixture_aware_refinement.py`
- `tests/integration/test_local_mini_moe_trainer.py`

#### Example 7 — Hierarchical package routing

Purpose: choose one specialization package first and then mix a local expert
pair without crossing package boundaries.

Status: **Research primitive** with validated routing and dispatch components.

Source starting points:

- `examples/verify_hierarchical_routing.py`
- `tests/integration/test_hierarchical_top2_dispatch.py`

#### Example 8 — Scaling to larger expert collections

Purpose: demonstrate prompt shortlisting, compatibility constraints,
dynamic-\(k\), or product-key retrieval as alternative scaling strategies.

Status: **Research primitives**. Each approach should have a separate tab or
subsection rather than being presented as one integrated runtime.

Source starting points:

- `examples/verify_prompt_shortlisting.py`
- `examples/verify_compatibility_top2.py`
- `examples/verify_dynamic_k_selection.py`
- `examples/verify_product_key_retrieval.py`

### 8.2 Example comparison table

The examples landing page should begin with a comparison table:

| Approach | Active paths | Expert type | Lifecycle integration | Best for |
|---|---:|---|---|---|
| Exact anchor | 1 anchor | None | Integrated | Equivalence baseline |
| Anchored top-1 | Anchor or 1 expert | Full MLP | Integrated | Conservative specialization |
| Weighted top-2 | Anchor or 2 experts | Any compatible complete expert | Composable | Testing expert synergy |
| Adaptive | 0–2 specialists | Compatible experts | Composable | Variable computation |
| Anchor plus deltas | Anchor + 0–2 deltas | LoRA/sparse delta | Composable | Preservation-first adaptation |
| Local mini-MoE | 2 specialists during training | Package-local pair | Training workflow | Mixture-aware specialization |
| Hierarchical | 1 package, local pair | Package-local experts | Research primitive | Many specialization packages |

---

## 9. Content implementation

### 9.1 Recommended Astro structure

Use an Astro content collection rather than hard-coded imports:

```text
website/src/
├── components/
│   └── docs/
│       ├── CodeDisclosure.astro
│       ├── EquationBlock.astro
│       ├── IntegrationBadge.astro
│       ├── PaperReference.astro
│       ├── StrategyTable.astro
│       └── TableOfContents.astro
├── content.config.ts
├── content/
│   └── docs/
│       ├── overview.mdx
│       ├── getting-started.mdx
│       ├── configuration-and-modeling.mdx
│       ├── backends.mdx
│       ├── experts.mdx
│       ├── routers.mdx
│       ├── policies.mdx
│       ├── training.mdx
│       ├── calibration.mdx
│       ├── registry.mdx
│       ├── inference.mdx
│       ├── evaluation.mdx
│       ├── utils.mdx
│       ├── examples.mdx
│       └── references.mdx
├── layouts/
│   └── DocsLayout.astro
└── pages/
    └── docs/
        ├── index.astro
        └── [...slug].astro
```

MDX is recommended because reusable disclosure, status, equation, and
reference components are central to the content model.

### 9.2 Content metadata

Each document should declare:

```yaml
title: Experts
description: Expert representations supported by Anchored Plug-in MoE.
order: 4
group: Core components
source:
  - src/anchored_moe/experts/README.md
feature_ids:
  - EXP-001
  - EXP-002
  - EXP-003
  - EXP-004
  - EXP-005
last_reviewed: 2026-07-30
```

Build validation should reject:

- duplicate routes or navigation order;
- missing source files;
- unknown feature IDs;
- invalid integration labels;
- malformed arXiv URLs;
- executable snippets without a verification source;
- pages without limitations and references sections.

---

## 10. Documentation design guidelines

The visual design should extend the existing restrained technical identity:

- light background, dark typography, and muted technical accents;
- readable article width of approximately 70–80 characters;
- strong hierarchy without excessive cards or decorative containers;
- equations visually separated but aligned with the article flow;
- integration badges that use both text and color;
- tables that remain horizontally usable on small screens;
- code blocks with high contrast and accessible copy feedback;
- reduced-motion behavior for all animated interactions.

Documentation should prioritize scanning:

- short introductory paragraphs;
- descriptive headings;
- compact comparison tables;
- visible definitions;
- expandable implementation depth;
- clear links to the next conceptual step.

---

## 11. Implementation status tracker

This section is the operational source of truth for website documentation
progress. Update it in the same change that implements or materially revises a
page.

### 11.1 Status conventions

| Status | Meaning |
|---|---|
| Not started | Requirements exist, but implementation has not begun |
| In progress | Work has started but does not yet satisfy its acceptance criteria |
| Implemented | The page or component exists and works locally |
| Validated | Implementation, content, links, accessibility, and build checks pass |
| Blocked | Work cannot continue until the recorded dependency is resolved |

Only one tracker item should normally be marked **In progress**. This gives a
future session an unambiguous resume point.

### 11.2 Current resume point

**Next item:** `WEB-DOC-005` — implement the reusable equation,
integration-status, paper-reference, and strategy-table components with
accessible labels and source-aware presentation.

The existing four-article documentation interface is a legacy baseline, not
an implementation of the new package-aligned documentation system.

### 11.3 Documentation infrastructure

| ID | Deliverable | Dependencies | Status | Last update | Next action or evidence |
|---|---|---|---|---|---|
| WEB-DOC-001 | Content collection, metadata schema, catch-all route, and `DocsLayout` | — | Validated | 2026-07-30 | Astro 7 content layer, strict schema, source/order validation, catch-all route, MDX integration, and base layout validated; `npm run build` passed with 0 diagnostics and generated `/docs/` plus `/docs/overview/`; desktop and 390 px browser checks passed |
| WEB-DOC-002 | Desktop sidebar, right table of contents, and previous/next navigation | WEB-DOC-001 | Validated | 2026-07-30 | Metadata-driven grouped sidebar, active-page semantics, heading-derived H2/H3 table of contents, and deterministic adjacent-page navigation implemented; `npm run build` passed with 0 diagnostics; navigation, focus targets, sticky rails, console output, and overflow validated at 1440 px, 1024 px, and 390 px |
| WEB-DOC-003 | Responsive and accessible mobile documentation navigation | WEB-DOC-002 | Validated | 2026-07-30 | Metadata-driven hierarchy and H2/H3 table-of-contents disclosures implemented with native-details fallback and keyboard normalization; `npm run build` passed with 0 diagnostics; Enter/Space activation, visible 2 px focus outlines, 60 px touch targets, active-page semantics, heading links, zero horizontal overflow, and empty console output validated at 390 px, 768 px, 1024 px, and 1440 px; desktop sticky rails remain unchanged |
| WEB-DOC-004 | Expandable verified code component | WEB-DOC-001 | Validated | 2026-07-30 | Reusable MDX component provides native-details fallback, Astro/Shiki highlighting, progressively enhanced clipboard copy with visible and announced feedback, optional verified labeling, and repository source links; both overview disclosures migrated; `verify_dense_equivalence.py` and `verify_package_runtime.py` passed; `npm run build` passed with 0 errors, warnings, or hints; server-rendered fallback retains summaries, code, and source links with copy controls hidden; Enter/Space activation, visible focus, exact clipboard contents, 44 px copy targets, zero horizontal overflow, and empty browser error output validated at 1440 px, 768 px, and 390 px |
| WEB-DOC-005 | Equation, integration-status, paper-reference, and strategy-table components | WEB-DOC-001 | Not started | 2026-07-30 | Build reusable MDX components with accessible status labels |
| WEB-DOC-006 | Documentation typography and responsive content styling | WEB-DOC-001 | Not started | 2026-07-30 | Extend the existing visual system for long-form technical content |
| WEB-DOC-007 | Static documentation search | All content pages | Not started | 2026-07-30 | Index titles, headings, feature IDs, and strategy names |
| WEB-DOC-008 | Content synchronization and build validation | WEB-DOC-001 | Not started | 2026-07-30 | Validate sources, feature IDs, arXiv links, and verified snippets |

### 11.4 Documentation subpages

| ID | Route | Source material | Status | Last update | Next action or evidence |
|---|---|---|---|---|---|
| WEB-PAGE-001 | `/docs/` | This plan and all page metadata | Validated | 2026-07-30 | Metadata-driven grouped landing page built with complete-pages-only rollout messaging; desktop and 390 px layouts, internal navigation, skip-link focus, and zero horizontal overflow validated |
| WEB-PAGE-002 | `/docs/overview/` | Root `README.md` and architecture blueprint | Validated | 2026-07-30 | Standard page contract, integration boundaries, plain-language routing equation, native disclosures, source links, limitations, and direct arXiv references validated; `verify_dense_equivalence.py` and `verify_package_runtime.py` passed |
| WEB-PAGE-003 | `/docs/getting-started/` | `pyproject.toml`, examples, and notebooks | Not started | 2026-07-30 | Define installation, first conversion, and offline smoke workflow |
| WEB-PAGE-004 | `/docs/configuration-and-modeling/` | `configuration.py`, `modeling.py`, root README | Not started | 2026-07-30 | Explain configuration, conversion, model façade, and routing modes |
| WEB-PAGE-005 | `/docs/backends/` | `backends/README.md` | Not started | 2026-07-30 | Document the backend contract and current Qwen3 implementation |
| WEB-PAGE-006 | `/docs/experts/` | `experts/README.md` | Not started | 2026-07-30 | Compare full, LoRA, sparse, micro, and retrieval representations |
| WEB-PAGE-007 | `/docs/routers/` | `routers/README.md` | Not started | 2026-07-30 | Explain router features, heads, compilation, and advanced routers |
| WEB-PAGE-008 | `/docs/policies/` | `policies/README.md` | Not started | 2026-07-30 | Explain selection, combination, capacity, and integration status |
| WEB-PAGE-009 | `/docs/training/` | `training/README.md` | Not started | 2026-07-30 | Present forced, independent, local-MoE, and refinement workflows |
| WEB-PAGE-010 | `/docs/calibration/` | `calibration/README.md` | Not started | 2026-07-30 | Explain feature banks, labels, fitting, thresholds, and oracles |
| WEB-PAGE-011 | `/docs/registry/` | `registry/README.md` | Not started | 2026-07-30 | Document package formats, compatibility, lifecycle, and Hub transfer |
| WEB-PAGE-012 | `/docs/inference/` | `inference/README.md` | Not started | 2026-07-30 | Explain dispatch, explicit synchronization, and telemetry |
| WEB-PAGE-013 | `/docs/evaluation/` | `evaluation/README.md` | Not started | 2026-07-30 | Cover equivalence, quality, efficiency, lifecycle, and scaling |
| WEB-PAGE-014 | `/docs/utils/` | `utils/README.md` | Not started | 2026-07-30 | Explain canonical hashing and shared infrastructure |
| WEB-PAGE-015 | `/docs/examples/` | Repository examples and integration tests | Not started | 2026-07-30 | Build the comparison table and publish the anchor baseline first |
| WEB-PAGE-016 | `/docs/references/` | Subpackage README reference sections | Not started | 2026-07-30 | Create the structured, deduplicated paper reference index |

### 11.5 End-to-end example status

| ID | Example | Integration label | Status | Last update | Next action or evidence |
|---|---|---|---|---|---|
| WEB-EX-001 | Exact anchor conversion | Integrated | Not started | 2026-07-30 | Adapt the dense-equivalence example and test |
| WEB-EX-002 | Packaged anchored top-1 | Integrated | Not started | 2026-07-30 | Document training through composition reload |
| WEB-EX-003 | Weighted top-2 specialists | Composable | Not started | 2026-07-30 | Adapt top-2 selection and dispatch examples |
| WEB-EX-004 | Adaptive anchor/top-1/top-2 | Composable | Not started | 2026-07-30 | Add a small variable-compute routing example |
| WEB-EX-005 | Always-on anchor with LoRA deltas | Composable | Not started | 2026-07-30 | Demonstrate exact fallback and routed deltas |
| WEB-EX-006 | Two-expert local mini-MoE training | Composable training workflow | Not started | 2026-07-30 | Adapt local training and refinement examples |
| WEB-EX-007 | Hierarchical package routing | Research primitive | Not started | 2026-07-30 | Show package-first and local-pair selection |
| WEB-EX-008 | Scaling larger expert collections | Research primitives | Not started | 2026-07-30 | Separate shortlisting, compatibility, dynamic-k, and retrieval variants |

### 11.6 Tracker update protocol

Whenever an item changes:

1. change its status;
2. update `Last update` using `YYYY-MM-DD`;
3. replace `Next action or evidence` with the most concrete next step or the
   validation evidence;
4. add newly discovered dependencies before beginning dependent work;
5. update **Current resume point** to the next incomplete item;
6. record validation commands and results in the completed item's evidence;
7. keep implementation labels synchronized with the Python framework's actual
   runtime integration.

An item moves to **Validated** only after:

- the static Astro build passes;
- internal links and referenced source paths resolve;
- responsive layout has been inspected;
- keyboard and disclosure behavior has been checked;
- mathematical notation and paper attribution have been reviewed;
- code marked **Verified** is tied to a passing example or test.

---

## 12. Implementation phases

### Phase 1 — Documentation foundation

- Introduce the Astro content collection and catch-all documentation route.
- Create `DocsLayout`, sidebar, table of contents, and mobile navigation.
- Add metadata validation and deterministic navigation ordering.
- Remove dependency on the four legacy Markdown imports.

Acceptance:

- every planned route builds statically;
- navigation works without client-side JavaScript;
- desktop and mobile layouts are keyboard accessible;
- missing or invalid content metadata fails the build.

### Phase 2 — Core reusable components

- Implement code disclosures, integration badges, equations, references, and
  strategy tables.
- Add syntax highlighting and copy-to-clipboard.
- Define global documentation typography and responsive table behavior.

Acceptance:

- disclosures work with keyboard and screen readers;
- hidden code remains readable without JavaScript;
- every paper link is a direct valid arXiv URL;
- integration status is visible without relying on color.

### Phase 3 — Package-aligned content

- Write overview, getting-started, and configuration/modeling pages.
- Create all ten subpackage pages.
- Link each page to its source README and public implementation modules.
- Add plain-language equations and paper references.

Acceptance:

- all feature-bearing subpackages have a documentation page;
- each page follows the standard page contract;
- strategy status matches `feature-implementation-plan.md`;
- limitations distinguish integrated and composable behavior.

### Phase 4 — End-to-end examples

- Build the examples landing page and comparison table.
- Publish the eight initial workflows.
- Extract or adapt snippets from existing examples and tests.
- Add expected invariants and source links.

Acceptance:

- the packaged top-1 workflow is complete from conversion through reload;
- advanced workflows state their integration boundary;
- every verified snippet maps to an executable repository source;
- examples are ordered from conservative baseline to experimental strategies.

### Phase 5 — Search and synchronization

- Add static full-text search.
- Generate the reference index from structured data.
- Add build-time feature-ID and source-link validation.
- Consider generating selected API tables from Python introspection.

Acceptance:

- users can search page titles, headings, strategy names, and feature IDs;
- reference metadata is defined once;
- stale source paths or unknown tracked features fail validation.

---

## 13. Definition of documentation success

The documentation foundation is successful when a reader can:

1. understand the immutable-anchor idea without reading the architecture paper;
2. identify the role of every `anchored_moe` subpackage;
3. tell which strategies work through the complete package lifecycle;
4. compare top-1, top-2, adaptive, delta, and hierarchical approaches;
5. reveal and run a verified example only when additional detail is needed;
6. understand the core equations without specialized MoE notation;
7. follow every paper-derived idea to its primary arXiv reference;
8. move from overview to a complete anchored top-1 implementation without
   leaving the documentation.
