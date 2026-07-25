# Anchored Plug-in Mixture-of-Experts

## A strategy blueprint for modular LLM post-training

**Initial target model:** Qwen3-4B  
**Design status:** Research and implementation blueprint  
**Primary objective:** Add and remove specialization packages without updating the frozen dense model or previously installed expert weights  
**Routing strategies covered:** Anchored top-1, anchored top-2, adaptive top-1/top-2, hierarchical routing, and prompt-shortlisted routing

---

## 1. Purpose

This document describes a family of **Anchored Plug-in Mixture-of-Experts (MoE)** architectures for extending an already capable dense language model through modular post-training.

The original dense model is kept as an immutable **anchor**. New capabilities are installed as packages containing:

- one or more specialist MLP experts;
- router parameters;
- routing and combination policy metadata;
- calibration statistics;
- compatibility and version information.

The framework is deliberately strategy-oriented. It does not prescribe one routing policy for every experiment. Instead, it defines common invariants and a step-by-step set of implementation choices.

The central research hypothesis is:

> Expert sets trained independently on sufficiently different specialization distributions can be composed into a partial MoE by fitting a lightweight router on representative subsets of each specialization's training distribution.

This hypothesis is most likely to hold when:

- all experts start from the same anchor;
- the shared transformer backbone remains frozen;
- each expert set is useful under forced routing before composition;
- router calibration data represents every installed specialization;
- mixed and boundary cases are included;
- weighted expert combinations are either trained directly or validated counterfactually.

The expected gap relative to an MoE trained jointly from the beginning comes primarily from missing expert-router co-adaptation. The framework therefore supports both strictly independent expert training and limited mixture-aware refinement.

---

## 2. Core model

For converted transformer layer \(\ell\), let:

- \(h_{\ell,t}\) be the pre-MLP representation of token \(t\);
- \(A_\ell\) be the frozen original MLP;
- \(E_{\ell,k}\) be installed expert \(k\);
- \(\phi_\ell(h)\) be the frozen router feature transformation;
- \(g_{\ell,k}(h)\) be the calibrated score for expert \(k\).

The anchor has a permanent score:

\[
g_{\ell,0}(h)=0.
\]

An expert can be represented as a full MLP or as an anchor-relative adaptation:

\[
E_{\ell,k}(h)=A_\ell(h)+\Delta_{\ell,k}(h).
\]

The delta form is especially useful when mixing expert outputs because it makes the shared anchor contribution explicit.

### 2.1 Terminology

The framework distinguishes the following levels:

```text
Base model
└── Specialization package, such as code or mathematics
    ├── Expert 1
    ├── Expert 2
    └── Router and calibration metadata
```

An expert does not need a human-interpretable role such as “generation” or “debugging.” Within a specialization, experts may learn low-level latent partitions of the specialization distribution.

### 2.2 Exact invariants

Every implementation strategy should preserve:

1. **Anchor immutability.** Original parameters never change.
2. **Shared-backbone immutability.** Attention, embeddings, normalization, and the language-model head remain frozen.
3. **Expert-weight modularity.** Installing a package does not update old expert weights.
4. **Exact fallback.** When the anchor is selected, it executes with coefficient one.
5. **No token dropping.** Capacity overflow returns to the anchor.
6. **Compatibility validation.** Packages target an exact base checkpoint and router feature space.
7. **Deterministic routing.** Tie-breaking and package ordering are stable.
8. **Auditable installation.** Active packages and their calibration results are recorded.

Some routing strategies additionally preserve old router parameters. Others intentionally allow router-only recalibration. That choice must be explicit.

---

## 3. Architecture decision map

The main design decisions are independent:

| Decision | Conservative choice | More expressive choice |
|---|---|---|
| Experts per specialization | One | Small latent expert set |
| Global routing | Independent top-1 scores | Hierarchical or global top-k |
| Active MLPs | One | Two |
| Output policy | Hard selection | Weighted expert mixture |
| Router updates on installation | Append-only head | Router-only recalibration |
| Expert training | Forced single path | Local mini-MoE training |
| Router target | Domain label | Counterfactual utility |
| Scale to many experts | Score all experts | Prompt-level shortlist |
| Package unit | One expert | Versioned expert set |

The architecture should expose these as configuration choices rather than hard-code them into expert weights.

---

## 4. Step 1 — Pin the base system

Before training any expert, record:

- model repository and exact revision;
- tokenizer repository and revision;
- configuration and weight hashes;
- framework and kernel versions;
- numerical precision;
- converted layer indices;
- router feature definition;
- router projection hash, if used;
- residual and MLP calling conventions.

Packages must refuse to load when these fields are incompatible.

### Approach A — Exact runtime pinning

Require matching checkpoint, framework, kernels, and precision.

**Advantages**

- strongest reproducibility;
- simplest dense-equivalence testing.

**Disadvantages**

- packages are less portable across runtimes.

### Approach B — Functional compatibility

Require matching weights and architecture, while allowing validated runtime variants.

**Advantages**

- easier deployment;
- supports optimized inference kernels.

**Disadvantages**

- equivalence becomes tolerance-based rather than bitwise.

### Initial recommendation

Use exact runtime pinning for research experiments, then introduce functional compatibility after the routing design is stable.

---

## 5. Step 2 — Select converted layers

Only selected transformer MLPs need to become routed layers.

### Approach A — Upper-layer conversion

Convert the upper half of the transformer.

**Advantages**

- lower memory and training cost;
- specialization closer to model output;
- easier initial debugging.

**Risks**

- insufficient capacity for low-level specialization.

### Approach B — Interleaved conversion

Convert every second or every third layer.

**Advantages**

- distributes specialist capacity across depth;
- may improve representation-level adaptations.

**Risks**

- longer routing trajectories;
- more opportunities for early route divergence.

### Approach C — Full conversion

Convert every MLP.

**Advantages**

- maximum specialist capacity;
- closest comparison with a conventional MoE.

**Risks**

- highest storage, training, and serving cost;
- more difficult behavioral stability.

### Approach D — Data-driven layer selection

Measure forced-expert improvement or representation separability by layer and convert only the most useful layers.

**Advantages**

- potentially best quality-to-cost ratio.

**Risks**

- adds a preliminary profiling stage;
- optimal layers may differ by specialization.

### Initial recommendation

Begin with the upper 18 of Qwen3-4B's 36 layers, then compare upper, interleaved, and data-driven selections.

---

## 6. Step 3 — Choose the specialization package structure

### Approach A — One expert per specialization

```text
Code package → one code MLP per converted layer
Math package → one math MLP per converted layer
```

**Advantages**

- simplest training and packaging;
- compatible with hard top-1 routing;
- one active MLP per token and layer.

**Limitations**

- no intra-specialization mixture;
- one MLP must model the entire specialization distribution.

### Approach B — Latent expert set per specialization

```text
Code package
├── latent expert 1
├── latent expert 2
├── latent expert 3
└── local routing metadata
```

Experts are not assigned manual semantic roles. They learn low-level partitions from data.

**Advantages**

- supports top-2 mixtures;
- closer to conventional MoE capacity;
- experts can specialize without human labels.

**Limitations**

- higher training and inference cost;
- requires mixture-aware training or validation;
- package installation and removal should normally be atomic.

### Approach C — Global expert pool

All experts compete in one global routing pool without package-local boundaries.

**Advantages**

- maximum flexibility;
- allows useful cross-specialization combinations.

**Limitations**

- hardest calibration problem;
- independently trained experts may not combine safely;
- removal can alter many top-k decisions.

### Approach D — Hybrid packages

Each package owns a latent expert set, but a global router may select experts across packages when compatibility conditions are satisfied.

**Advantages**

- retains package structure while allowing cross-domain composition.

**Limitations**

- requires compatibility metadata or learned pair utility;
- more complicated evaluation.

### Initial recommendation

Use a latent expert set as the package unit. Train two to four experts per specialization and install or remove the set atomically.

---

## 7. Step 4 — Choose expert parameterization

### Approach A — Full MLP copies

Initialize every specialist from the anchor:

\[
E_{\ell,k}\leftarrow A_\ell.
\]

**Advantages**

- maximum adaptation capacity;
- clean comparison with dense-to-MoE upcycling;
- exact equality at initialization.

**Disadvantages**

- substantial storage and optimizer memory.

### Approach B — Low-rank anchor deltas

\[
E_{\ell,k}(h)=A_\ell(h)+\Delta_{\ell,k}^{\text{low-rank}}(h).
\]

**Advantages**

- compact packages;
- explicit anchor-relative representation;
- inexpensive experimentation with larger expert sets.

**Disadvantages**

- lower capacity;
- adapter structure may bias specialization.

### Approach C — Structured or sparse deltas

Train sparse neuron, block, or projection updates around the anchor.

**Advantages**

- potentially efficient storage and execution;
- supports expert merging experiments.

**Disadvantages**

- more complex kernels;
- capacity depends heavily on structure.

### Approach D — Mixed parameterization

Use full copies in selected high-value layers and compact deltas elsewhere.

**Advantages**

- flexible quality-to-cost trade-off.

**Disadvantages**

- additional package and runtime complexity.

### Initial recommendation

Use full MLP copies for the first scientific comparison. Use low-rank deltas for scale experiments after routing behavior is understood.

---

## 8. Step 5 — Train a new expert or expert set

The anchor, shared backbone, old experts, and normally old router heads remain frozen.

### Approach A — Forced-path single-expert training

Force specialization examples through the new expert at every converted layer.

```text
Specialization example
    → new expert in every converted layer
```

**Advantages**

- avoids router-expert cold start;
- simple and stable;
- directly tests whether the expert is useful.

**Limitations**

- does not train expert complementarity;
- weighted mixtures may be suboptimal.

### Approach B — Independent training of multiple experts

Train several experts independently on different random or data partitions.

**Advantages**

- fully parallel expert creation;
- high modularity.

**Limitations**

- diversity is not guaranteed;
- independently useful outputs may not interpolate well.

### Approach C — Local mini-MoE training

Within one specialization package, jointly train its latent experts and a temporary or permanent local top-k router. The global model remains frozen.

**Advantages**

- experts co-adapt to weighted combinations;
- most likely to approximate a partial MoE trained from the start;
- encourages low-level latent specialization.

**Limitations**

- package training is more complex;
- the expert set becomes the natural atomic package.

### Approach D — Independent training followed by mixture-aware refinement

Train experts independently, then freeze or partially freeze them and refine:

- the new expert set;
- its local router;
- optionally only small expert deltas.

**Advantages**

- retains parallel initial training;
- repairs some mixture incompatibility.

**Limitations**

- not as strictly independent;
- adds another training stage.

### Expert objectives

A general objective is:

\[
\mathcal L_{\text{expert}}
=
\mathcal L_{\text{specialization}}
+
\lambda_{\text{KD}}\mathcal L_{\text{preserve}}
+
\lambda_{\text{drift}}\mathcal L_{\text{drift}}
+
\lambda_{\text{div}}\mathcal L_{\text{diversity}}.
\]

Possible terms include:

- next-token or post-training loss on specialization data;
- KL preservation against the anchor or incumbent on replay data;
- distance from anchor initialization;
- diversity or load-balancing terms for multi-expert packages.

### Expert acceptance gate

Before router fitting, verify:

1. forced routing improves the target distribution;
2. training is numerically stable;
3. the expert set adds enough value to justify its cost;
4. weighted mixtures are no worse than the best single expert, when top-2 is planned.

---

## 9. Step 6 — Choose router architecture

### Approach A — Independent append-only heads

Each expert owns one affine score per converted layer:

\[
s_{\ell,k}(h)=w_{\ell,k}^{\top}\phi_\ell(h)+b_{\ell,k},
\]

\[
g_{\ell,k}(h)=\frac{s_{\ell,k}(h)}{T_{\ell,k}}-\tau_{\ell,k}.
\]

Installing a package appends heads without updating old heads.

**Advantages**

- old scores are invariant for the same hidden state;
- clean installation and removal;
- simple analytical fitting.

**Limitations**

- independently trained score scales require careful calibration;
- a new head can still change the winner.

### Approach B — One logical router with append-only output columns

A single runtime router contains a shared frozen feature transformation and one independently owned output column per expert.

**Advantages**

- efficient implementation;
- behaves like one token-level router;
- retains append-only score computation.

**Limitations**

- the shared router trunk must remain frozen;
- changing normalization across all outputs can break modularity.

### Approach C — Router-only global recalibration

After installing a package, fine-tune a global router on small replay subsets from:

- every installed specialization;
- general data;
- mixed and boundary data.

All experts and the base model remain frozen.

**Advantages**

- may approach the routing quality of a partial MoE trained jointly;
- old and new score scales can be optimized together;
- particularly suitable when specialization scopes are distinct.

**Limitations**

- old routing behavior can change;
- installation is no longer strictly append-only at the router level;
- removal may require restoring or recalibrating a router version.

### Approach D — Hierarchical router

Use two routing stages:

1. select anchor or specialization package;
2. select one or two latent experts within the chosen package.

**Advantages**

- separates cross-specialization calibration from local mixing;
- local router scores need only be comparable within a package;
- prevents unintended mixtures across unrelated packages.

**Limitations**

- two routing stages;
- package-level mistakes block useful internal experts.

### Approach E — Global router with compatibility constraints

Select globally but only allow expert pairs declared compatible.

**Advantages**

- supports cross-package composition;
- retains bounded top-2 computation.

**Limitations**

- compatibility may require pairwise metadata or learning;
- naive pair modeling scales as \(O(K^2)\).

### Initial recommendation

Implement one logical router with append-only expert heads first. Add optional router-only global recalibration as an explicit operating mode, not as an invisible installation side effect.

---

## 10. Step 7 — Choose the routing policy

### Strategy A — Anchored hard top-1

\[
r_\ell(h)=\arg\max\{0,g_{\ell,1}(h),\ldots,g_{\ell,K}(h)\}.
\]

\[
y_\ell(h)=
\begin{cases}
A_\ell(h), & r_\ell(h)=0,\\
E_{\ell,r_\ell(h)}(h), & \text{otherwise}.
\end{cases}
\]

**Best for**

- first prototype;
- strict sparse execution;
- one expert per specialization;
- maximum interpretability.

**Trade-off**

- cannot combine complementary expert transformations.

### Strategy B — Anchor-veto weighted top-2

Take the two highest scores from the anchor and all specialists.

- If the anchor is in the top two, execute only the anchor.
- Otherwise, combine the two selected specialist experts.

Because the anchor score is zero, this is equivalent to requiring at least two specialists with positive adjusted scores.

For selected specialists \(i,j\):

\[
\alpha_i=
\frac{\exp(g_i/T_{\text{mix}})}
{\exp(g_i/T_{\text{mix}})+\exp(g_j/T_{\text{mix}})},
\qquad
\alpha_j=1-\alpha_i.
\]

\[
y_\ell(h)=
\alpha_i E_{\ell,i}(h)+\alpha_j E_{\ell,j}(h).
\]

Equivalently:

\[
y_\ell(h)
=
A_\ell(h)
+
\alpha_i\Delta_{\ell,i}(h)
+
\alpha_j\Delta_{\ell,j}(h).
\]

**Best for**

- packages containing at least two latent experts;
- conservative specialist activation;
- standard-MoE-like weighted execution.

**Trade-offs**

- one strong specialist cannot activate alone;
- removing one expert can disable a useful partner;
- two positive scores do not automatically prove mixture compatibility.

### Strategy C — Adaptive anchor/top-1/top-2

\[
y_\ell(h)=
\begin{cases}
A_\ell(h), & g_{(1)}\le 0,\\
E_{\ell,(1)}(h), & g_{(1)}>0,\;g_{(2)}\le\tau_{\text{mix}},\\
\alpha_1E_{\ell,(1)}(h)+\alpha_2E_{\ell,(2)}(h),
& g_{(2)}>\tau_{\text{mix}}.
\end{cases}
\]

Here \(g_{(1)}\) and \(g_{(2)}\) are the two highest specialist scores.

**Best for**

- gradual transition from top-1 to top-2;
- systems where a specialization may contain one or several experts;
- avoiding unnecessary second-expert computation.

**Trade-off**

- more thresholds and behaviors to calibrate.

### Strategy D — Hierarchical specialization top-1, local expert top-2

First select:

\[
c^*=\arg\max\{0,q_{\text{code}},q_{\text{math}},\ldots\}.
\]

If a specialization wins, its local router selects and mixes its top two experts.

**Best for**

- expert sets trained as local mini-MoEs;
- preventing cross-specialization mixing;
- scaling package development independently.

**Trade-off**

- requires package-level and expert-level routers.

### Strategy E — Global weighted top-2

The two highest specialists may come from any installed package.

**Best for**

- discovering cross-specialization compositions;
- expert sets that share a common mixture-aware training protocol.

**Trade-off**

- highest risk of incompatible mixtures;
- requires global calibration or compatibility controls.

### Strategy F — Prompt-shortlisted token routing

A prompt-level retriever chooses a small candidate set \(C(x)\), after which one of the preceding token-level policies is applied:

\[
C(x)=\operatorname{TopM}_k\pi_k(x).
\]

**Best for**

- large expert registries;
- controlling router and memory overhead.

**Trade-off**

- a retrieval miss prevents token-level recovery unless a broader fallback is provided.

---

## 11. Step 8 — Choose the anchor policy

### Approach A — Fixed-zero abstention

The anchor score is always zero. Specialists must exceed it.

**Advantages**

- simple and stable;
- independent thresholds have clear meaning.

### Approach B — Anchor veto in top-2

If the anchor appears in the selected top two, it executes alone.

**Advantages**

- requires two positive specialist signals;
- reduces activation from isolated false positives.

**Risks**

- errors from related experts may be correlated;
- single experts cannot activate.

### Approach C — Learned but frozen anchor confidence

Fit an anchor-confidence head once, then freeze it.

**Advantages**

- anchor preference can vary by hidden state.

**Risks**

- less exact and less interpretable than a fixed zero baseline;
- all specialist scores must be calibrated against it.

### Approach D — Dynamic incumbent threshold

Compare a new expert with the best currently available route:

\[
m_{\ell,\text{old}}(h)=
\max\left(0,\max_{k\in\mathcal E_{\text{old}}}g_{\ell,k}(h)\right).
\]

**Advantages**

- new packages learn to improve upon the installed system, not merely the anchor.

**Risks**

- the meaning of a score depends on the registry used during training.

### Initial recommendation

Keep the anchor score fixed at zero. Treat anchor-veto top-2 as a routing strategy rather than changing the anchor itself.

---

## 12. Step 9 — Fit the router

Router training can use progressively stronger supervision.

### Approach A — Domain labels

Train the router to recognize the source specialization.

**Advantages**

- inexpensive;
- requires no counterfactual execution.

**Limitations**

- domain membership does not imply expert utility;
- mixed-domain tokens are ambiguous.

### Approach B — Forced-path loss labels

Compare the incumbent path with a forced expert path.

\[
a_{\ell,t,k}
=
\mathcal L_{\text{incumbent},\ell,t}
-
\mathcal L_{\text{expert }k,\ell,t}.
\]

Positive advantage indicates that expert \(k\) helped.

**Advantages**

- directly targets utility;
- suitable for top-1 routing.

**Limitations**

- layerwise counterfactual labels are expensive.

### Approach C — Pair utility labels

For top-2 routing, evaluate:

\[
a_{\ell,t,i,j}
=
\mathcal L_{\text{incumbent},\ell,t}
-
\mathcal L_{\alpha E_i+(1-\alpha)E_j,\ell,t}.
\]

The pair should ideally outperform the best single expert:

\[
a_{i,j}>\max(a_i,a_j)+\delta_{\text{mix}}.
\]

**Advantages**

- directly tests whether the second MLP is useful;
- detects destructive interpolation.

**Limitations**

- pair evaluation can scale as \(O(K^2)\);
- requires sampling or package-local restrictions.

### Approach D — Analytical ridge initialization

For cached router features \(X\) and targets \(y\):

\[
w=(X^\top X+\lambda I)^{-1}X^\top y.
\]

**Advantages**

- inexpensive and deterministic;
- strong baseline;
- independently fit heads.

### Approach E — Margin refinement

On old-domain states:

\[
\mathcal L_{\text{stability}}
=
\mathbb E
\left[
\max(0,g_{\text{new}}-m_{\text{old}}+\gamma)
\right].
\]

On useful new-domain states:

\[
\mathcal L_{\text{win}}
=
\mathbb E
\left[
\max(0,m_{\text{old}}+\gamma_{\text{win}}-g_{\text{new}})
\right].
\]

**Advantages**

- explicitly controls route changes;
- supports sequential installation.

### Approach F — Router distillation from an oracle

For a calibration subset:

1. execute candidate experts or pairs;
2. select the lowest-loss route;
3. train the router to approximate the oracle.

**Advantages**

- applies to top-1, top-2, and adaptive policies;
- directly measures router regret.

**Limitations**

- expensive oracle labeling;
- the oracle is local to the calibration distribution.

### Recommended progression

1. Domain-label baseline.
2. Ridge regression on individual utility.
3. Stability-margin refinement.
4. Pair-utility sampling for top-2.
5. Optional router-only global recalibration.

---

## 13. Step 10 — Calibrate scores and mixture weights

Independent expert scores are not automatically comparable.

### Score calibration options

#### Option A — Per-head temperature and threshold

\[
g_k=\frac{s_k}{T_k}-\tau_k.
\]

Choose thresholds against a target false-activation or route-change budget.

#### Option B — Shared reference-bank normalization

Normalize every head using statistics from the same general and specialization reference banks.

#### Option C — Global router-only calibration

Freeze experts and jointly adjust router temperatures, biases, or all router weights using replay subsets.

#### Option D — Rank-based routing

Use scores mainly for ranking and separately decide whether the selected route beats the anchor.

### Mixture-weight options

#### Option A — Softmax over selected experts

\[
\alpha_k=\frac{\exp(g_k/T_{\text{mix}})}
{\sum_{j\in S}\exp(g_j/T_{\text{mix}})}.
\]

#### Option B — Normalized positive evidence

\[
\alpha_k=
\frac{\max(g_k,0)}
{\sum_{j\in S}\max(g_j,0)}.
\]

#### Option C — Learned mixture head

Use separate outputs for:

- whether an expert should activate;
- how much weight it should receive after selection.

This avoids assuming that selection confidence is the optimal interpolation coefficient.

#### Option D — Fixed equal weights

Use \(0.5/0.5\) as a diagnostic baseline.

### Initial recommendation

Start with temperature-scaled softmax over the selected experts, but measure it against equal weights and a learned mixture head.

---

## 14. Step 11 — Decide what changes when a package is installed

### Mode A — Strict append-only installation

Freeze:

- base model;
- old experts;
- old router heads;
- shared router features.

Append only the new package.

**Guarantee**

Old scores are unchanged for identical hidden states.

**Limitation**

The global decision can still change when a new score enters top-1 or top-2.

### Mode B — Router-only recalibration

Freeze all model and expert weights, but fine-tune the router on replay subsets from all specializations.

**Advantages**

- likely better global routing;
- can approximate a partial MoE router trained from the start.

**Limitations**

- old routing decisions can change;
- the router must be versioned with the active registry.

### Mode C — Package-local refinement

Freeze all old components and refine only:

- the new expert set;
- its local router;
- its package-level activation head.

**Advantages**

- preserves old packages;
- supports mixture-aware training.

**Limitations**

- cannot globally repair old/new routing boundaries.

### Mode D — Hybrid installation

Use append-only heads for normal installation, with optional scheduled router-only recalibration after several packages accumulate.

**Advantages**

- operationally modular;
- periodically recovers routing quality.

**Limitations**

- behavior differs between pre- and post-recalibration registries.

### Initial recommendation

Support both strict append-only and router-only recalibration as explicit framework modes. Store the active registry hash with every recalibrated router.

---

## 15. Step 12 — Removal and upgrade semantics

### Removal strategy A — Pure unloading

Remove the package weights and heads. Affected tokens fall back to:

1. the next eligible route;
2. otherwise the anchor.

**Best for**

- append-only routers;
- simple package management.

### Removal strategy B — Restore a registry-specific router

When using router-only recalibration, restore the router checkpoint corresponding to the remaining package registry.

**Best for**

- reproducible deployments;
- frequently used registry combinations.

### Removal strategy C — Recalibrate after removal

Fit a new router over the remaining frozen experts.

**Best for**

- maximizing post-removal quality.

**Trade-off**

- removal is no longer instantaneous.

### Upgrade strategy

Treat an upgrade as a new immutable package version:

1. train against a declared parent registry;
2. evaluate beside the old version;
3. activate atomically;
4. retain the old package and router version for rollback.

For a latent expert set, replace the package atomically rather than independently replacing internal experts unless that behavior has been explicitly tested.

---

## 16. Step 13 — Capacity and scaling

### Active computation

Hard top-1 executes one MLP per token and converted layer.

If a fraction \(p\) of tokens uses top-2 specialists, expected MLP executions are approximately:

\[
1+p.
\]

Top-2 therefore doubles converted-layer MLP work only when specialist routing is active for nearly every token.

### Router scaling options

#### Option A — Score every expert

Appropriate for small registries. Router heads are cheap compared with MLPs.

#### Option B — Score every package, then local experts

Hierarchical routing bounds local scoring and prevents unrelated mixtures.

#### Option C — Prompt-level shortlist

Retrieve a small candidate set before token routing.

#### Option D — Approximate maximum-score search

Index router keys or package embeddings for large registries.

### Capacity overflow options

1. Return overflow to the anchor.
2. Try the next eligible specialist.
3. Reduce top-2 decisions to top-1 under load.

The anchor must never reject overflow.

### Initial recommendation

Use score-all routing for the first two or three packages. Add hierarchical or prompt-level shortlisting only after router scoring becomes material.

---

## 17. Reference inference policies

### Anchored top-1

```python
def anchored_top1(hidden, layer, experts):
    winner = None
    winner_score = 0.0

    for expert in experts:
        score = expert.adjusted_score(layer, hidden)
        if score > winner_score:
            winner = expert
            winner_score = score

    if winner is None or not winner.has_capacity(layer):
        return layer.anchor_mlp(hidden)

    return winner.mlp(layer, hidden)
```

### Anchor-veto weighted top-2

```python
def anchor_veto_top2(hidden, layer, experts, mix_temperature):
    scored = [(0.0, None)]  # None represents the anchor.
    scored += [
        (expert.adjusted_score(layer, hidden), expert)
        for expert in experts
    ]

    top2 = stable_topk(scored, k=2, prefer_anchor_on_ties=True)

    if any(expert is None for _, expert in top2):
        return layer.anchor_mlp(hidden)

    (score_a, expert_a), (score_b, expert_b) = top2

    if not pair_is_compatible(expert_a, expert_b):
        return layer.anchor_mlp(hidden)

    weights = softmax([score_a, score_b], temperature=mix_temperature)
    output_a = expert_a.mlp(layer, hidden)
    output_b = expert_b.mlp(layer, hidden)
    return weights[0] * output_a + weights[1] * output_b
```

### Adaptive top-1/top-2

```python
def adaptive_topk(hidden, layer, experts, mix_threshold, temperature):
    ranked = stable_rank_specialists(hidden, layer, experts)

    if not ranked or ranked[0].score <= 0:
        return layer.anchor_mlp(hidden)

    first = ranked[0]

    if len(ranked) == 1 or ranked[1].score <= mix_threshold:
        return first.expert.mlp(layer, hidden)

    second = ranked[1]

    if not pair_is_compatible(first.expert, second.expert):
        return first.expert.mlp(layer, hidden)

    weights = softmax(
        [first.score, second.score],
        temperature=temperature,
    )
    return (
        weights[0] * first.expert.mlp(layer, hidden)
        + weights[1] * second.expert.mlp(layer, hidden)
    )
```

Production implementations must batch tokens by route, preserve deterministic ordering, and avoid storing sensitive token content in route logs.

---

## 18. Package manifest

Each package should contain:

```yaml
package_format_version:
specialization_name:
specialization_version:
package_structure: single_expert | latent_expert_set
base_model_repository:
base_model_revision:
base_config_sha256:
tokenizer_revision:
converted_layers:
expert_parameterization:
expert_count:
router_architecture:
routing_policies_supported:
router_feature_specification:
router_projection_hash:
mixture_compatibility_group:
mixture_temperature:
activation_thresholds:
weight_dtype:
expert_weight_files:
router_files:
calibration_dataset_version:
calibration_metrics:
training_parent_registry:
package_hashes:
```

`training_parent_registry` identifies the installed system against which interference and utility were calibrated.

`mixture_compatibility_group` specifies which experts may be combined. For strict hierarchical routing, this will normally be the package identifier.

---

## 19. Evaluation blueprint

Evaluate the sequential lifecycle:

```text
Dense
  → +Code expert set
  → +Math expert set
  → +Biomedical expert set
  → −Code
  → +Upgraded Code
```

### 19.1 Common metrics

| Metric family | Measurements |
|---|---|
| Target capability | Benchmark score, held-out loss, task pass rate |
| General retention | General benchmarks, perplexity, preference evaluation |
| Previous-specialist retention | Old-domain quality before and after installation |
| Route stability | Route agreement, first divergence depth, changed token-layer decisions |
| False activation | Specialist activation on irrelevant data |
| Utility alignment | Router regret against a counterfactual oracle |
| Calibration | Precision, recall, reliability, threshold curves |
| Modularity | Removal invariance, reinsertion reproducibility |
| Efficiency | Parameters, memory, latency, throughput, MLP executions |
| Robustness | Mixed prompts, paraphrases, multilingual and adversarial inputs |

### 19.2 Top-1 versus top-2 metrics

Measure:

- top-1 expert loss;
- equal-weight top-2 loss;
- score-weighted top-2 loss;
- learned-weight top-2 loss;
- oracle best-single loss;
- oracle best-pair loss;
- incremental gain from the second expert;
- fraction of top-2 decisions where mixing is worse than top-1;
- compute-adjusted improvement.

The most important criterion is:

> Does the selected pair consistently improve upon the best selected single expert enough to justify a second MLP execution?

### 19.3 Router-subset hypothesis test

To test whether small replay subsets can recover partial-MoE behavior:

1. Train each specialization expert set independently.
2. Freeze all experts.
3. Fit the router using increasing fractions of each specialization dataset.
4. Compare against a matched partial MoE trained jointly.
5. Plot performance and router regret against router-training sample count.
6. Evaluate specialization interiors, boundaries, and mixed inputs separately.

Expected result:

- small subsets may be sufficient for clearly separated specialization interiors;
- mixed and boundary cases will require disproportionately more calibration data;
- mixture-aware expert training should reduce the gap to joint MoE training.

### 19.4 Required lifecycle tests

1. Forced-anchor equivalence.
2. Empty-registry equivalence.
3. Forced specialist usefulness.
4. Irrelevant-package installation.
5. Route trajectory stability.
6. Removal and reinsertion.
7. Installation-order sensitivity.
8. Every package subset for small registries.
9. Capacity overflow.
10. Router-version rollback.

---

## 20. Experimental matrix

The first controlled study should vary one dimension at a time:

| Track | Package structure | Router update | Routing policy | Expert training |
|---|---|---|---|---|
| A | One expert | Append-only | Anchored top-1 | Forced path |
| B | Two latent experts | Append-only | Anchor-veto top-2 | Independent |
| C | Two latent experts | Append-only | Adaptive top-1/top-2 | Independent |
| D | Two latent experts | Router recalibration | Weighted top-2 | Independent |
| E | Two latent experts | Router recalibration | Weighted top-2 | Local mini-MoE |
| F | Four latent experts | Hierarchical | Local top-2 | Local mini-MoE |
| G | Multiple packages | Global recalibration | Global top-2 | Package-local training |

Track A establishes the modular baseline.

Tracks B and C isolate the effect of the anchor-veto rule.

Tracks D and E test whether router-only fine-tuning over small specialization subsets approaches partial-MoE quality.

Tracks F and G test scale and cross-package composition.

---

## 21. Recommended implementation roadmap

### Phase 1 — Dense equivalence

1. Pin Qwen3-4B.
2. Wrap selected MLPs.
3. Implement forced-anchor routing.
4. Verify hidden-state and logit equivalence.

### Phase 2 — Single-expert modular baseline

1. Train one code expert through forced routing.
2. Fit independent utility heads.
3. Implement anchored top-1.
4. Validate installation, removal, and route stability.

### Phase 3 — Latent expert set

1. Train two code experts.
2. Compare independent training with local mini-MoE training.
3. Implement anchor-veto top-2.
4. Implement adaptive top-1/top-2.
5. Compare mixture weighting strategies.

### Phase 4 — Router-subset composition

1. Add a mathematics expert set.
2. Freeze all expert weights.
3. Fine-tune only the router on small code, math, general, and mixed subsets.
4. Compare against strict append-only heads.
5. Measure the gap to a matched jointly trained partial MoE.

### Phase 5 — Registry scaling

1. Add a third specialization.
2. Compare global and hierarchical routing.
3. Add prompt-level candidate retrieval if needed.
4. Test removal, upgrades, and router-version restoration.

### Phase 6 — Storage optimization

1. Replace full expert copies with anchor-relative deltas.
2. Repeat the routing comparisons.
3. Measure quality, memory, and throughput trade-offs.

---

## 22. Recommended first configuration

| Setting | Choice |
|---|---|
| Base | Exact pinned Qwen3-4B checkpoint |
| Converted layers | Upper 18 |
| First specialization | Code |
| Experts in first package | Two latent experts |
| Expert representation | Full MLP copies |
| Expert training comparison | Independent versus local mini-MoE |
| Router implementation | One logical router with append-only heads |
| Router fitting | Utility ridge initialization plus margin refinement |
| Routing policies | Anchored top-1, anchor-veto top-2, adaptive top-1/top-2 |
| Mixture weights | Selected-score softmax |
| Anchor score | Fixed zero |
| Overflow | Anchor |
| Global router recalibration | Optional experimental mode |
| Candidate retrieval | Disabled initially |

This configuration directly tests whether modular, separately post-trained expert sets can recover the behavior of a partial MoE through lightweight router fitting.

---

## 23. Relationship to prior work

| Work | Retained idea | Blueprint extension |
|---|---|---|
| Sparse Upcycling | Dense checkpoint copied into sparse experts | Permanent immutable anchor and incremental packages |
| BTX | Branch, train, and compose domain models | Multiple install and router-update strategies |
| BAR | Modular domain post-training | Append-only or registry-versioned routing |
| FlexOlmo | Experts trained against a common frozen model | Anchored abstention and configurable top-k execution |
| DUME | Analytical router construction | Independent heads and pair-utility extensions |
| DEMix Layers | Removable domain modules | Token routing, latent expert sets, and prompt shortlisting |

The framework is not a claim that dense-to-MoE conversion is new. Its focus is the engineering and experimental space created by combining:

- an exact dense fallback;
- independently trainable specialization packages;
- configurable top-1 and top-2 routing;
- explicit installation and removal semantics;
- router-only composition experiments.

---

## 24. Open research questions

1. How separable are independently trained specialization distributions in layerwise hidden-state space?
2. How much router data is required to approach a partial MoE trained jointly?
3. Does local mini-MoE training materially outperform independently trained latent experts?
4. When does score-weighted top-2 outperform the best top-1 route?
5. Are selection confidence and optimal mixture weight sufficiently correlated?
6. How much full-model route drift results from small early-layer routing errors?
7. Can pair compatibility be inferred without \(O(K^2)\) labels?
8. Does hierarchical routing retain quality while scaling to many packages?
9. How sensitive is composition to package installation order?
10. Can compact anchor-relative deltas retain the benefits of full expert copies?
11. When should router-only recalibration be preferred over strict append-only routing?
12. Can router versions be transferred across compatible package upgrades?

---

## 25. Final design principle

The anchor, expert weights, router policy, and package lifecycle should be treated as separate concerns.

The framework should make it possible to answer experimentally:

```text
How are experts trained?
        ×
How is the router fitted?
        ×
How many experts execute?
        ×
How are outputs combined?
        ×
What changes during installation?
```

This separation turns the architecture from a single MoE proposal into a reusable post-training framework for comparing modular composition strategies under common invariants.

---

## 26. Future development

The following strategies are not required for the first implementation, but should remain explicit research tracks in the framework. Each should be implemented behind the same expert, router, combination, and registry interfaces so that it can be compared without rewriting the base system.

### 26.1 Always-on anchor with routed deltas

Instead of choosing between the anchor and specialists, always execute the anchor and add one or more routed adaptations:

\[
y_\ell(h)
=
A_\ell(h)
+
\sum_{k\in S_\ell(h)}
\alpha_{\ell,k}(h)\Delta_{\ell,k}(h).
\]

This gives the anchor the role of a shared expert. It may reduce the amount of general knowledge that routed experts must reproduce and makes the specialist contribution explicitly residual.

**Advantages**

- the dense computation is always preserved;
- specialists learn only incremental behavior;
- compact LoRA or structured deltas become natural expert formats.

**Trade-offs**

- the anchor and specialists execute together, increasing computation;
- exact dense fallback requires every routed delta to be absent or zero;
- full-copy experts are less attractive than compact deltas in this mode.

**Relevant work**

- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) separates shared experts from routed experts to capture common knowledge and reduce redundancy.

### 26.2 Dynamic-\(k\) routing and learned compute allocation

Generalize fixed top-1 and top-2 policies to:

\[
k_{\ell,t}\in\{0,1,\ldots,K_{\max}\}.
\]

The router can determine \(k\) using:

- score thresholds;
- cumulative routing probability;
- routing entropy;
- predicted token difficulty;
- a per-layer or model-wide compute budget.

This strategy includes the anchor as the \(k=0\) outcome and allows more computation for difficult tokens or layers.

**Advantages**

- compute follows estimated utility;
- easy tokens can use the anchor or one expert;
- difficult tokens can combine several experts.

**Trade-offs**

- latency becomes less uniform;
- calibration must jointly control quality and compute;
- capacity planning becomes more difficult.

**Relevant work**

- [Harder Tasks Need More Experts: Dynamic Routing in MoE Models](https://arxiv.org/abs/2403.07652) selects a variable number of experts based on routing confidence.
- [DA-MoE: Towards Dynamic Expert Allocation for Mixture-of-Experts Models](https://arxiv.org/abs/2409.06669) uses token importance to determine the number of active experts.
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) provides a related strategy for allocating a fixed compute budget non-uniformly across tokens and layers.

### 26.3 Router stabilization and distillation

Use a two-stage router lifecycle:

1. train or recalibrate a flexible router;
2. distill its assignments or score distribution into a lightweight router;
3. freeze the distilled router for expert training or deployment.

Possible variants include:

- distilling hard routes;
- distilling calibrated scores;
- freezing only package-level routing while retaining local expert routing;
- adding hysteresis or route-stickiness across adjacent layers.

**Advantages**

- reduces routing fluctuations;
- improves reproducibility;
- separates route discovery from stable expert optimization.

**Trade-offs**

- a frozen router adapts less readily to later packages;
- installation may require a new router version;
- distillation errors can suppress useful routes.

**Relevant work**

- [StableMoE: Stable Routing Strategy for Mixture of Experts](https://arxiv.org/abs/2204.08396) learns routing, distills it into a lightweight router, and freezes the resulting assignments during a later training stage.

### 26.4 Parameter-space expert mixing

Instead of executing two experts and mixing their activations, construct one input-dependent merged expert:

\[
\theta_{\text{merged}}(h)
=
\sum_{k\in S(h)}\alpha_k(h)\theta_k.
\]

The merged parameters are then used for one expert execution. This strategy is especially relevant for low-rank or adapter experts, whose parameters are small enough to combine efficiently.

**Advantages**

- one logical expert execution after merging;
- differentiable routing without discrete activation mixing;
- attractive for compact expert deltas.

**Trade-offs**

- parameter interpolation and activation interpolation are not equivalent for nonlinear MLPs;
- dynamically materializing full merged MLP weights may be expensive;
- independently trained parameters may lie in incompatible regions.

**Relevant work**

- [Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745) constructs a single input-dependent expert by weighting expert parameters.
- [X-LoRA: Mixture of Low-Rank Adapter Experts](https://arxiv.org/abs/2402.07148) performs layerwise token-level mixing of pretrained LoRA experts over a frozen base model.

### 26.5 Fine-grained micro-experts and retrieval

Split coarse MLP experts into a larger number of smaller experts. Activate several micro-experts under a compute budget comparable with one or two full experts.

Possible implementations include:

- smaller intermediate-width MLP experts;
- neuron or block experts;
- product-key retrieval over a large expert table;
- package-local micro-experts with a global specialization shortlist.

**Advantages**

- supports low-level latent specialization;
- increases the number of possible expert combinations;
- may improve parameter efficiency.

**Trade-offs**

- routing and dispatch become more complex;
- small experts may underutilize accelerator kernels;
- package manifests must describe expert granularity and grouping.

**Relevant work**

- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) uses fine-grained expert segmentation and activates more narrowly focused experts.
- [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153) introduces product-key retrieval for selecting from a very large pool of small experts.

### 26.6 Capacity-aware and load-balanced routing

Compare semantic utility routing with explicit capacity management.

Possible strategies include:

- auxiliary load-balancing losses;
- per-expert routing biases updated from observed load;
- expert-choice routing, in which experts select tokens;
- capacity-aware fallback from top-2 to top-1;
- overflow to the anchor.

The framework should distinguish training-time balancing from inference-time routing. A strategy that works for batched training or prompt prefill may be unsuitable for single-token autoregressive decoding.

**Advantages**

- avoids overloaded or unused experts;
- improves accelerator utilization;
- can stabilize multi-expert training.

**Trade-offs**

- load objectives can conflict with semantic utility;
- expert-choice decisions depend on the available token batch;
- balancing rules may reduce deterministic per-token modularity.

**Relevant work**

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) describes auxiliary-loss-free load balancing using routing biases.
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368) lets experts select tokens, providing fixed expert capacity and a variable number of experts per token.

### 26.7 Suggested priority

| Priority | Strategy | Reason |
|---:|---|---|
| 1 | Router stabilization and distillation | Directly supports reproducible modular composition |
| 2 | Dynamic-\(k\) routing | Generalizes the existing top-1/top-2 policies cleanly |
| 3 | Always-on anchor with routed deltas | Provides a strong alternative interpretation of the anchor |
| 4 | Parameter-space mixing | May reduce active computation for compact experts |
| 5 | Fine-grained micro-experts | Expands latent specialization after the basic system works |
| 6 | Capacity-aware routing | Becomes important when training and serving at scale |

These strategies should be evaluated as orthogonal configuration dimensions where possible. For example, a system may combine an always-on anchor, dynamic-\(k\) delta experts, a stabilized router, and capacity-aware overflow without changing the package registry abstraction.
