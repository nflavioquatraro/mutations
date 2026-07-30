export type ReferenceKind = "arxiv" | "doi" | "standard";

export interface RelatedDocument {
  href: string;
  label: string;
}

export interface ReferenceRecord {
  title: string;
  authors: string;
  year: number;
  kind: ReferenceKind;
  identifier: string;
  href: string;
  sourceLabel: string;
  purpose: string;
  difference: string;
  relatedDocs: readonly RelatedDocument[];
}

export const referenceCatalog = {
  "knowledge-distillation": {
    title: "Distilling the Knowledge in a Neural Network",
    authors: "Geoffrey Hinton, Oriol Vinyals, and Jeff Dean",
    year: 2015,
    kind: "arxiv",
    identifier: "1503.02531",
    href: "https://arxiv.org/abs/1503.02531",
    sourceLabel: "arXiv:1503.02531",
    purpose:
      "Motivates temperature-scaled preservation against a fixed teacher distribution.",
    difference:
      "Mutations consumes precomputed immutable-anchor logits, supports token-level replay masks, and retains a separate causal-language-model objective.",
    relatedDocs: [
      { href: "/docs/training/", label: "Training" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "temperature-scaling": {
    title: "On Calibration of Modern Neural Networks",
    authors: "Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger",
    year: 2017,
    kind: "arxiv",
    identifier: "1706.04599",
    href: "https://arxiv.org/abs/1706.04599",
    sourceLabel: "arXiv:1706.04599",
    purpose:
      "Motivates fitting a post-hoc temperature on held-out predictions.",
    difference:
      "Mutations calibrates independent binary utility heads and applies separate false-activation-constrained thresholds rather than reproducing multiclass calibration.",
    relatedDocs: [
      { href: "/docs/calibration/", label: "Calibration" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "starting-point-regularization": {
    title:
      "Explicit Inductive Bias for Transfer Learning with Convolutional Networks",
    authors: "Xuhong Li, Yves Grandvalet, and Franck Davoine",
    year: 2018,
    kind: "arxiv",
    identifier: "1802.01483",
    href: "https://arxiv.org/abs/1802.01483",
    sourceLabel: "arXiv:1802.01483",
    purpose:
      "Motivates regularizing trainable parameters toward their optimization starting point.",
    difference:
      "Mutations applies drift only to expert-owned parameters and normalizes the squared distance by the expert scalar count.",
    relatedDocs: [{ href: "/docs/training/", label: "Training" }],
  },
  "switch-transformers": {
    title:
      "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
    authors: "William Fedus, Barret Zoph, and Noam Shazeer",
    year: 2021,
    kind: "arxiv",
    identifier: "2101.03961",
    href: "https://arxiv.org/abs/2101.03961",
    sourceLabel: "arXiv:2101.03961",
    purpose:
      "Motivates sparse routing comparisons and the assignment-fraction times mean-probability balancing objective.",
    difference:
      "Mutations retains permanent dense fallback and applies its balancing variant to package-local training rather than a Switch Transformer runtime.",
    relatedDocs: [
      { href: "/docs/training/", label: "Training" },
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  lora: {
    title: "LoRA: Low-Rank Adaptation of Large Language Models",
    authors: "Edward J. Hu et al.",
    year: 2021,
    kind: "arxiv",
    identifier: "2106.09685",
    href: "https://arxiv.org/abs/2106.09685",
    sourceLabel: "arXiv:2106.09685",
    purpose:
      "Defines the frozen-weight low-rank update used for compact anchor-relative expert state.",
    difference:
      "Mutations injects LoRA into a private MLP copy and exposes an explicit anchor-relative delta behind its own immutable-anchor lifecycle.",
    relatedDocs: [
      { href: "/docs/experts/", label: "Experts" },
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "expert-choice": {
    title: "Mixture-of-Experts with Expert Choice Routing",
    authors: "Yanqi Zhou et al.",
    year: 2022,
    kind: "arxiv",
    identifier: "2202.09368",
    href: "https://arxiv.org/abs/2202.09368",
    sourceLabel: "arXiv:2202.09368",
    purpose:
      "Motivates fixed expert token buckets, token-axis ranking, and variable expert counts per token.",
    difference:
      "Mutations returns standalone assignment metadata with explicit integer capacity and anchor fallback, without expert execution or distributed training.",
    relatedDocs: [
      { href: "/docs/training/", label: "Training" },
      { href: "/docs/policies/", label: "Policies" },
    ],
  },
  stablemoe: {
    title: "StableMoE: Stable Routing Strategy for Mixture of Experts",
    authors: "Damai Dai et al.",
    year: 2022,
    kind: "arxiv",
    identifier: "2204.08396",
    href: "https://arxiv.org/abs/2204.08396",
    sourceLabel: "arXiv:2204.08396",
    purpose:
      "Motivates separating route discovery from a lightweight frozen deployment router.",
    difference:
      "Mutations distills fixed-zero-anchor assignments from an immutable composed teacher and keeps cross-layer switching as a separate deterministic rule.",
    relatedDocs: [{ href: "/docs/routers/", label: "Routers" }],
  },
  smear: {
    title: "Soft Merging of Experts with Adaptive Routing",
    authors: "Mohammed Muqeeth, Haokun Liu, and Colin Raffel",
    year: 2023,
    kind: "arxiv",
    identifier: "2306.03745",
    href: "https://arxiv.org/abs/2306.03745",
    sourceLabel: "arXiv:2306.03745",
    purpose:
      "Motivates constructing an input-dependent expert by combining parameters before execution.",
    difference:
      "Mutations provides a two-LoRA correctness backend driven by a frozen pair-weight head and does not reproduce SMEAR training.",
    relatedDocs: [
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/inference/", label: "Inference" },
    ],
  },
  deepseekmoe: {
    title:
      "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
    authors: "Damai Dai et al.",
    year: 2024,
    kind: "arxiv",
    identifier: "2401.06066",
    href: "https://arxiv.org/abs/2401.06066",
    sourceLabel: "arXiv:2401.06066",
    purpose:
      "Motivates fine-grained expert segmentation and the separation of shared from routed computation.",
    difference:
      "Mutations partitions a pinned dense anchor for exact initialization and independently composes plug-ins rather than jointly training the DeepSeekMoE architecture.",
    relatedDocs: [
      { href: "/docs/experts/", label: "Experts" },
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "x-lora": {
    title:
      "X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design",
    authors: "Eric L. Buehler and Markus J. Buehler",
    year: 2024,
    kind: "arxiv",
    identifier: "2402.07148",
    href: "https://arxiv.org/abs/2402.07148",
    sourceLabel: "arXiv:2402.07148",
    purpose:
      "Motivates layerwise, token-dependent composition of frozen LoRA experts.",
    difference:
      "Mutations merges exactly one selected pair's effective updates and does not implement X-LoRA's scaling network or training lifecycle.",
    relatedDocs: [
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/inference/", label: "Inference" },
    ],
  },
  "dynamic-routing": {
    title: "Harder Tasks Need More Experts: Dynamic Routing in MoE Models",
    authors: "Quzhe Huang et al.",
    year: 2024,
    kind: "arxiv",
    identifier: "2403.07652",
    href: "https://arxiv.org/abs/2403.07652",
    sourceLabel: "arXiv:2403.07652",
    purpose:
      "Motivates selecting a variable number of experts from routing confidence.",
    difference:
      "Mutations uses caller-supplied calibrated thresholds, deterministic ties, fixed-zero anchor fallback, and a hard per-token cap rather than a learned dynamic router.",
    relatedDocs: [
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  peer: {
    title: "Mixture of A Million Experts",
    authors: "Xu Owen He",
    year: 2024,
    kind: "arxiv",
    identifier: "2407.04153",
    href: "https://arxiv.org/abs/2407.04153",
    sourceLabel: "arXiv:2407.04153",
    purpose:
      "Defines split-query product-key search over a large expert identifier space.",
    difference:
      "Mutations implements one exact retrieval head over caller-projected queries and stops before query projection, selected-expert execution, aggregation, or package persistence.",
    relatedDocs: [
      { href: "/docs/experts/", label: "Experts" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "loss-free-balancing": {
    title:
      "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts",
    authors: "Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai",
    year: 2024,
    kind: "arxiv",
    identifier: "2408.15664",
    href: "https://arxiv.org/abs/2408.15664",
    sourceLabel: "arXiv:2408.15664",
    purpose:
      "Motivates expert-wise selection biases updated from historical load without auxiliary-loss gradients.",
    difference:
      "Mutations implements a standalone anchored primitive with explicit completed-batch updates and no distributed Trainer integration.",
    relatedDocs: [{ href: "/docs/training/", label: "Training" }],
  },
  "da-moe": {
    title:
      "DA-MoE: Towards Dynamic Expert Allocation for Mixture-of-Experts Models",
    authors: "Maryam Akhavan Aghdam, Hongpeng Jin, and Yanzhao Wu",
    year: 2024,
    kind: "arxiv",
    identifier: "2409.06669",
    href: "https://arxiv.org/abs/2409.06669",
    sourceLabel: "arXiv:2409.06669",
    purpose:
      "Motivates allocating different expert counts according to estimated token importance.",
    difference:
      "Mutations exposes deterministic selection metadata and does not learn token importance, optimize a compute-allocation objective, or execute the selected set.",
    relatedDocs: [
      { href: "/docs/policies/", label: "Policies" },
      { href: "/docs/examples/", label: "Examples" },
    ],
  },
  "deepseek-v3": {
    title: "DeepSeek-V3 Technical Report",
    authors: "DeepSeek-AI et al.",
    year: 2024,
    kind: "arxiv",
    identifier: "2412.19437",
    href: "https://arxiv.org/abs/2412.19437",
    sourceLabel: "arXiv:2412.19437",
    purpose:
      "Documents auxiliary-loss-free expert balancing with routing biases at model scale.",
    difference:
      "Mutations provides a composable local balancing primitive and does not claim the report's distributed training or throughput results.",
    relatedDocs: [{ href: "/docs/training/", label: "Training" }],
  },
  "ridge-regression": {
    title: "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    authors: "Arthur E. Hoerl and Robert W. Kennard",
    year: 1970,
    kind: "doi",
    identifier: "10.1080/00401706.1970.10488634",
    href: "https://doi.org/10.1080/00401706.1970.10488634",
    sourceLabel: "DOI:10.1080/00401706.1970.10488634",
    purpose:
      "Motivates the positive L2 penalty used for deterministic affine router initialization.",
    difference:
      "Mutations leaves the intercept unregularized, solves layer-local multi-output utility targets together, and emits frozen router heads.",
    relatedDocs: [{ href: "/docs/calibration/", label: "Calibration" }],
  },
  "fips-180-4": {
    title: "FIPS PUB 180-4: Secure Hash Standard",
    authors: "National Institute of Standards and Technology",
    year: 2015,
    kind: "doi",
    identifier: "10.6028/NIST.FIPS.180-4",
    href: "https://doi.org/10.6028/NIST.FIPS.180-4",
    sourceLabel: "DOI:10.6028/NIST.FIPS.180-4",
    purpose:
      "Defines SHA-256, used for deterministic artifact identities and integrity checks.",
    difference:
      "Mutations uses the digest for reproducibility and integrity, not as a source of secret randomness.",
    relatedDocs: [
      { href: "/docs/registry/", label: "Registry" },
      { href: "/docs/utils/", label: "Utilities" },
    ],
  },
  "rfc-8785": {
    title: "RFC 8785: JSON Canonicalization Scheme",
    authors: "A. Rundgren, B. Jordan, and S. Erdtman",
    year: 2020,
    kind: "standard",
    identifier: "RFC 8785",
    href: "https://www.rfc-editor.org/rfc/rfc8785",
    sourceLabel: "RFC 8785",
    purpose:
      "Motivates deterministic JSON serialization before hashing.",
    difference:
      "Mutations defines a smaller project-specific canonical JSON contract and does not claim full RFC 8785 conformance.",
    relatedDocs: [{ href: "/docs/utils/", label: "Utilities" }],
  },
} as const satisfies Record<string, ReferenceRecord>;

export type ReferenceId = keyof typeof referenceCatalog;

export const referenceGroups = [
  {
    id: "expert-representations",
    title: "Expert representations",
    referenceIds: ["lora", "deepseekmoe", "peer"],
  },
  {
    id: "routing-and-composition",
    title: "Routing and composition",
    referenceIds: [
      "switch-transformers",
      "expert-choice",
      "stablemoe",
      "smear",
      "x-lora",
      "dynamic-routing",
      "da-moe",
    ],
  },
  {
    id: "training-and-calibration",
    title: "Training and calibration",
    referenceIds: [
      "knowledge-distillation",
      "temperature-scaling",
      "starting-point-regularization",
      "loss-free-balancing",
      "deepseek-v3",
      "ridge-regression",
    ],
  },
  {
    id: "standards-and-reproducibility",
    title: "Standards and reproducibility",
    referenceIds: ["fips-180-4", "rfc-8785"],
  },
] as const satisfies readonly {
  id: string;
  title: string;
  referenceIds: readonly ReferenceId[];
}[];

export type ReferenceGroupId = (typeof referenceGroups)[number]["id"];

const arxivIdentifier = /^\d{4}\.\d{4,5}(v\d+)?$/;
const seenIdentifiers = new Set<string>();

for (const [id, reference] of Object.entries(referenceCatalog)) {
  if (
    reference.kind === "arxiv" &&
    (!arxivIdentifier.test(reference.identifier) ||
      reference.href !== `https://arxiv.org/abs/${reference.identifier}`)
  ) {
    throw new Error(`Reference "${id}" has invalid arXiv metadata.`);
  }

  if (seenIdentifiers.has(reference.identifier)) {
    throw new Error(
      `Reference identifier "${reference.identifier}" is defined more than once.`,
    );
  }

  seenIdentifiers.add(reference.identifier);
}

const groupedReferenceIds = referenceGroups.flatMap((group) =>
  Array.from(group.referenceIds),
);

if (
  groupedReferenceIds.length !== Object.keys(referenceCatalog).length ||
  new Set(groupedReferenceIds).size !== groupedReferenceIds.length
) {
  throw new Error(
    "Every reference must appear in the index exactly once.",
  );
}

export function getReference(referenceId: ReferenceId): ReferenceRecord {
  const reference = referenceCatalog[referenceId];

  if (!reference) {
    throw new Error(`Unknown reference "${referenceId}".`);
  }

  return reference;
}

export function getReferenceGroup(groupId: ReferenceGroupId) {
  const group = referenceGroups.find((candidate) => candidate.id === groupId);

  if (!group) {
    throw new Error(`Unknown reference group "${groupId}".`);
  }

  return group;
}
