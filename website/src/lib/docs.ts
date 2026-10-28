import { existsSync } from "node:fs";
import { resolve, sep } from "node:path";
import {
  getCollection,
  render,
  type CollectionEntry,
} from "astro:content";
import {
  extractDocumentationStrategyNames,
  type DocumentationSearchRecord,
} from "./docs-search";
import { sitePath } from "./site-path";

export type DocsEntry = CollectionEntry<"docs">;

export const documentationGroupOrder = [
  "Introduction",
  "Core components",
  "Workflows",
  "Learn by example",
  "Project",
] as const;

export type DocumentationGroup = (typeof documentationGroupOrder)[number];

export interface GroupedDocs {
  group: DocumentationGroup;
  entries: DocsEntry[];
}

export interface DocumentNeighbors {
  previous?: DocsEntry;
  next?: DocsEntry;
}

const executionRoot = process.cwd();
const repositoryRoot = existsSync(
  resolve(executionRoot, "feature-implementation-plan.md"),
)
  ? executionRoot
  : resolve(executionRoot, "..");
// A manually dispatched Pages build may publish a feature branch. Link to the
// exact source commit rather than sending readers to files missing from main.
const sourceRevision = process.env.GITHUB_SHA?.trim() || "main";
const repositorySourceBase =
  `https://github.com/nflavioquatraro/mutations/blob/${encodeURIComponent(sourceRevision)}`;

function validateSourcePath(source: string, documentId: string): void {
  const sourcePath = resolve(repositoryRoot, source);
  const isInsideRepository =
    sourcePath === repositoryRoot ||
    sourcePath.startsWith(`${repositoryRoot}${sep}`);

  if (!isInsideRepository || !existsSync(sourcePath)) {
    throw new Error(
      `Documentation entry "${documentId}" references a missing source: ${source}`,
    );
  }
}

export async function getOrderedDocs(): Promise<DocsEntry[]> {
  const entries = await getCollection("docs");
  const orders = new Map<number, string>();

  for (const entry of entries) {
    const existing = orders.get(entry.data.order);
    if (existing) {
      throw new Error(
        `Documentation entries "${existing}" and "${entry.id}" share order ${entry.data.order}.`,
      );
    }

    orders.set(entry.data.order, entry.id);
    entry.data.source.forEach((source) =>
      validateSourcePath(source, entry.id),
    );
  }

  return entries.toSorted((left, right) => left.data.order - right.data.order);
}

/**
 * Builds the small, serializable index consumed by the static documentation
 * search. Rendering happens at build time so heading fragments always match
 * Astro's generated page IDs.
 */
export async function buildDocumentationSearchIndex(
  entries?: DocsEntry[],
): Promise<DocumentationSearchRecord[]> {
  const orderedEntries = entries ?? (await getOrderedDocs());

  return Promise.all(
    orderedEntries.map(async (entry) => {
      const { headings } = await render(entry);

      return {
        id: entry.id,
        title: entry.data.title,
        description: entry.data.description,
        group: entry.data.group,
        order: entry.data.order,
        href: sitePath(`docs/${entry.id}/`),
        featureIds: entry.data.feature_ids,
        headings: headings
          .filter(
            (
              heading,
            ): heading is typeof heading & {
              depth: 2 | 3;
            } => heading.depth === 2 || heading.depth === 3,
          )
          .map(({ depth, text, slug }) => ({ depth, text, slug })),
        strategyNames: extractDocumentationStrategyNames(entry.body ?? ""),
      };
    }),
  );
}

/**
 * Preserves the documented navigation hierarchy while omitting empty groups.
 *
 * Only validated content entries exist in the collection, so consumers never
 * need a second list of planned routes or disabled navigation placeholders.
 */
export function groupDocumentationEntries(entries: DocsEntry[]): GroupedDocs[] {
  return documentationGroupOrder
    .map((group) => ({
      group,
      entries: entries.filter((entry) => entry.data.group === group),
    }))
    .filter(({ entries: groupEntries }) => groupEntries.length > 0);
}

/**
 * Resolves adjacent published documents from the same order used everywhere
 * else in the documentation interface.
 */
export function getDocumentNeighbors(
  entries: DocsEntry[],
  documentId: string,
): DocumentNeighbors {
  const currentIndex = entries.findIndex((entry) => entry.id === documentId);

  if (currentIndex === -1) {
    throw new Error(
      `Cannot build documentation navigation for unknown entry "${documentId}".`,
    );
  }

  return {
    previous: currentIndex > 0 ? entries[currentIndex - 1] : undefined,
    next:
      currentIndex < entries.length - 1
        ? entries[currentIndex + 1]
        : undefined,
  };
}

export function repositorySourceUrl(source: string): string {
  const encodedSource = source
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");

  return `${repositorySourceBase}/${encodedSource}`;
}
