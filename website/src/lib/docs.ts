import { existsSync } from "node:fs";
import { resolve, sep } from "node:path";
import { getCollection, type CollectionEntry } from "astro:content";

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
const repositorySourceBase =
  "https://github.com/nflavioquatraro/mutations/blob/main";

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
