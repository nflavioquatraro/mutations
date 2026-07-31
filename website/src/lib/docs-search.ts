export interface DocumentationSearchHeading {
  depth: 2 | 3;
  text: string;
  slug: string;
}

export interface DocumentationSearchRecord {
  id: string;
  title: string;
  description: string;
  group: string;
  order: number;
  href: string;
  featureIds: string[];
  headings: DocumentationSearchHeading[];
  strategyNames: string[];
}

export type DocumentationSearchMatchKind =
  | "Title"
  | "Heading"
  | "Feature ID"
  | "Strategy";

export interface DocumentationSearchResult {
  id: string;
  title: string;
  description: string;
  group: string;
  order: number;
  href: string;
  matchKind: DocumentationSearchMatchKind;
  matchValue: string;
}

interface ScoredDocumentationSearchResult
  extends DocumentationSearchResult {
  score: number;
}

interface SearchCandidate {
  kind: DocumentationSearchMatchKind;
  value: string;
  href: string;
  exactScore: number;
  prefixScore: number;
  containsScore: number;
}

const strategyNamePattern =
  /^\s{4}(?:approach|representation|strategy):\s*"([^"]+)",?\s*$/gm;

/**
 * Reads the user-facing names that already populate each MDX strategy table.
 *
 * Keeping search terms tied to the rendered table data avoids a second,
 * manually synchronized strategy-name list. The accepted property names match
 * the first-column vocabulary used by the documentation components.
 */
export function extractDocumentationStrategyNames(body: string): string[] {
  const names = Array.from(
    body.matchAll(strategyNamePattern),
    (match) => match[1].trim(),
  );

  return [...new Set(names)];
}

/**
 * Normalizes punctuation-heavy terms such as `POL-002`, `top-2`, and
 * `dynamic-k` without changing their visible labels in search results.
 */
export function normalizeDocumentationSearchValue(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{Mark}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function scoreCandidate(
  candidate: SearchCandidate,
  normalizedQuery: string,
): number | undefined {
  const normalizedCandidate = normalizeDocumentationSearchValue(
    candidate.value,
  );

  if (normalizedCandidate === normalizedQuery) {
    return candidate.exactScore;
  }

  const queryTokens = normalizedQuery.split(" ");
  if (!queryTokens.every((token) => normalizedCandidate.includes(token))) {
    return undefined;
  }

  return normalizedCandidate.startsWith(normalizedQuery)
    ? candidate.prefixScore
    : candidate.containsScore;
}

/**
 * Returns one deterministic best match per page.
 *
 * Exact title and feature-ID matches lead the ranking, while an exact heading
 * match receives a direct fragment link. Stable documentation order resolves
 * equal scores so the result list never depends on collection iteration order.
 */
export function searchDocumentation(
  records: DocumentationSearchRecord[],
  query: string,
  limit = 8,
): DocumentationSearchResult[] {
  const normalizedQuery = normalizeDocumentationSearchValue(query);
  if (!normalizedQuery || limit <= 0) {
    return [];
  }

  const results: ScoredDocumentationSearchResult[] = [];

  for (const record of records) {
    const candidates: SearchCandidate[] = [
      {
        kind: "Title",
        value: record.title,
        href: record.href,
        exactScore: 1400,
        prefixScore: 1120,
        containsScore: 920,
      },
      ...record.featureIds.map((featureId) => ({
        kind: "Feature ID" as const,
        value: featureId,
        href: record.href,
        exactScore: 1350,
        prefixScore: 1080,
        containsScore: 880,
      })),
      ...record.headings.map((heading) => ({
        kind: "Heading" as const,
        value: heading.text,
        href: `${record.href}#${heading.slug}`,
        exactScore: 1300,
        prefixScore: 1040,
        containsScore: 840,
      })),
      ...record.strategyNames.map((strategyName) => ({
        kind: "Strategy" as const,
        value: strategyName,
        href: record.href,
        exactScore: 1250,
        prefixScore: 1000,
        containsScore: 800,
      })),
    ];

    let best:
      | (SearchCandidate & {
          score: number;
        })
      | undefined;

    for (const candidate of candidates) {
      const score = scoreCandidate(candidate, normalizedQuery);
      if (score !== undefined && (!best || score > best.score)) {
        best = { ...candidate, score };
      }
    }

    if (best) {
      results.push({
        id: record.id,
        title: record.title,
        description: record.description,
        group: record.group,
        order: record.order,
        href: best.href,
        matchKind: best.kind,
        matchValue: best.value,
        score: best.score,
      });
    }
  }

  return results
    .toSorted(
      (left, right) =>
        right.score - left.score ||
        left.order - right.order ||
        (left.title < right.title ? -1 : left.title > right.title ? 1 : 0),
    )
    .slice(0, limit)
    .map(({ score: _score, ...result }) => result);
}
