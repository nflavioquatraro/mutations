import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  searchDocumentation,
} from "../src/lib/docs-search.ts";

const siteBase = "/mutations";

const indexHtml = await readFile(
  new URL("../dist/docs/index.html", import.meta.url),
  "utf8",
);
const indexMatch = indexHtml.match(
  /<script id="documentation-search-index" type="application\/json">([\s\S]*?)<\/script>/,
);

assert(indexMatch, "the generated documentation page must embed a search index");
assert(
  indexHtml.includes('data-docs-search data-enhanced="false"'),
  "the static page must preserve the non-enhanced search state",
);
assert(
  indexHtml.includes("<noscript>"),
  "the static page must explain the no-JavaScript directory fallback",
);

const records = JSON.parse(indexMatch[1]);
assert.equal(records.length, 15, "every content page must be indexed");
assert.equal(
  new Set(records.map(({ href }) => href)).size,
  records.length,
  "search records must have unique routes",
);
assert(
  records.every(
    ({ headings, featureIds, strategyNames }) =>
      Array.isArray(headings) &&
      Array.isArray(featureIds) &&
      Array.isArray(strategyNames),
  ),
  "every record must expose all searchable fields",
);
assert(
  records.some(({ strategyNames }) => strategyNames.length > 0),
  "the index must include strategy-table terms",
);

for (const record of records) {
  assert.ok(
    record.href.startsWith(`${siteBase}/docs/`),
    `${record.href} must include the GitHub Pages project base`,
  );
  const outputRoute = record.href.slice(siteBase.length);
  const pageHtml = await readFile(
    new URL(`../dist${outputRoute}index.html`, import.meta.url),
    "utf8",
  );

  for (const heading of record.headings) {
    assert(
      pageHtml.includes(`id="${heading.slug}"`),
      `${record.href} must contain the indexed heading #${heading.slug}`,
    );
  }
}

const cases = [
  {
    query: "Registry",
    title: "Registry",
    matchKind: "Title",
    href: `${siteBase}/docs/registry/`,
  },
  {
    query: "POL-002",
    title: "Policies",
    matchKind: "Feature ID",
    href: `${siteBase}/docs/policies/`,
  },
  {
    query: "anchor-veto top-2",
    title: "Policies",
    matchKind: "Strategy",
    href: `${siteBase}/docs/policies/`,
  },
  {
    query: "Product-key micro-expert retrieval",
    title: "End-to-end examples",
    matchKind: "Heading",
    hrefPrefix: `${siteBase}/docs/examples/#`,
  },
];

for (const expected of cases) {
  const firstRun = searchDocumentation(records, expected.query);
  const secondRun = searchDocumentation(records, expected.query);

  assert(firstRun.length > 0, `${expected.query} must return a result`);
  assert.deepEqual(
    firstRun,
    secondRun,
    `${expected.query} ranking must be deterministic`,
  );
  assert.equal(firstRun[0].title, expected.title);
  assert.equal(firstRun[0].matchKind, expected.matchKind);

  if (expected.href) {
    assert.equal(firstRun[0].href, expected.href);
  } else {
    assert(firstRun[0].href.startsWith(expected.hrefPrefix));
  }
}

assert.deepEqual(searchDocumentation(records, "not-a-real-framework-term"), []);

console.log(
  `Validated ${records.length} search records and ${cases.length} representative queries.`,
);
