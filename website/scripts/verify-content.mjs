import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { referenceCatalog } from "../src/data/references.ts";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "../..");
const documentationRoot = resolve(repositoryRoot, "website/src/content/docs");
const trackerPath = resolve(repositoryRoot, "feature-implementation-plan.md");
const homepagePath = resolve(repositoryRoot, "website/src/pages/index.astro");
const documentationIndexPath = resolve(
  repositoryRoot,
  "website/src/pages/docs/index.astro",
);
const featureIdPattern = /^[A-Z]+-\d{3}$/;
const arxivIdentifierPattern = /^\d{4}\.\d{4,5}(v\d+)?$/;
const stalePublishedPagePattern =
  /\bplanned \*\*[^*]+\*\* page\b|\bwhen that page is published\b/i;

function parseFrontmatter(source, documentName) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  assert.ok(match, `${documentName} must begin with YAML frontmatter.`);
  return match[1];
}

function parseFrontmatterList(frontmatter, key, documentName) {
  const lines = frontmatter.split(/\r?\n/);
  const keyIndex = lines.findIndex((line) => line === `${key}:`);
  assert.notEqual(
    keyIndex,
    -1,
    `${documentName} must define a "${key}" frontmatter list.`,
  );

  const values = [];
  for (const line of lines.slice(keyIndex + 1)) {
    const item = line.match(/^  -\s+(.+?)\s*$/);
    if (item) {
      values.push(item[1].replace(/^(['"])(.*)\1$/, "$2"));
      continue;
    }
    if (line.trim() === "") continue;
    break;
  }

  return values;
}

function collectTrackedFeatureIds(trackerSource) {
  const featureIds = new Set();

  for (const line of trackerSource.split(/\r?\n/)) {
    const match = line.match(/^\|\s*([A-Z]+-\d{3})\s*\|/);
    if (!match) continue;
    assert.ok(
      !featureIds.has(match[1]),
      `Feature tracker defines "${match[1]}" more than once.`,
    );
    featureIds.add(match[1]);
  }

  assert.ok(featureIds.size > 0, "Feature tracker must define feature IDs.");
  return featureIds;
}

function resolveRepositoryFile(sourcePath, context) {
  assert.ok(
    !isAbsolute(sourcePath),
    `${context} must use a repository-relative source path: ${sourcePath}`,
  );

  const absolutePath = resolve(repositoryRoot, sourcePath);
  const repositoryRelativePath = relative(repositoryRoot, absolutePath);
  const isInsideRepository =
    repositoryRelativePath !== "" &&
    repositoryRelativePath !== ".." &&
    !repositoryRelativePath.startsWith(`..${sep}`) &&
    !isAbsolute(repositoryRelativePath);

  assert.ok(
    isInsideRepository,
    `${context} points outside the repository: ${sourcePath}`,
  );
  assert.ok(existsSync(absolutePath), `${context} is missing: ${sourcePath}`);
  assert.ok(
    statSync(absolutePath).isFile(),
    `${context} must reference a file: ${sourcePath}`,
  );

  return absolutePath;
}

function validateFeatureIds(featureIds, trackedFeatureIds, documentName) {
  const seen = new Set();
  for (const featureId of featureIds) {
    assert.match(
      featureId,
      featureIdPattern,
      `${documentName} has a malformed feature ID: ${featureId}`,
    );
    assert.ok(
      trackedFeatureIds.has(featureId),
      `${documentName} references unknown feature ID "${featureId}".`,
    );
    assert.ok(
      !seen.has(featureId),
      `${documentName} repeats feature ID "${featureId}".`,
    );
    seen.add(featureId);
  }
}

function validateCodeDisclosures(source, documentName) {
  const disclosurePattern =
    /<CodeDisclosure\b([\s\S]*?)>([\s\S]*?)<\/CodeDisclosure>/g;
  let disclosureCount = 0;
  let verifiedCount = 0;

  for (const match of source.matchAll(disclosurePattern)) {
    disclosureCount += 1;
    const [, attributes, body] = match;
    const sourceMatch = attributes.match(/\bsource\s*=\s*"([^"]+)"/);
    const verified =
      /\bverified(?=\s|$)/.test(attributes) ||
      /\bverified\s*=\s*(?:\{true\}|"true")/.test(attributes);

    if (sourceMatch) {
      resolveRepositoryFile(
        sourceMatch[1],
        `${documentName} code disclosure source`,
      );
    }

    if (!verified) continue;
    verifiedCount += 1;

    assert.ok(
      sourceMatch,
      `${documentName} has a Verified code disclosure without a source.`,
    );
    assert.match(
      sourceMatch[1],
      /^(?:examples|tests)\/.+\.py$/,
      `${documentName} Verified code disclosure must reference a Python example or test: ${sourceMatch[1]}`,
    );
    assert.match(
      body,
      /```python\r?\n[\s\S]*?\r?\n```/,
      `${documentName} Verified code disclosure must contain a Python code block.`,
    );
  }

  return { disclosureCount, verifiedCount };
}

function validateArxivReferences(catalog) {
  const identifiers = new Set();
  let arxivCount = 0;

  for (const [referenceId, reference] of Object.entries(catalog)) {
    assert.ok(
      !identifiers.has(reference.identifier),
      `Reference identifier "${reference.identifier}" is defined more than once.`,
    );
    identifiers.add(reference.identifier);

    if (reference.kind !== "arxiv") continue;
    arxivCount += 1;
    assert.match(
      reference.identifier,
      arxivIdentifierPattern,
      `Reference "${referenceId}" has a malformed arXiv identifier.`,
    );
    assert.equal(
      reference.href,
      `https://arxiv.org/abs/${reference.identifier}`,
      `Reference "${referenceId}" must use its canonical direct arXiv URL.`,
    );
  }

  return arxivCount;
}

function validatePublicSiteCopy(homepageSource, documentationIndexSource) {
  assert.match(
    homepageSource,
    /const documentationGuideCount = \(await getOrderedDocs\(\)\)\.length;/,
    "Homepage guide count must derive from the documentation collection.",
  );
  assert.doesNotMatch(
    homepageSource,
    /D\s*\/\s*\d+\s+(?:notes|guides)/i,
    "Homepage must not hard-code the published documentation count.",
  );
  assert.match(
    homepageSource,
    /href="\/docs\/getting-started\/"/,
    "Homepage must provide a direct Getting started pathway.",
  );
  assert.match(
    homepageSource,
    /href="\/docs\/examples\/"/,
    "Homepage must provide a direct Examples pathway.",
  );
  assert.doesNotMatch(
    documentationIndexSource,
    /documentation is being published|upcoming sections/i,
    "Documentation index must not describe the complete collection as upcoming.",
  );
}

function runNegativeContractChecks(trackedFeatureIds) {
  assert.throws(
    () =>
      validateFeatureIds(
        ["UNKNOWN-999"],
        trackedFeatureIds,
        "invalid-feature.mdx",
      ),
    /unknown feature ID/,
  );
  assert.throws(
    () =>
      resolveRepositoryFile(
        "../outside.py",
        "invalid-source.mdx code disclosure source",
      ),
    /outside the repository/,
  );
  assert.throws(
    () =>
      validateCodeDisclosures(
        `<CodeDisclosure summary="Missing provenance" verified>

\`\`\`python
print("untraceable")
\`\`\`

</CodeDisclosure>`,
        "invalid-disclosure.mdx",
      ),
    /without a source/,
  );
  assert.throws(
    () =>
      validateArxivReferences({
        invalid: {
          identifier: "not-an-arxiv-id",
          href: "https://example.com/paper",
          kind: "arxiv",
        },
      }),
    /malformed arXiv identifier/,
  );
}

const trackerSource = readFileSync(trackerPath, "utf8");
const homepageSource = readFileSync(homepagePath, "utf8");
const documentationIndexSource = readFileSync(documentationIndexPath, "utf8");
const trackedFeatureIds = collectTrackedFeatureIds(trackerSource);
const documentationFiles = readdirSync(documentationRoot)
  .filter((fileName) => fileName.endsWith(".mdx"))
  .toSorted();

assert.ok(documentationFiles.length > 0, "Documentation collection is empty.");

let sourcePathCount = 0;
let featureIdCount = 0;
let disclosureCount = 0;
let verifiedCount = 0;

for (const fileName of documentationFiles) {
  const documentPath = resolve(documentationRoot, fileName);
  const source = readFileSync(documentPath, "utf8");
  const frontmatter = parseFrontmatter(source, fileName);
  const sourcePaths = parseFrontmatterList(frontmatter, "source", fileName);
  const featureIds = parseFrontmatterList(
    frontmatter,
    "feature_ids",
    fileName,
  );

  assert.ok(
    sourcePaths.length > 0,
    `${fileName} must reference at least one source file.`,
  );
  sourcePaths.forEach((sourcePath) =>
    resolveRepositoryFile(sourcePath, `${fileName} metadata source`),
  );
  validateFeatureIds(featureIds, trackedFeatureIds, fileName);
  assert.doesNotMatch(
    source,
    stalePublishedPagePattern,
    `${fileName} describes an already published documentation page as planned.`,
  );

  const disclosureReport = validateCodeDisclosures(source, fileName);
  sourcePathCount += sourcePaths.length;
  featureIdCount += featureIds.length;
  disclosureCount += disclosureReport.disclosureCount;
  verifiedCount += disclosureReport.verifiedCount;
}

const arxivCount = validateArxivReferences(referenceCatalog);
validatePublicSiteCopy(homepageSource, documentationIndexSource);
runNegativeContractChecks(trackedFeatureIds);

process.stdout.write(
  [
    "Documentation content validation passed:",
    `${documentationFiles.length} pages`,
    `${sourcePathCount} metadata sources`,
    `${featureIdCount} tracked feature references`,
    `${verifiedCount}/${disclosureCount} verified disclosures`,
    `${arxivCount} canonical arXiv references`,
    "public-site copy synchronized",
    "4 negative contract checks",
  ].join(" ") + "\n",
);
