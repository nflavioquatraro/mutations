import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { relative, resolve, sep } from "node:path";

const distRoot = resolve(import.meta.dirname, "../dist");
const siteBase = "/mutations";
const canonicalOrigin = "https://nflavioquatraro.github.io";

function collectFiles(directory, extension) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = resolve(directory, entry.name);
    if (entry.isDirectory()) return collectFiles(entryPath, extension);
    return entry.name.endsWith(extension) ? [entryPath] : [];
  });
}

function resolveBuiltTarget(url, sourceFile) {
  const path = url.split(/[?#]/, 1)[0];
  assert.ok(
    path === `${siteBase}/` || path.startsWith(`${siteBase}/`),
    `${sourceFile} contains a root URL outside ${siteBase}: ${url}`,
  );

  const outputPath = path.slice(`${siteBase}/`.length);
  const target = path.endsWith("/")
    ? resolve(distRoot, outputPath, "index.html")
    : resolve(distRoot, outputPath);

  assert.ok(existsSync(target), `${sourceFile} links to missing output: ${url}`);
  assert.ok(statSync(target).isFile(), `${url} must resolve to a built file.`);
}

assert.ok(existsSync(distRoot), "Run the Astro build before Pages validation.");

const htmlFiles = collectFiles(distRoot, ".html");
let internalUrlCount = 0;
const canonicalUrls = new Set();
const titles = new Set();
const descriptions = new Set();

for (const htmlFile of htmlFiles) {
  const html = readFileSync(htmlFile, "utf8");
  const head = html.match(/<head\b[^>]*>([\s\S]*?)<\/head>/)?.[1];
  assert.ok(head, `${htmlFile} must include an HTML head.`);
  const canonical = head.match(/<link rel="canonical" href="([^"]+)"/);
  assert.ok(canonical, `${htmlFile} must include a canonical URL.`);
  const outputPath = relative(distRoot, htmlFile).split(sep).join("/");
  const expectedCanonical = `${canonicalOrigin}${siteBase}/${outputPath.replace(/index\.html$/, "")}`;
  assert.equal(
    canonical[1], expectedCanonical,
    `${htmlFile} must have its own exact, trailing-slash canonical URL.`,
  );
  assert.ok(!canonicalUrls.has(canonical[1]), `Duplicate canonical: ${canonical[1]}`);
  canonicalUrls.add(canonical[1]);

  const title = head.match(/<title>([^<]+)<\/title>/)?.[1];
  const description = head.match(/<meta name="description" content="([^"]+)"/)?.[1];
  assert.ok(title?.includes("Mutations"), `${htmlFile} needs a descriptive project title.`);
  assert.ok(description, `${htmlFile} needs a search description.`);
  assert.ok(!titles.has(title), `Duplicate page title: ${title}`);
  assert.ok(!descriptions.has(description), `Duplicate page description: ${description}`);
  titles.add(title);
  descriptions.add(description);
  assert.equal([...html.matchAll(/<h1\b/g)].length, 1, `${htmlFile} needs one primary heading.`);
  assert.equal(
    head.match(/<meta property="og:url" content="([^"]+)"/)?.[1],
    expectedCanonical,
    `${htmlFile} must use its canonical URL in social metadata.`,
  );
  assert.equal(
    head.match(/<link rel="sitemap" type="application\/xml" href="([^"]+)"/)?.[1],
    `${siteBase}/sitemap.xml`,
    `${htmlFile} must advertise the deployed sitemap.`,
  );
  for (const directive of head.matchAll(/<meta name="(?:robots|googlebot)" content="([^"]*)"/g)) {
    assert.doesNotMatch(directive[1], /\b(?:noindex|none)\b/i, `${htmlFile} blocks indexing.`);
  }

  const schemaSource = head.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)?.[1];
  assert.ok(schemaSource, `${htmlFile} needs parseable structured data.`);
  const schema = JSON.parse(schemaSource);
  assert.equal(schema["@context"], "https://schema.org");
  const page = schema["@graph"].find((node) => ["WebPage", "TechArticle"].includes(node["@type"]));
  assert.equal(page?.url, expectedCanonical, `${htmlFile} schema must describe this page.`);
  if (outputPath.startsWith("docs/")) {
    const breadcrumb = schema["@graph"].find((node) => node["@type"] === "BreadcrumbList");
    assert.ok(breadcrumb?.itemListElement.length >= 2, `${htmlFile} needs breadcrumb markup.`);
    assert.equal(breadcrumb.itemListElement.at(-1).item, expectedCanonical);
    breadcrumb.itemListElement.forEach((item, index) => {
      assert.equal(item.position, index + 1);
      const url = new URL(item.item);
      assert.equal(url.origin, canonicalOrigin);
      resolveBuiltTarget(url.pathname, htmlFile);
    });
  } else {
    const software = schema["@graph"].find((node) => node["@type"] === "SoftwareSourceCode");
    assert.equal(software?.codeRepository, "https://github.com/nflavioquatraro/mutations");
  }

  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const url = match[1];
    if (
      url.startsWith("#") ||
      /^[a-z][a-z\d+.-]*:/i.test(url) ||
      url.startsWith("//")
    ) {
      continue;
    }

    internalUrlCount += 1;
    resolveBuiltTarget(url, htmlFile);
  }
}

const docsRoot = resolve(import.meta.dirname, "../src/content/docs");
const docCount = readdirSync(docsRoot).filter((name) => /\.mdx?$/.test(name)).length;
assert.equal(htmlFiles.length, docCount + 2, "All guides, docs index, and homepage must be built.");
assert.ok(internalUrlCount > 0, "The build must contain internal URLs.");

const sitemap = readFileSync(resolve(distRoot, "sitemap.xml"), "utf8");
assert.match(sitemap, /<urlset xmlns="http:\/\/www\.sitemaps\.org\/schemas\/sitemap\/0\.9">/);
const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
assert.equal(new Set(sitemapUrls).size, sitemapUrls.length, "Sitemap URLs must be unique.");
assert.deepEqual(
  sitemapUrls.toSorted(), [...canonicalUrls].toSorted(),
  "The sitemap must cover exactly the canonical homepage, docs index, and every published guide.",
);

process.stdout.write(
  `GitHub Pages build validation passed: ${htmlFiles.length} pages, ${internalUrlCount} internal URLs, unique metadata, structured data, breadcrumbs, and complete sitemap coverage\n`,
);
