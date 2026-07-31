import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

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

for (const htmlFile of htmlFiles) {
  const html = readFileSync(htmlFile, "utf8");
  const canonical = html.match(/<link rel="canonical" href="([^"]+)"/);
  assert.ok(canonical, `${htmlFile} must include a canonical URL.`);
  assert.ok(
    canonical[1].startsWith(`${canonicalOrigin}${siteBase}/`),
    `${htmlFile} has an invalid Pages canonical URL: ${canonical[1]}`,
  );

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

assert.equal(htmlFiles.length, 17, "The complete static site must be built.");
assert.ok(internalUrlCount > 0, "The build must contain internal URLs.");

process.stdout.write(
  `GitHub Pages build validation passed: ${htmlFiles.length} pages ${internalUrlCount} internal URLs\n`,
);
