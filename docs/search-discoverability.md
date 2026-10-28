# Search discoverability and publication

Mutations is the project name; Anchored Plug-in MoE is its architecture, and
`anchored-moe` / `anchored_moe` are its Python distribution / import names. Keep
these names connected in public descriptions. Describe implemented behavior
accurately, including the separate LoRA-adapter and Needle-branch workflows.

## Repository About settings

The README, package metadata, and citation file link the source to the website.
GitHub's About settings are separate from committed files. Use these values:

- **Website:** `https://nflavioquatraro.github.io/mutations/`
- **Description:** `Python research framework for modular Mixture-of-Experts (MoE), LoRA adapters, and routed specialization of frozen language models with exact base-model fallback.`
- **Topics:** `mixture-of-experts`, `lora`, `llm`, `expert-routing`, `peft`,
  `model-composition`, `frozen-backbone`, `pytorch`, `huggingface-transformers`,
  `qwen3`, `local-inference`, `wasm`, `python`.

An authenticated administrator can apply the settings with GitHub CLI:

```bash
gh repo edit nflavioquatraro/mutations \
  --description "Python research framework for modular Mixture-of-Experts (MoE), LoRA adapters, and routed specialization of frozen language models with exact base-model fallback." \
  --homepage "https://nflavioquatraro.github.io/mutations/" \
  --add-topic "mixture-of-experts,lora,llm,expert-routing,peft,model-composition,frozen-backbone,pytorch,huggingface-transformers,qwen3,local-inference,wasm,python"
```

Verify the public About section after applying the change. Topics aid discovery
within GitHub; they are not a promise of a particular Google ranking.

## Build and publish

From `website/`, use Node 24 and run:

```bash
npm ci
npm run build
npm run test:search
```

The build checks the content contract, Astro types, internal links, unique page
metadata, exact canonical URLs, structured data, breadcrumbs, and sitemap
coverage. The sitemap is generated from the same collection that generates
the documentation pages; adding a guide updates it automatically.
Each guide has a short `title` for navigation, a descriptive `seo_title` for
its primary heading and search title, and a unique `description`. Keep the
existing document ID when improving titles so published URLs remain stable.

The Pages workflow publishes pushes to `main`. Its manual dispatch can publish
another explicitly selected branch. Check that the intended branch includes
the documented source code before publishing: `feature/system-one` contains
Needle workflows that were not yet in `main` during the October 2026 audit.
The pull-request workflow validates changes without deploying them.
Published guide source links use the deployment's `GITHUB_SHA` so a manually
selected branch links to its own source files. Local builds default to `main`.

After deployment, check the live homepage, a guide, and
`https://nflavioquatraro.github.io/mutations/sitemap.xml`. All must return 200.
Inspect the HTML head for the correct canonical URL, description, social URL,
and JSON-LD. Keep existing guide URLs stable when editing their titles.

## Google Search Console

1. Open [Google Search Console](https://search.google.com/search-console/).
   Select an existing property if available. Otherwise add the **URL-prefix**
   property `https://nflavioquatraro.github.io/mutations/`.
2. For HTML-tag verification, copy only the `content` value of Google's
   `google-site-verification` tag into the GitHub Actions repository variable
   `GOOGLE_SITE_VERIFICATION` and redeploy. The Pages workflow passes it as
   `PUBLIC_GOOGLE_SITE_VERIFICATION`; the shared layout emits the tag only
   when a value is supplied. For local builds, set the same `PUBLIC_` variable.
3. Complete verification in Search Console. Keep the verification token in
   subsequent deployments so ownership verification remains valid.
4. Submit `https://nflavioquatraro.github.io/mutations/sitemap.xml` in Sitemaps.
5. Inspect the homepage and key guides with URL Inspection. Check the indexed
   status, crawl result, and Google-selected canonical; request indexing for
   the homepage and a few important guides after a successful live test.
6. Monitor the Pages and Performance reports. Compare actual query impressions
   and clicks for branded searches and specific MoE/LoRA workflows before
   deciding which tutorials or examples readers need next.

The token is a public ownership-verification value, not a Google account
password. Account sign-in and ownership confirmation still require the owner.
Public `site:` searches are only a rough diagnostic, not a complete index report.

## GitHub Pages and robots.txt

Crawler rules belong at `https://nflavioquatraro.github.io/robots.txt`, the
origin root. A file under `/mutations/robots.txt` would not control crawling for
this project. The root file returned 404 during the audit, which permits
crawling; no project-level robots file is necessary. If an owner-level Pages
site later provides a root robots file, keep `/mutations/` crawlable and add:

```text
Sitemap: https://nflavioquatraro.github.io/mutations/sitemap.xml
```

The sitemap is also linked in every page's HTML head. Submit it directly in
Search Console while there is no origin-level robots file.

## Content and external discovery

Write specific guides around working examples and users' actual questions.
The current entry points cover Qwen3 conversion, MoE expert training, LoRA
adapter routing, and browser inference. Preserve the integration labels and
avoid claims of measured quality or performance without corresponding results.
Link the website from the author's public profile and relevant model or
research pages when those pages exist and the owner chooses to update them.
Useful external references take time to earn; metadata alone cannot create
them.

References:

- [Google Search Essentials](https://developers.google.com/search/docs/essentials)
- [Google's sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/overview)
- [URL Inspection](https://support.google.com/webmasters/answer/9012289)
- [robots.txt scope and HTTP responses](https://developers.google.com/crawling/docs/robots-txt/robots-txt-spec)
- [GitHub repository topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)
