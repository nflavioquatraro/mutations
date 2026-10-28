import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";
import rehypeBasePath from "./src/lib/rehype-base-path.mjs";

const siteBase = "/mutations";

export default defineConfig({
  site: "https://nflavioquatraro.github.io",
  base: siteBase,
  output: "static",
  trailingSlash: "always",
  integrations: [mdx()],
  markdown: {
    processor: unified({
      rehypePlugins: [[rehypeBasePath, { base: siteBase }]],
    }),
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
