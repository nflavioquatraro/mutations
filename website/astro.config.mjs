import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  output: "static",
  integrations: [mdx()],
  markdown: {
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
