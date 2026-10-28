import type { APIRoute } from "astro";
import { getOrderedDocs } from "../lib/docs";
import { sitePath } from "../lib/site-path";

/** The same collection drives documentation routes, navigation, and discovery. */
export const GET: APIRoute = async ({ site }) => {
  if (!site) throw new Error("A canonical site URL is required for the sitemap.");

  const docs = await getOrderedDocs();
  const paths = ["", "docs/", ...docs.map((entry) => `docs/${entry.id}/`)];
  const entries = paths.map((path) => {
    const url = new URL(sitePath(path), site).href
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return `  <url><loc>${url}</loc></url>`;
  });

  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries.join("\n")}\n</urlset>\n`,
    { headers: { "Content-Type": "application/xml; charset=utf-8" } },
  );
};
