/**
 * Prefixes root-relative links emitted from Markdown with the configured Astro
 * base path. Astro components use `sitePath()` directly; this transformer keeps
 * ordinary Markdown links equally portable without replacing them with HTML.
 */
export default function rehypeBasePath({ base }) {
  const normalizedBase = `/${base.replace(/^\/+|\/+$/g, "")}`;

  const prefixRootPath = (value) => {
    if (
      typeof value !== "string" ||
      !value.startsWith("/") ||
      value.startsWith("//") ||
      value === normalizedBase ||
      value.startsWith(`${normalizedBase}/`)
    ) {
      return value;
    }

    return `${normalizedBase}${value}`;
  };

  return (tree) => {
    const visit = (node) => {
      if (node?.properties) {
        node.properties.href = prefixRootPath(node.properties.href);
        node.properties.src = prefixRootPath(node.properties.src);
        node.properties.poster = prefixRootPath(node.properties.poster);
      }

      node?.children?.forEach(visit);
    };

    visit(tree);
  };
}
