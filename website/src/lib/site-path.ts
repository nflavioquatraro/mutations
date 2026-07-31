export const siteBase = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/**
 * Resolves an internal route or public asset beneath Astro's configured base.
 * External URLs and same-page fragments intentionally pass through unchanged.
 */
export function sitePath(path = ""): string {
  if (/^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith("#")) {
    return path;
  }

  return `${siteBase}${path.replace(/^\/+/, "")}`;
}
