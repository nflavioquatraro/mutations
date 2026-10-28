import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const documentationGroups = [
  "Introduction",
  "Core components",
  "Workflows",
  "Learn by example",
  "Project",
] as const;

const docs = defineCollection({
  loader: glob({
    base: "./src/content/docs",
    pattern: "**/*.{md,mdx}",
  }),
  schema: z.object({
    title: z.string().trim().min(1),
    seo_title: z.string().trim().min(1),
    description: z.string().trim().min(1),
    order: z.number().int().positive(),
    group: z.enum(documentationGroups),
    source: z.array(z.string().trim().min(1)).min(1),
    feature_ids: z
      .array(z.string().regex(/^[A-Z]+-[0-9]{3}$/))
      .default([]),
    last_reviewed: z.coerce.date(),
  }),
});

export const collections = { docs };
