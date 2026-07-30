import { access, copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectDirectory = path.dirname(fileURLToPath(import.meta.url));
const websiteDirectory = path.dirname(projectDirectory);
const repositoryDirectory = path.dirname(websiteDirectory);
const publicDirectory = path.join(websiteDirectory, "public");

await mkdir(publicDirectory, { recursive: true });

async function syncOptionalSource(source, destination) {
  try {
    await access(source);
    await copyFile(source, destination);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;

    try {
      await access(destination);
    } catch {
      throw new Error(
        `Required website asset is missing from both source and public paths: ${source}`,
        { cause: error },
      );
    }
  }
}

await Promise.all([
  syncOptionalSource(
    path.join(repositoryDirectory, "model_architecture_image.png"),
    path.join(publicDirectory, "architecture.png"),
  ),
  syncOptionalSource(
    path.join(
      repositoryDirectory,
      "preprints",
      "Mutations__A_Framework_for_Bounded_Architectural_Adaptation_in_Shared_Base_Language_Models.pdf",
    ),
    path.join(publicDirectory, "paper.pdf"),
  ),
]);
