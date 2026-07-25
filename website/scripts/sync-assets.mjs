import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectDirectory = path.dirname(fileURLToPath(import.meta.url));
const websiteDirectory = path.dirname(projectDirectory);
const repositoryDirectory = path.dirname(websiteDirectory);
const publicDirectory = path.join(websiteDirectory, "public");

await mkdir(publicDirectory, { recursive: true });

await Promise.all([
  copyFile(
    path.join(repositoryDirectory, "model_architecture_image.png"),
    path.join(publicDirectory, "architecture.png"),
  ),
  copyFile(
    path.join(
      repositoryDirectory,
      "preprints",
      "Mutations__A_Framework_for_Bounded_Architectural_Adaptation_in_Shared_Base_Language_Models.pdf",
    ),
    path.join(publicDirectory, "paper.pdf"),
  ),
]);
