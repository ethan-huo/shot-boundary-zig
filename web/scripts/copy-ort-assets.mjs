import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, "..");
const dist = join(webRoot, "node_modules", "onnxruntime-web", "dist");
const output = join(webRoot, "public", "ort-wasm");

if (!existsSync(dist)) {
  process.exit(0);
}

mkdirSync(output, { recursive: true });

for (const file of readdirSync(dist)) {
  if (file.startsWith("ort-wasm") && (file.endsWith(".wasm") || file.endsWith(".mjs"))) {
    copyFileSync(join(dist, file), join(output, file));
  }
}
