import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

/**
 * The `./stt` entry is vendored into transport-matters byte for byte, so its
 * import closure must stay small and free of native packages: the host
 * injects the sherpa binding.
 */

const SRC = dirname(fileURLToPath(import.meta.url));
const SPECIFIER =
  /(?:\bfrom\s+|\bimport\s*\(?\s*|\brequire\s*\(\s*|<reference\s+path=)["']([^"']+)["']/g;
const FORBIDDEN_FILES = [/^tts\//, /^capture\//, /^engines\/moonshine\.ts$/, /^engines\/sherpa-binding\.ts$/];

function importClosure(entry: string): { files: string[]; packages: string[] } {
  const files = new Set<string>();
  const packages = new Set<string>();
  const pending = [resolve(SRC, entry)];
  for (let path = pending.pop(); path; path = pending.pop()) {
    const file = relative(SRC, path);
    if (files.has(file)) continue;
    files.add(file);
    for (const [, specifier] of readFileSync(path, "utf8").matchAll(SPECIFIER)) {
      if (!specifier) continue;
      if (specifier.startsWith(".")) {
        pending.push(join(dirname(path), specifier));
      } else {
        packages.add(specifier);
      }
    }
  }
  return { files: [...files].sort(), packages: [...packages].sort() };
}

test("stt closure holds the engines and contract only", () => {
  const { files } = importClosure("stt.ts");
  assert.deepEqual(files, [
    "contract.ts",
    "engines/assets.ts",
    "engines/hotwords.ts",
    "engines/result.ts",
    "engines/sherpa-api.ts",
    "engines/sherpa-factory.ts",
    "engines/sherpa-models.ts",
    "engines/sherpa-offline.ts",
    "engines/sherpa.ts",
    "stt.ts",
  ]);
  for (const file of files) {
    for (const forbidden of FORBIDDEN_FILES) {
      assert.doesNotMatch(file, forbidden);
    }
  }
});

test("stt closure imports node builtins only, never a native package", () => {
  const { packages } = importClosure("stt.ts");
  const external = packages.filter((name) => !name.startsWith("node:"));
  assert.deepEqual(external, []);
});

test("the closure scan sees requires, so the CLI binding stays outside", () => {
  const { packages } = importClosure("engines/sherpa-binding.ts");
  assert.ok(packages.includes("sherpa-onnx-node"));
});
