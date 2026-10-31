import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { after, before, test } from "node:test";
import { COMPLETE_MARKER, type FetchArchive } from "./assets.ts";
import {
  ensureFileAsset,
  fileAssetMarker,
  type FileAssetSpec,
  type PinnedFile,
} from "./file-assets.ts";

const CONTENT: Record<string, string> = {
  "onnx/model.onnx": "graph",
  "onnx/model.onnx.data": "weights",
  "vocab.pack": "pack",
};

let work: string;

before(async () => {
  work = await mkdtemp(join(tmpdir(), "speak-easy-file-assets-"));
});

after(async () => {
  await rm(work, { recursive: true, force: true });
});

function pinned(path: string): PinnedFile {
  const bytes = Buffer.from(CONTENT[path]!);
  return {
    path,
    url: `https://example.invalid/${path}`,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    bytes: bytes.length,
  };
}

async function freshSpec(): Promise<FileAssetSpec> {
  const root = await mkdtemp(join(work, "root-"));
  return { dir: join(root, "model-x"), files: Object.keys(CONTENT).map(pinned) };
}

function serving(
  body: (path: string) => string = (path) => CONTENT[path]!,
): { fetch: FetchArchive; calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    fetch: async (url) => {
      const path = url.replace("https://example.invalid/", "");
      calls.push(path);
      return Readable.from([Buffer.from(body(path))]);
    },
  };
}

const offline: FetchArchive = async () => {
  throw new Error("offline");
};

test("downloads every file into place and marks the list", async () => {
  const spec = await freshSpec();
  const server = serving();
  await ensureFileAsset(spec, server.fetch);
  for (const [path, content] of Object.entries(CONTENT)) {
    assert.equal(await readFile(join(spec.dir, path), "utf8"), content);
  }
  assert.equal(
    await readFile(join(spec.dir, COMPLETE_MARKER), "utf8"),
    fileAssetMarker(spec),
  );
  await ensureFileAsset(spec, offline);
  assert.equal(server.calls.length, 3);
});

test("a digest mismatch fails closed with no marker or part file", async () => {
  const spec = await freshSpec();
  const server = serving((path) => (path === "vocab.pack" ? "tampered" : CONTENT[path]!));
  await assert.rejects(ensureFileAsset(spec, server.fetch), /Digest mismatch/);
  assert.ok(!existsSync(join(spec.dir, "vocab.pack")));
  assert.ok(!existsSync(join(spec.dir, "vocab.pack.part")));
  assert.ok(!existsSync(join(spec.dir, COMPLETE_MARKER)));
});

test("a repeat after partial failure fetches only what is missing", async () => {
  const spec = await freshSpec();
  await assert.rejects(
    ensureFileAsset(
      spec,
      serving((path) => (path === "vocab.pack" ? "tampered" : CONTENT[path]!)).fetch,
    ),
  );
  const server = serving();
  await ensureFileAsset(spec, server.fetch);
  assert.deepEqual(server.calls, ["vocab.pack"]);
});

test("a corrupt file on disk is replaced", async () => {
  const spec = await freshSpec();
  await mkdir(join(spec.dir, "onnx"), { recursive: true });
  await writeFile(join(spec.dir, "onnx/model.onnx"), "grap!");
  const server = serving();
  await ensureFileAsset(spec, server.fetch);
  assert.ok(server.calls.includes("onnx/model.onnx"));
  assert.equal(await readFile(join(spec.dir, "onnx/model.onnx"), "utf8"), "graph");
});

test("a marker for another file list is not trusted", async () => {
  const spec = await freshSpec();
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, COMPLETE_MARKER), fileAssetMarker({ ...spec, files: [] }));
  await assert.rejects(ensureFileAsset(spec, offline), /offline/);
});
