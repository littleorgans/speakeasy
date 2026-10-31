import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { after, before, test } from "node:test";
import {
  COMPLETE_MARKER,
  ensureAsset,
  type AssetSpec,
  type FetchArchive,
} from "./assets.ts";

let work: string;
let archiveBytes: Buffer;
let digest: string;

before(async () => {
  work = await mkdtemp(join(tmpdir(), "speak-easy-assets-"));
  const source = join(work, "source");
  await mkdir(join(source, "model-x"), { recursive: true });
  await writeFile(join(source, "model-x", "tokens.txt"), "a 0\nb 1\n");
  const archive = join(work, "fixture.tar.bz2");
  execFileSync("tar", ["-cjf", archive, "-C", source, "model-x"]);
  archiveBytes = await readFile(archive);
  digest = createHash("sha256").update(archiveBytes).digest("hex");
});

after(async () => {
  await rm(work, { recursive: true, force: true });
});

async function freshSpec(overrides: Partial<AssetSpec> = {}): Promise<AssetSpec> {
  const root = await mkdtemp(join(work, "root-"));
  return {
    url: "https://example.invalid/model-x.tar.bz2",
    sha256: digest,
    archive: join(root, "model-x.tar.bz2"),
    dir: join(root, "model-x"),
    ...overrides,
  };
}

function serving(bytes: () => Buffer): { fetch: FetchArchive; calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    fetch: async (url) => {
      calls.push(url);
      return Readable.from([bytes()]);
    },
  };
}

const offline: FetchArchive = async () => {
  throw new Error("offline");
};

test("download verifies, installs and marks the digest", async () => {
  const spec = await freshSpec();
  const server = serving(() => archiveBytes);
  await ensureAsset(spec, server.fetch);
  assert.equal(await readFile(join(spec.dir, "tokens.txt"), "utf8"), "a 0\nb 1\n");
  assert.equal(await readFile(join(spec.dir, COMPLETE_MARKER), "utf8"), digest);
  assert.ok(existsSync(spec.archive));
  await ensureAsset(spec, offline);
  assert.equal(server.calls.length, 1);
});

test("a digest mismatch fails closed and leaves nothing behind", async () => {
  const spec = await freshSpec({ sha256: "0".repeat(64) });
  await assert.rejects(
    ensureAsset(spec, serving(() => archiveBytes).fetch),
    /Digest mismatch/,
  );
  assert.ok(!existsSync(spec.archive));
  assert.ok(!existsSync(`${spec.archive}.part`));
  assert.ok(!existsSync(spec.dir));
});

test("a partial archive is replaced, never extracted", async () => {
  const spec = await freshSpec();
  await mkdir(join(spec.archive, ".."), { recursive: true });
  await writeFile(spec.archive, archiveBytes.subarray(0, archiveBytes.length / 2));
  const server = serving(() => archiveBytes);
  await ensureAsset(spec, server.fetch);
  assert.equal(server.calls.length, 1);
  assert.equal(await readFile(join(spec.dir, COMPLETE_MARKER), "utf8"), digest);
});

test("a seeded archive installs offline", async () => {
  const spec = await freshSpec();
  await mkdir(join(spec.archive, ".."), { recursive: true });
  await writeFile(spec.archive, archiveBytes);
  await ensureAsset(spec, offline);
  assert.equal(await readFile(join(spec.dir, "tokens.txt"), "utf8"), "a 0\nb 1\n");
});

test("offline with no archive fails", async () => {
  const spec = await freshSpec();
  await assert.rejects(ensureAsset(spec, offline), /offline/);
  assert.ok(!existsSync(spec.dir));
});

test("a directory without the completion marker is reinstalled", async () => {
  const spec = await freshSpec();
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, "stale.onnx"), "half written");
  await ensureAsset(spec, serving(() => archiveBytes).fetch);
  assert.ok(!existsSync(join(spec.dir, "stale.onnx")));
  assert.equal(await readFile(join(spec.dir, COMPLETE_MARKER), "utf8"), digest);
});

test("a failed extract leaves no directory, marker or temp folder", async () => {
  const spec = await freshSpec({ archiveRoot: "not-in-archive" });
  await assert.rejects(
    ensureAsset(spec, serving(() => archiveBytes).fetch),
    /no not-in-archive directory/,
  );
  assert.ok(!existsSync(spec.dir));
  assert.deepEqual(await readdir(join(spec.dir, "..")), ["model-x.tar.bz2"]);
});

test("an unpinned asset counts an existing sentinel as installed", async () => {
  const spec = await freshSpec({ sha256: null });
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, "model.onnx"), "weights");
  await ensureAsset({ ...spec, sentinel: join(spec.dir, "model.onnx") }, offline);
  assert.ok(!existsSync(join(spec.dir, COMPLETE_MARKER)));
});
