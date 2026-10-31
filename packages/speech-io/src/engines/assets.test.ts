import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync } from "node:fs";
import fs from "node:fs/promises";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { Readable } from "node:stream";
import { after, before, test } from "node:test";
import {
  COMPLETE_MARKER,
  completeMarkerValue,
  ensureAsset,
  isAssetComplete,
  type AssetSpec,
  type FetchArchive,
  type FileSetSpec,
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

function deferred(): { promise: Promise<void>; release: () => void } {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => { release = resolve; });
  return { promise, release };
}

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

test("concurrent archive installs of the same missing asset all succeed", async () => {
  const spec = await freshSpec();
  const installs = 4;
  const barrier = deferred();
  let fetches = 0;
  const fetch: FetchArchive = async () => {
    if (++fetches === installs) barrier.release();
    await barrier.promise;
    return Readable.from([archiveBytes]);
  };
  const results = await Promise.allSettled(
    Array.from({ length: installs }, () => ensureAsset(spec, fetch)),
  );
  assert.deepEqual(
    results.map((result) => result.status),
    Array.from({ length: installs }, () => "fulfilled"),
  );
  assert.ok(await isAssetComplete(spec));
  assert.equal(await readFile(join(spec.dir, "tokens.txt"), "utf8"), "a 0\nb 1\n");
  assert.deepEqual((await readdir(join(spec.dir, ".."))).sort(), ["model-x", "model-x.tar.bz2"]);
});

test("concurrent archive replacements stay readable after each ensure resolves", async (context) => {
  const spec = await freshSpec();
  await writeFile(spec.archive, archiveBytes);
  await mkdir(spec.dir);
  await writeFile(join(spec.dir, COMPLETE_MARKER), "older pin");
  await writeFile(join(spec.dir, "tokens.txt"), "old tokens");
  const write = fs.writeFile;
  const rename = fs.rename;
  const staged = deferred();
  const peerPublished = deferred();
  const readerFinished = deferred();
  let markers = 0;
  let failedPublishes = 0;
  let firstAside: string | undefined;
  let raced = false;
  context.mock.method(fs, "writeFile", async (...args: Parameters<typeof fs.writeFile>) => {
    await write(...args);
    if (basename(String(args[0])) === COMPLETE_MARKER) {
      assert.equal(readFileSync(join(spec.dir, "tokens.txt"), "utf8"), "old tokens");
      assert.equal(readFileSync(join(spec.dir, COMPLETE_MARKER), "utf8"), "older pin");
      if (++markers === 2) staged.release();
      await staged.promise;
    }
  });
  context.mock.method(fs, "rename", async (from: string, to: string) => {
    if (from === spec.dir) {
      if (!firstAside) {
        firstAside = to;
        // A observed the stale pin. Let B publish before A moves it aside.
        raced = failedPublishes > 1;
        if (raced) await peerPublished.promise;
      }
      renameSync(from, to);
      return;
    }
    if (to === spec.dir) {
      if (raced && from === join(dirname(firstAside!), basename(spec.dir))) {
        // Hold A's retry until B has loaded files after ensureAsset.
        await readerFinished.promise;
      }
      try {
        renameSync(from, to);
      } catch (error) {
        failedPublishes++;
        throw error;
      }
      peerPublished.release();
      return;
    }
    await rename(from, to);
  });
  syncBuiltinESMExports();
  try {
    const results = await Promise.allSettled(Array.from({ length: 2 }, async () => {
      await ensureAsset(spec, offline);
      try {
        // Native STT and TTS constructors also open paths after ensureAsset.
        return readFileSync(join(spec.dir, "tokens.txt"), "utf8");
      } finally {
        readerFinished.release();
      }
    }));
    assert.deepEqual(results, [
      { status: "fulfilled", value: "a 0\nb 1\n" },
      { status: "fulfilled", value: "a 0\nb 1\n" },
    ]);
    assert.ok(await isAssetComplete(spec));
    assert.deepEqual((await readdir(dirname(spec.dir))).sort(), ["model-x", "model-x.tar.bz2"]);
  } finally {
    context.mock.restoreAll();
    syncBuiltinESMExports();
  }
});

test("a digest mismatch fails closed and leaves nothing behind", async () => {
  const spec = await freshSpec({ sha256: "0".repeat(64) });
  await assert.rejects(
    ensureAsset(spec, serving(() => archiveBytes).fetch),
    /Digest mismatch/,
  );
  assert.ok(!existsSync(spec.archive));
  assert.deepEqual(await readdir(join(spec.archive, "..")), []);
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

test("archive members limit extraction; without them the whole root installs", async () => {
  const source = await mkdtemp(join(work, "members-source-"));
  await mkdir(join(source, "model-x", "espeak-ng-data"), { recursive: true });
  await writeFile(join(source, "model-x", "model.onnx"), "weights");
  await writeFile(join(source, "model-x", "tokens.txt"), "a 0\n");
  await writeFile(join(source, "model-x", "espeak-ng-data", "en_dict"), "dict");
  const bytes = execFileSync("tar", ["-cjf", "-", "-C", source, "model-x"]);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const whole = await freshSpec({ sha256 });
  await ensureAsset(whole, serving(() => bytes).fetch);
  assert.deepEqual(
    (await readdir(whole.dir)).sort(),
    [COMPLETE_MARKER, "espeak-ng-data", "model.onnx", "tokens.txt"],
  );
  const filtered = await freshSpec({ sha256, members: ["model.onnx", "tokens.txt"] });
  await ensureAsset(filtered, serving(() => bytes).fetch);
  assert.deepEqual((await readdir(filtered.dir)).sort(), [COMPLETE_MARKER, "model.onnx", "tokens.txt"]);
  assert.equal(await readFile(join(filtered.dir, COMPLETE_MARKER), "utf8"), sha256);
});

test("optional members extract when present; absent ones are skipped", async () => {
  const source = await mkdtemp(join(work, "optional-source-"));
  await mkdir(join(source, "model-x", "espeak-ng-data"), { recursive: true });
  await writeFile(join(source, "model-x", "tokens.txt"), "a 0\n");
  await writeFile(join(source, "model-x", "LICENSE"), "licence");
  await writeFile(join(source, "model-x", "espeak-ng-data", "LICENSE"), "gpl");
  const bytes = execFileSync("tar", ["-cjf", "-", "-C", source, "model-x"]);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const spec = await freshSpec({ sha256, members: ["tokens.txt"], optionalMembers: ["LICENSE", "MODEL_CARD"] });
  await ensureAsset(spec, serving(() => bytes).fetch);
  assert.deepEqual((await readdir(spec.dir)).sort(), [COMPLETE_MARKER, "LICENSE", "tokens.txt"]);
  assert.equal(await readFile(join(spec.dir, "LICENSE"), "utf8"), "licence");
});

test("a member missing from the archive fails the install", async () => {
  const spec = await freshSpec({ members: ["tokens.txt", "voices.bin"] });
  await assert.rejects(ensureAsset(spec, serving(() => archiveBytes).fetch), /tar exited/);
  assert.ok(!existsSync(spec.dir));
  assert.deepEqual(await readdir(dirname(spec.dir)), ["model-x.tar.bz2"]);
});

test("an unpinned asset counts an existing sentinel as installed", async () => {
  const spec = await freshSpec({ sha256: null });
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, "model.onnx"), "weights");
  await ensureAsset({ ...spec, sentinel: join(spec.dir, "model.onnx") }, offline);
  assert.ok(!existsSync(join(spec.dir, COMPLETE_MARKER)));
});

const FILE_BYTES = { "onnx/encoder.onnx": Buffer.from("weights"), "tokens.txt": Buffer.from("a 0\n") };
type FileName = keyof typeof FILE_BYTES;
const FILE_BASE = "https://example.invalid/model-y/";

async function fileSetSpec(digests: Partial<Record<FileName, string>> = {}): Promise<FileSetSpec> {
  const root = await mkdtemp(join(work, "files-"));
  return {
    dir: join(root, "model-y"),
    files: (Object.keys(FILE_BYTES) as FileName[]).map((name) => ({
      name,
      url: `${FILE_BASE}${name}`,
      sha256: digests[name] ?? createHash("sha256").update(FILE_BYTES[name]).digest("hex"),
      bytes: FILE_BYTES[name].length,
    })),
  };
}

function servingFiles(
  body: (name: FileName) => Buffer = (name) => FILE_BYTES[name],
): { fetch: FetchArchive; calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    fetch: async (url) => {
      calls.push(url);
      return Readable.from([body(url.slice(FILE_BASE.length) as FileName)]);
    },
  };
}

test("a file set downloads each pinned file, then marks every digest", async () => {
  const spec = await fileSetSpec();
  const server = servingFiles();
  await ensureAsset(spec, server.fetch);
  assert.equal(await readFile(join(spec.dir, "onnx/encoder.onnx"), "utf8"), "weights");
  assert.equal(await readFile(join(spec.dir, "tokens.txt"), "utf8"), "a 0\n");
  assert.equal(
    await readFile(join(spec.dir, COMPLETE_MARKER), "utf8"),
    completeMarkerValue(spec),
  );
  await ensureAsset(spec, offline);
  assert.equal(server.calls.length, 2);
});

test("concurrent file set installs of the same missing asset both succeed", async () => {
  const bytes = Buffer.from("fixture model bytes");
  const spec: FileSetSpec = {
    dir: await mkdtemp(join(work, "concurrent-")),
    files: [{
      name: "model.onnx",
      url: `${FILE_BASE}model.onnx`,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    }],
  };
  const barrier = deferred();
  let fetches = 0;
  const fetch: FetchArchive = async () => {
    if (++fetches === 2) barrier.release();
    await barrier.promise;
    return Readable.from([bytes]);
  };
  const results = await Promise.allSettled([
    ensureAsset(spec, fetch),
    ensureAsset(spec, fetch),
  ]);
  assert.deepEqual(results.map((result) => result.status), ["fulfilled", "fulfilled"]);
  assert.ok(await isAssetComplete(spec));
  assert.deepEqual(await readFile(join(spec.dir, "model.onnx")), bytes);
  assert.deepEqual((await readdir(spec.dir)).sort(), [COMPLETE_MARKER, "model.onnx"]);
});

test("a file set keeps verified files and replaces a stale one", async () => {
  const spec = await fileSetSpec();
  await mkdir(join(spec.dir, "onnx"), { recursive: true });
  await writeFile(join(spec.dir, "onnx/encoder.onnx"), "weights");
  await writeFile(join(spec.dir, "tokens.txt"), "stale");
  const server = servingFiles();
  await ensureAsset(spec, server.fetch);
  assert.deepEqual(server.calls, [`${FILE_BASE}tokens.txt`]);
  assert.equal(await readFile(join(spec.dir, "tokens.txt"), "utf8"), "a 0\n");
});

test("a file set replaces a corrupt file of the pinned size", async () => {
  const spec = await fileSetSpec();
  await mkdir(join(spec.dir, "onnx"), { recursive: true });
  await writeFile(join(spec.dir, "onnx/encoder.onnx"), "weight!");
  const server = servingFiles();
  await ensureAsset(spec, server.fetch);
  assert.ok(server.calls.includes(`${FILE_BASE}onnx/encoder.onnx`));
  assert.equal(await readFile(join(spec.dir, "onnx/encoder.onnx"), "utf8"), "weights");
});

test("a file replacement keeps the installed bytes until the new pin is verified", async () => {
  const spec = await fileSetSpec();
  const target = join(spec.dir, "onnx/encoder.onnx");
  await mkdir(join(spec.dir, "onnx"), { recursive: true });
  await writeFile(target, "old pin");
  const replacement: FetchArchive = async () => {
    assert.equal(await readFile(target, "utf8"), "old pin", "the installed file stays during download");
    return Readable.from([Buffer.from("bad pin")]);
  };
  await assert.rejects(ensureAsset(spec, replacement), /Digest mismatch/);
  assert.equal(await readFile(target, "utf8"), "old pin", "a failed replacement preserves installed bytes");
  assert.deepEqual(await readdir(join(target, "..")), ["encoder.onnx"]);
  assert.ok(!existsSync(join(spec.dir, COMPLETE_MARKER)));
  await ensureAsset(spec, servingFiles().fetch);
  assert.equal(await readFile(target, "utf8"), "weights");
});

test("a file set repeat after partial failure fetches only what is missing", async () => {
  const spec = await fileSetSpec();
  const tampered = servingFiles((name) =>
    name === "tokens.txt" ? Buffer.from("tampered") : FILE_BYTES[name],
  );
  await assert.rejects(ensureAsset(spec, tampered.fetch), /Digest mismatch/);
  const server = servingFiles();
  await ensureAsset(spec, server.fetch);
  assert.deepEqual(server.calls, [`${FILE_BASE}tokens.txt`]);
});

test("a file set digest mismatch fails closed without a marker", async () => {
  const spec = await fileSetSpec({ "tokens.txt": "0".repeat(64) });
  await assert.rejects(ensureAsset(spec, servingFiles().fetch), /Digest mismatch/);
  assert.deepEqual((await readdir(spec.dir)).sort(), ["onnx"]);
  assert.deepEqual(await readdir(join(spec.dir, "onnx")), ["encoder.onnx"]);
});

test("a file set drops the marker of another pin before changing files", async () => {
  const spec = await fileSetSpec();
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, COMPLETE_MARKER), "older pin");
  await assert.rejects(ensureAsset(spec, offline), /offline/);
  assert.ok(!existsSync(join(spec.dir, COMPLETE_MARKER)));
});
