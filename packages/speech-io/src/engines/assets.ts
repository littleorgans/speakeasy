import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { get } from "node:https";
import { basename, dirname, join, resolve } from "node:path";
import type { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { spawn } from "node:child_process";

const MAX_REDIRECTS = 5;
const STDERR_LIMIT = 4_096;
/** Written into the extracted directory last; holds the archive digest. */
export const COMPLETE_MARKER = ".complete";
const UNPINNED_MARKER = "unpinned";
const publications = new Map<string, Promise<void>>();

export async function hasNonEmptyFile(path: string): Promise<boolean> {
  const existing = await stat(path).catch(() => undefined);
  return Boolean(existing?.size);
}

/** GET `url`, following redirects, and resolve with the 200 response body. */
export function openDownload(url: string, redirects = 0): Promise<Readable> {
  return new Promise<Readable>((resolve, reject) => {
    get(url, (response) => {
      const location = response.headers.location;
      const status = response.statusCode ?? 0;
      if (status >= 300 && status < 400 && location) {
        response.resume();
        if (redirects >= MAX_REDIRECTS) {
          reject(new Error(`Download failed for ${url}: too many redirects`));
          return;
        }
        openDownload(new URL(location, url).href, redirects + 1).then(
          resolve,
          reject,
        );
        return;
      }
      if (status !== 200) {
        response.resume();
        reject(new Error(`Download failed for ${url}: HTTP ${status}`));
        return;
      }
      resolve(response);
    }).on("error", reject);
  });
}

export async function downloadFile(
  url: string,
  destination: string,
): Promise<void> {
  await mkdir(dirname(destination), { recursive: true });
  await pipeline(await openDownload(url), createWriteStream(destination));
}

export type AssetSpec = {
  /** Download URL for the .tar.bz2 or .tgz archive. */
  url: string;
  /**
   * Expected archive sha256 (hex). Required: pass null only for an asset not
   * yet pinned, which skips verification.
   */
  sha256: string | null;
  /** On-disk path for the archive; a verified archive here needs no network. */
  archive: string;
  /** Directory the asset ends up in. */
  dir: string;
  /** Top directory inside the archive; defaults to the basename of `dir`. */
  archiveRoot?: string;
  /**
   * Paths under the archive root to extract, `/` separated; omit to extract
   * everything. A listed path missing from the archive fails the install.
   */
  members?: readonly string[];
  /** With `members`: paths also extracted when the archive has them, skipped when not. */
  optionalMembers?: readonly string[];
  /**
   * Unpinned assets only: a file whose presence marks an extraction made
   * before completion markers existed.
   */
  sentinel?: string;
};

/** One pinned file of a model published as loose files rather than an archive. */
export type AssetFile = {
  url: string;
  sha256: string;
  /** Path inside `dir`, `/` separated; folders are created as needed. */
  name: string;
  /** Expected size; a file on disk of another size is refetched unhashed. */
  bytes?: number;
};

/** A model published as loose files, each downloaded straight into `dir`. */
export type FileSetSpec = {
  dir: string;
  files: readonly AssetFile[];
};

export type FetchArchive = (url: string) => Promise<Readable>;

/**
 * Idempotent, crash safe download and extract shared by the STT and TTS
 * registries. Done when `dir` holds a completion marker for this digest.
 * Otherwise reuse the archive if its digest matches, else stream it to
 * `<archive>.<unique>.part` hashing as it goes, verify, and rename. Extract into a
 * temp folder beside `dir`, write the marker there, and rename into place, so
 * `dir` appears only complete and a concurrent install that loses the rename
 * finds the asset done. A file set verifies each file the same way, in place
 * under `dir`, then writes the marker; verified files already there are kept,
 * so a repeat after partial failure fetches only what is missing.
 */
export async function ensureAsset(
  spec: AssetSpec | FileSetSpec,
  fetchArchive: FetchArchive = openDownload,
): Promise<void> {
  if (await isAssetComplete(spec)) {
    return;
  }
  if ("files" in spec) {
    await installFileSet(spec, fetchArchive);
    return;
  }
  await ensureVerifiedFile(spec.url, spec.sha256, spec.archive, fetchArchive);
  await installArchive(spec);
}

export async function isAssetComplete(
  spec: AssetSpec | FileSetSpec,
): Promise<boolean> {
  const marker = await readFile(join(spec.dir, COMPLETE_MARKER), "utf8").catch(
    () => undefined,
  );
  if (marker === completeMarkerValue(spec)) {
    return true;
  }
  return (
    !("files" in spec) &&
    spec.sha256 === null &&
    spec.sentinel !== undefined &&
    (await hasNonEmptyFile(spec.sentinel))
  );
}

/** Leave a file at `path` whose digest matches, reusing one already there. */
async function ensureVerifiedFile(
  url: string,
  sha256: string | null,
  path: string,
  fetchArchive: FetchArchive,
  bytes?: number,
): Promise<void> {
  const existing = await stat(path).catch(() => undefined);
  if (existing?.size) {
    if (sha256 === null) {
      return;
    }
    const sizeMatches = bytes === undefined || existing.size === bytes;
    if (sizeMatches && (await sha256File(path)) === sha256) {
      return;
    }
  }
  await mkdir(dirname(path), { recursive: true });
  const part = `${path}.${randomUUID()}.part`;
  try {
    const hash = createHash("sha256");
    await pipeline(
      await fetchArchive(url),
      async function* (source: AsyncIterable<Buffer>) {
        for await (const chunk of source) {
          hash.update(chunk);
          yield chunk;
        }
      },
      createWriteStream(part),
    );
    const actual = hash.digest("hex");
    if (sha256 !== null && actual !== sha256) {
      throw new Error(
        `Digest mismatch for ${url}: expected ${sha256}, received ${actual}`,
      );
    }
    await rename(part, path);
  } finally {
    await rm(part, { force: true });
  }
}

async function installArchive(spec: AssetSpec): Promise<void> {
  const parent = dirname(spec.dir);
  await mkdir(parent, { recursive: true });
  const temp = await mkdtemp(join(parent, `.${basename(spec.dir)}.extract-`));
  try {
    const root = spec.archiveRoot ?? basename(spec.dir);
    await extractArchive(spec.archive, temp, await archiveMembers(spec, root));
    const extracted = join(temp, root);
    if (!(await stat(extracted).catch(() => undefined))?.isDirectory()) {
      throw new Error(`Archive ${spec.archive} has no ${root} directory`);
    }
    await writeFile(join(extracted, COMPLETE_MARKER), completeMarkerValue(spec), "utf8");
    await publishDir(extracted, spec, join(temp, "stale"));
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
}

/** Archive paths to extract: none for the whole archive, else members plus optional ones present. */
async function archiveMembers(spec: AssetSpec, root: string): Promise<string[]> {
  if (spec.members === undefined) return [];
  const optional = spec.optionalMembers?.map((member) => `${root}/${member}`) ?? [];
  const present = optional.length > 0 ? new Set(await listArchive(spec.archive)) : undefined;
  return [
    ...spec.members.map((member) => `${root}/${member}`),
    ...optional.filter((member) => present?.has(member)),
  ];
}

/**
 * Rename a marked extraction into `dir`. A concurrent install of the same
 * asset may win the rename. Serialize publication within this process so a
 * stale check cannot move a peer's completed install out from under readers.
 * A stale `dir` is moved aside whole, never deleted in place.
 */
async function publishDir(source: string, spec: AssetSpec, aside: string): Promise<void> {
  const key = resolve(spec.dir);
  const previous = publications.get(key);
  let release!: () => void;
  const current = new Promise<void>((resolve) => { release = resolve; });
  publications.set(key, current);
  try {
    await previous;
    if (await isAssetComplete(spec)) return;
    for (let attempt = 0; ; attempt++) {
      try {
        await rename(source, spec.dir);
        return;
      } catch (error) {
        if (await isAssetComplete(spec)) return;
        if (attempt > 0) throw error;
      }
      await rename(spec.dir, aside).catch((error: unknown) => {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      });
    }
  } finally {
    release();
    if (publications.get(key) === current) publications.delete(key);
  }
}

async function installFileSet(
  spec: FileSetSpec,
  fetchArchive: FetchArchive,
): Promise<void> {
  // Drop a marker for other pins first: files change in place below.
  await rm(join(spec.dir, COMPLETE_MARKER), { force: true });
  for (const file of spec.files) {
    await ensureVerifiedFile(
      file.url,
      file.sha256,
      join(spec.dir, file.name),
      fetchArchive,
      file.bytes,
    );
  }
  await writeFile(join(spec.dir, COMPLETE_MARKER), completeMarkerValue(spec), "utf8");
}

/** Completion marker content: the archive digest, or one line per pinned file. */
export function completeMarkerValue(spec: AssetSpec | FileSetSpec): string {
  if ("files" in spec) {
    return spec.files.map((file) => `${file.sha256}  ${file.name}\n`).join("");
  }
  return spec.sha256 ?? UNPINNED_MARKER;
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
  }
  return hash.digest("hex");
}

/**
 * Extract a .tar.bz2 or .tgz archive, or only `members` of it when given;
 * tar output is captured, not inherited.
 */
export async function extractArchive(
  archivePath: string,
  destinationDir: string,
  members: readonly string[] = [],
): Promise<void> {
  await mkdir(destinationDir, { recursive: true });
  await runTar([tarFlags("x", archivePath), archivePath, "-C", destinationDir, ...members]);
}

/** Paths of every entry in a .tar.bz2 or .tgz archive. */
async function listArchive(archivePath: string): Promise<string[]> {
  const stdout = await runTar([tarFlags("t", archivePath), archivePath]);
  return stdout.split("\n").filter((line) => line.length > 0);
}

/** Run tar, resolving with its stdout; stderr is kept for the failure message. */
function runTar(args: readonly string[]): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const child = spawn("tar", args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString("utf8")).slice(-STDERR_LIMIT);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(
          new Error(`tar exited with code ${code ?? "unknown"}: ${stderr.trim()}`),
        );
      }
    });
  });
}

function tarFlags(mode: "x" | "t", archivePath: string): string {
  if (archivePath.endsWith(".tar.bz2")) {
    return `-${mode}jf`;
  }
  if (archivePath.endsWith(".tgz") || archivePath.endsWith(".tar.gz")) {
    return `-${mode}zf`;
  }
  throw new Error(`Unsupported archive type: ${archivePath}`);
}
