import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { get } from "node:https";
import { basename, dirname, join } from "node:path";
import type { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { spawn } from "node:child_process";

const MAX_REDIRECTS = 5;
const STDERR_LIMIT = 4_096;
/** Written into the extracted directory last; holds the archive digest. */
export const COMPLETE_MARKER = ".complete";
const UNPINNED_MARKER = "unpinned";

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
 * `<archive>.part` hashing as it goes, verify, and rename. Extract into a
 * temp folder beside `dir`, rename into place, and write the marker last, so
 * an interrupted run never leaves a directory that looks complete. A file set
 * verifies each file the same way, in place under `dir`, then writes the
 * marker; verified files already there are kept, so a repeat after partial
 * failure fetches only what is missing.
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
    await rm(path, { force: true });
  }
  await mkdir(dirname(path), { recursive: true });
  const part = `${path}.part`;
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
    await extractArchive(spec.archive, temp);
    const extracted = join(temp, spec.archiveRoot ?? basename(spec.dir));
    if (!(await stat(extracted).catch(() => undefined))?.isDirectory()) {
      throw new Error(
        `Archive ${spec.archive} has no ${spec.archiveRoot ?? basename(spec.dir)} directory`,
      );
    }
    await rm(spec.dir, { recursive: true, force: true });
    await rename(extracted, spec.dir);
    await writeFile(join(spec.dir, COMPLETE_MARKER), completeMarkerValue(spec), "utf8");
  } finally {
    await rm(temp, { recursive: true, force: true });
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

/** Extract a .tar.bz2 or .tgz archive; tar output is captured, not inherited. */
export async function extractArchive(
  archivePath: string,
  destinationDir: string,
): Promise<void> {
  await mkdir(destinationDir, { recursive: true });
  const flags = tarFlags(archivePath);
  await new Promise<void>((resolve, reject) => {
    const child = spawn("tar", [flags, archivePath, "-C", destinationDir], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    child.stdout.resume();
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString("utf8")).slice(-STDERR_LIMIT);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(
          new Error(`tar exited with code ${code ?? "unknown"}: ${stderr.trim()}`),
        );
      }
    });
  });
}

function tarFlags(archivePath: string): string {
  if (archivePath.endsWith(".tar.bz2")) {
    return "-xjf";
  }
  if (archivePath.endsWith(".tgz") || archivePath.endsWith(".tar.gz")) {
    return "-xzf";
  }
  throw new Error(`Unsupported archive type: ${archivePath}`);
}
