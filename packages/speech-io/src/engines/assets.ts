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

export type FetchArchive = (url: string) => Promise<Readable>;

/**
 * Idempotent, crash safe download and extract shared by the STT and TTS
 * registries. Done when `dir` holds a completion marker for this digest.
 * Otherwise reuse the archive if its digest matches, else stream it to
 * `<archive>.part` hashing as it goes, verify, and rename. Extract into a
 * temp folder beside `dir`, rename into place, and write the marker last, so
 * an interrupted run never leaves a directory that looks complete.
 */
export async function ensureAsset(
  spec: AssetSpec,
  fetchArchive: FetchArchive = openDownload,
): Promise<void> {
  if (await isAssetComplete(spec)) {
    return;
  }
  await ensureArchive(spec, fetchArchive);
  await installArchive(spec);
}

export async function isAssetComplete(spec: AssetSpec): Promise<boolean> {
  const marker = await readFile(join(spec.dir, COMPLETE_MARKER), "utf8").catch(
    () => undefined,
  );
  if (marker === markerValue(spec)) {
    return true;
  }
  return (
    spec.sha256 === null &&
    spec.sentinel !== undefined &&
    (await hasNonEmptyFile(spec.sentinel))
  );
}

async function ensureArchive(
  spec: AssetSpec,
  fetchArchive: FetchArchive,
): Promise<void> {
  if (await hasNonEmptyFile(spec.archive)) {
    if (spec.sha256 === null) {
      return;
    }
    if ((await sha256File(spec.archive)) === spec.sha256) {
      return;
    }
    await rm(spec.archive, { force: true });
  }
  await mkdir(dirname(spec.archive), { recursive: true });
  const part = `${spec.archive}.part`;
  try {
    const hash = createHash("sha256");
    await pipeline(
      await fetchArchive(spec.url),
      async function* (source: AsyncIterable<Buffer>) {
        for await (const chunk of source) {
          hash.update(chunk);
          yield chunk;
        }
      },
      createWriteStream(part),
    );
    const actual = hash.digest("hex");
    if (spec.sha256 !== null && actual !== spec.sha256) {
      throw new Error(
        `Digest mismatch for ${spec.url}: expected ${spec.sha256}, received ${actual}`,
      );
    }
    await rename(part, spec.archive);
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
    await writeFile(join(spec.dir, COMPLETE_MARKER), markerValue(spec), "utf8");
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
}

function markerValue(spec: AssetSpec): string {
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
