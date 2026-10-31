import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { pipeline } from "node:stream/promises";
import {
  COMPLETE_MARKER,
  openDownload,
  sha256File,
  type FetchArchive,
} from "./assets.ts";

/**
 * Pinned per file download for models published as loose files rather than
 * an archive (Hugging Face exports). The file list counterpart of
 * `assets.ts::ensureAsset`, with the same guarantees: every file is streamed
 * to `<file>.part`, hashed as it goes, verified, then renamed; the
 * completion marker is written last and holds a digest of the whole list, so
 * an interrupted or changed list is never mistaken for an install. Verified
 * files already on disk are kept, so a repeat after partial failure converges.
 */

export type PinnedFile = {
  /** Path inside the asset directory, `/` separated. */
  path: string;
  url: string;
  sha256: string;
  bytes: number;
};

export type FileAssetSpec = {
  dir: string;
  files: readonly PinnedFile[];
};

export async function ensureFileAsset(
  spec: FileAssetSpec,
  fetchFile: FetchArchive = openDownload,
): Promise<void> {
  if (await isFileAssetComplete(spec)) {
    return;
  }
  for (const file of spec.files) {
    await ensureFile(join(spec.dir, file.path), file, fetchFile);
  }
  await writeFile(join(spec.dir, COMPLETE_MARKER), fileAssetMarker(spec), "utf8");
}

export async function isFileAssetComplete(spec: FileAssetSpec): Promise<boolean> {
  const marker = await readFile(join(spec.dir, COMPLETE_MARKER), "utf8").catch(
    () => undefined,
  );
  return marker === fileAssetMarker(spec);
}

async function ensureFile(
  path: string,
  file: PinnedFile,
  fetchFile: FetchArchive,
): Promise<void> {
  const existing = await stat(path).catch(() => undefined);
  if (existing?.size === file.bytes && (await sha256File(path)) === file.sha256) {
    return;
  }
  await mkdir(dirname(path), { recursive: true });
  const part = `${path}.part`;
  try {
    const hash = createHash("sha256");
    await pipeline(
      await fetchFile(file.url),
      async function* (source: AsyncIterable<Buffer>) {
        for await (const chunk of source) {
          hash.update(chunk);
          yield chunk;
        }
      },
      createWriteStream(part),
    );
    const actual = hash.digest("hex");
    if (actual !== file.sha256) {
      throw new Error(
        `Digest mismatch for ${file.url}: expected ${file.sha256}, received ${actual}`,
      );
    }
    await rename(part, path);
  } finally {
    await rm(part, { force: true });
  }
}

/** Completion marker content: a digest of the pinned file list. */
export function fileAssetMarker(spec: FileAssetSpec): string {
  const hash = createHash("sha256");
  for (const file of spec.files) {
    hash.update(`${file.path}\0${file.sha256}\n`);
  }
  return hash.digest("hex");
}
