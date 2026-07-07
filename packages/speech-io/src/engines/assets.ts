import { createWriteStream } from "node:fs";
import { mkdir, stat } from "node:fs/promises";
import { get } from "node:https";
import { dirname } from "node:path";
import { pipeline } from "node:stream/promises";
import { spawn } from "node:child_process";

export async function hasNonEmptyFile(path: string): Promise<boolean> {
  const existing = await stat(path).catch(() => undefined);
  return Boolean(existing?.size);
}

export async function downloadFile(
  url: string,
  destination: string,
): Promise<void> {
  await mkdir(dirname(destination), { recursive: true });
  await new Promise<void>((resolve, reject) => {
    get(url, (response) => {
      if (
        response.statusCode &&
        response.statusCode >= 300 &&
        response.statusCode < 400 &&
        response.headers.location
      ) {
        response.resume();
        downloadFile(response.headers.location, destination).then(
          resolve,
          reject,
        );
        return;
      }

      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Download failed for ${url}: HTTP ${response.statusCode}`));
        return;
      }

      const file = createWriteStream(destination);
      pipeline(response, file).then(resolve, reject);
    }).on("error", reject);
  });
}

export type AssetSpec = {
  /** Download URL for the .tar.bz2 archive. */
  url: string;
  /** On-disk path for the downloaded archive. */
  archive: string;
  /** Directory the archive extracts into. */
  extractTo: string;
  /** File whose presence proves the asset is already extracted. */
  sentinel: string;
};

/**
 * Idempotent download + extract shared by the STT and TTS model registries:
 * skip when the sentinel exists, reuse a previously downloaded archive,
 * otherwise fetch and unpack.
 */
export async function ensureAsset(spec: AssetSpec): Promise<void> {
  if (await hasNonEmptyFile(spec.sentinel)) {
    return;
  }
  await mkdir(spec.extractTo, { recursive: true });
  if (!(await hasNonEmptyFile(spec.archive))) {
    await downloadFile(spec.url, spec.archive);
  }
  await extractTarBz2(spec.archive, spec.extractTo);
}

export async function extractTarBz2(
  archivePath: string,
  destinationDir: string,
): Promise<void> {
  await mkdir(destinationDir, { recursive: true });
  await new Promise<void>((resolve, reject) => {
    const child = spawn("tar", ["-xjf", archivePath, "-C", destinationDir], {
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`tar exited with code ${code ?? "unknown"}`));
      }
    });
  });
}
