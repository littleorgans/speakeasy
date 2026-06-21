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
