import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

export const RUNTIME = fileURLToPath(new URL("../runtime/", import.meta.url));
// Resolve once so request deadlines exclude a version manager shim.
export const PYTHON = execFileSync("python3", ["-c", "import sys; print(sys.executable)"],
  { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();

// The production worker and native_asr bridge; only the runner hash is stubbed, so no digest cache is written.
export const NATIVE_WORKER = `
import sys
sys.path.insert(0, ${JSON.stringify(RUNTIME)})
import native_asr, worker
native_asr.file_sha256 = lambda path: "0" * 64
sys.exit(worker.main())
`;
// An Apple speech runner that answers ready, records its pid on the first take, then stays busy
// as native inference does, never reading the input EOF its Python owner's exit leaves.
export const BUSY_RUNNER = `
import json, os, signal, sys, time
if os.environ.get("BUSY_RUNNER_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stdout.write(json.dumps({"kind": "ready", "backend": "apple", "variant": "speech-transcriber"}) + "\\n")
sys.stdout.flush()
sys.stdin.buffer.read(5)
with open(os.environ["BUSY_RUNNER_MARKER"], "w") as target:
    target.write(str(os.getpid()))
while True:
    time.sleep(1)
`;

export async function markerPid(path: string): Promise<number> {
  const deadline = performance.now() + 3000;
  while (performance.now() < deadline) {
    try {
      const pid = Number(await readFile(path, "utf8"));
      if (Number.isInteger(pid) && pid > 0) return pid;
    } catch { /* Wait for the fake operation to start. */ }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("fake process did not start");
}
export function assertExited(pid: number): void {
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
}
export function parentPid(pid: number): number {
  const parent = Number(execFileSync("ps", ["-o", "ppid=", "-p", String(pid)],
    { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim());
  assert.ok(Number.isInteger(parent) && parent > 1);
  return parent;
}
