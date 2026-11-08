import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { PythonSetupWorker } from "./setup-workers.ts";
import { assertExited, BUSY_RUNNER, markerPid, NATIVE_WORKER, parentPid, PYTHON } from "./setup-worker-test-fixtures.ts";

let directory: string;
let workerPath: string;
let runnerPath: string;
before(async () => {
  directory = await mkdtemp(join(tmpdir(), "speakeasy-worker-lifetime-"));
  workerPath = join(directory, "native_worker.py");
  runnerPath = join(directory, "busy_runner");
  await writeFile(workerPath, NATIVE_WORKER);
  await writeFile(runnerPath, `#!${PYTHON}\n${BUSY_RUNNER}`, { mode: 0o755 });
});
after(() => rm(directory, { recursive: true, force: true }));

test("host SIGKILL ends Python and its busy native runner even when the runner ignores SIGTERM", async (t) => {
  const marker = join(directory, "orphan-runner.pid");
  const hostPath = join(directory, "host.mjs");
  await writeFile(hostPath, `
import { PythonSetupWorker } from ${JSON.stringify(new URL("./setup-workers.ts", import.meta.url).href)};
const worker = new PythonSetupWorker(${JSON.stringify({ setupId: "apple/speech", pythonPath: PYTHON,
    workerPath, backendOptions: { runnerPath }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 })});
worker.transcribe(new Float32Array([0.5])).catch(() => { process.exitCode = 1; });
`);
  const host = spawn(process.execPath, [hostPath], { stdio: "ignore",
    env: { ...process.env, BUSY_RUNNER_MARKER: marker, BUSY_RUNNER_IGNORE_TERM: "1" } });
  const closed = new Promise<void>((resolve) => host.once("close", () => resolve()));
  let pythonPid: number | undefined;
  let retired = false;
  t.after(async () => {
    if (host.exitCode === null && host.signalCode === null) host.kill("SIGKILL");
    await closed;
    if (pythonPid !== undefined && !retired) {
      try { process.kill(-pythonPid, "SIGKILL"); } catch { /* Already exited. */ }
    }
  });
  const runnerPid = await markerPid(marker);
  pythonPid = parentPid(runnerPid);
  host.kill("SIGKILL");
  await closed;
  const deadline = performance.now() + 2000;
  while (performance.now() < deadline) {
    try { assertExited(pythonPid); assertExited(runnerPid); retired = true; return; }
    catch { await new Promise((resolve) => setTimeout(resolve, 10)); }
  }
  assertExited(pythonPid);
  assertExited(runnerPid);
});

test("retirement reaches a native runner whose Python group leader already exited", async (t) => {
  const marker = join(directory, "leader-exited-runner.pid");
  process.env.BUSY_RUNNER_MARKER = marker;
  process.env.BUSY_RUNNER_IGNORE_TERM = "1";
  t.after(() => { delete process.env.BUSY_RUNNER_MARKER; delete process.env.BUSY_RUNNER_IGNORE_TERM; });
  const worker = new PythonSetupWorker({ setupId: "apple/speech", pythonPath: PYTHON, workerPath,
    backendOptions: { runnerPath }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 });
  t.after(() => worker.close());
  const rejected = assert.rejects(worker.transcribe(new Float32Array([0.5])), /worker (exited|protocol pipe closed)/);
  const runnerPid = await markerPid(marker);
  const pythonPid = parentPid(runnerPid);
  const started = performance.now();
  process.kill(pythonPid, "SIGKILL");
  await rejected;
  assert.ok(performance.now() - started < 2000, "retirement must not wait for the operation deadline");
  assertExited(pythonPid);
  assertExited(runnerPid);
});

test("retirement ends a busy native runner before the slot is reusable, even one that ignores SIGTERM", async (t) => {
  for (const ignoreTerm of [false, true]) {
    await t.test(ignoreTerm ? "SIGTERM ignored" : "default SIGTERM", async (s) => {
      const marker = join(directory, `busy-runner-${ignoreTerm}.pid`);
      process.env.BUSY_RUNNER_MARKER = marker;
      if (ignoreTerm) process.env.BUSY_RUNNER_IGNORE_TERM = "1";
      else delete process.env.BUSY_RUNNER_IGNORE_TERM;
      s.after(() => { delete process.env.BUSY_RUNNER_MARKER; delete process.env.BUSY_RUNNER_IGNORE_TERM; });
      const worker = new PythonSetupWorker({ setupId: "apple/speech", pythonPath: PYTHON, workerPath,
        backendOptions: { runnerPath }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 });
      s.after(() => worker.close());
      const controller = new AbortController();
      const rejected = assert.rejects(worker.transcribe(new Float32Array([0.5]), controller.signal), /cancelled/);
      const pid = await markerPid(marker);
      controller.abort();
      // The slot is held until the runner's inherited stderr closes, so a surviving runner holds it indefinitely.
      let deadline: NodeJS.Timeout | undefined;
      const retired = await Promise.race([rejected.then(() => true),
        new Promise<false>((resolve) => { deadline = setTimeout(() => resolve(false), 2000); })]);
      clearTimeout(deadline);
      if (!retired) {
        process.kill(pid, "SIGKILL");
        await rejected;
        assert.fail("retirement left the busy native runner alive");
      }
      assertExited(pid);
    });
  }
});

