import assert from "node:assert/strict";
import { test } from "node:test";
import { HttpError } from "./http-guards.ts";
import type { SetupLoadResult } from "./setup-protocol.ts";
import type { PythonWorkerResult, TextCleanupInput } from "./setup-workers.ts";
import { CLEANUP_MODELS, CleanupBusyError, DEFAULT_INSTRUCTION, parseCleanup, TextCleanups, type CleanupModelId,
  type CleanupRequest, type CleanupWorker } from "./text-cleanup.ts";

const S1 = "superwhisper/s1-mini";
const FM = "apple/foundation-models";

/** A worker that loads on first use and spawns a fresh generation after any failure, as PythonSetupWorker does. */
class FakeWorker implements CleanupWorker {
  inputs: TextCleanupInput[] = [];
  generations = 0;
  loaded = false;
  released = 0;
  closed = 0;
  failNext: string | undefined;
  failLoad: string | undefined;
  hold: Promise<void> | undefined;
  #used = false;

  async load(): Promise<SetupLoadResult> {
    if (this.loaded) return { startupMs: 0, loadMs: 0, firstUse: !this.#used };
    this.generations++;
    if (this.failLoad) throw new Error(this.failLoad);
    this.loaded = true;
    this.#used = false;
    return { startupMs: null, loadMs: 40, firstUse: true };
  }

  async cleanText(input: TextCleanupInput, signal?: AbortSignal): Promise<PythonWorkerResult> {
    this.inputs.push(input);
    await this.hold;
    if (signal?.aborted) throw new Error("worker cancelled");
    if (this.failNext) {
      const message = this.failNext;
      this.failNext = undefined;
      this.loaded = false;
      throw new Error(message);
    }
    const firstUse = !this.#used;
    this.#used = true;
    return { text: input.text.replace(/\bum\b ?/g, ""), processingMs: 5, metadata: {}, firstUse,
      startupLoadWallMs: 0, workerWallMs: 7 };
  }

  async release() { this.released++; this.loaded = false; }
  async close() { this.closed++; }
}

function service(workers: Partial<Record<CleanupModelId, CleanupWorker>>, idleReleaseMs?: number) {
  return new TextCleanups({ workers, idleReleaseMs,
    readiness: () => ({ [S1]: { ready: true, sizeBytes: 1 }, [FM]: { ready: true } }) });
}
function request(modelId: CleanupModelId, text = "um so ship it", extra: Partial<CleanupRequest> = {}): CleanupRequest {
  return parseCleanup({ modelId, text, ...extra } as Record<string, unknown>);
}
function refused(value: Record<string, unknown>, status: number) {
  assert.throws(() => parseCleanup(value), (error) => error instanceof HttpError && error.status === status,
    JSON.stringify(value).slice(0, 120));
}

test("the registry declares S1-mini with no settings and Apple Foundation Models with its three, defaults offered", () => {
  assert.deepEqual(CLEANUP_MODELS.map((model) => [model.id, model.label, model.settings.map((setting) => setting.id)]), [
    [S1, "S1-mini by Superwhisper", []],
    [FM, "Apple Foundation Models", ["instruction", "names", "earlier"]],
  ]);
  const fm = parseCleanup({ modelId: FM, text: "x" });
  assert.deepEqual(fm.settings, { instruction: DEFAULT_INSTRUCTION, names: "Grok, Claude, Codex, GPT, TM", earlier: false });
  assert.ok(DEFAULT_INSTRUCTION.length <= 2000);
});

test("requests are bounded: S1-mini takes text only, unknown models and values are refused, overflow is 413", () => {
  refused({ modelId: "mlx-cleanup/qwen-targeted", text: "x" }, 400);
  refused({ modelId: S1, text: "x", settings: {} }, 400);
  refused({ modelId: S1, text: "x", earlier: [] }, 400);
  refused({ modelId: FM, text: "   " }, 400);
  refused({ modelId: FM, text: 3 }, 400);
  refused({ modelId: FM, text: "é".repeat(4097) }, 413);
  refused({ modelId: FM, text: "x", settings: { tone: "calm" } }, 400);
  refused({ modelId: FM, text: "x", settings: { instruction: "x".repeat(2001) } }, 400);
  refused({ modelId: FM, text: "x", settings: { names: "x".repeat(401) } }, 400);
  refused({ modelId: FM, text: "x", earlier: ["a"] }, 400);
  refused({ modelId: FM, text: "x", settings: { earlier: true }, earlier: ["a", "b", "c", "d"] }, 400);
  refused({ modelId: FM, text: "x", settings: { earlier: true }, earlier: [1] }, 400);
  refused({ modelId: FM, text: "x", settings: { earlier: true }, earlier: ["x".repeat(2049), "x".repeat(2048)] }, 413);
  assert.deepEqual(parseCleanup({ modelId: FM, text: "x", settings: { earlier: true }, earlier: ["a", "b"] }).earlier, ["a", "b"]);
});

test("a complete reply carries its timings, and Apple gets its instruction, split names and earlier takes", async () => {
  const s1 = new FakeWorker();
  const fm = new FakeWorker();
  const cleanups = service({ [S1]: s1, [FM]: fm });
  const first = await cleanups.clean(request(S1));
  assert.deepEqual(first, { status: "complete", text: "so ship it", processingMs: 5, workerWallMs: 7, loadMs: 40, firstUse: true });
  assert.deepEqual(s1.inputs, [{ text: "um so ship it" }], "S1-mini gets the text alone");
  const warm = await cleanups.clean(request(S1));
  assert.equal(warm.status === "complete" && warm.loadMs, null, "a warm model reports no load");
  assert.equal(warm.status === "complete" && warm.firstUse, false);
  await cleanups.clean(request(FM, "um tell shawn", { settings: { names: " Sean, ,Michaela ", earlier: true },
    earlier: ["Earlier."] } as Partial<CleanupRequest>));
  assert.deepEqual(fm.inputs, [{ text: "um tell shawn", instruction: DEFAULT_INSTRUCTION, names: ["Sean", "Michaela"],
    earlier: ["Earlier."] }]);
  await cleanups.close();
});

test("a model that is not ready or not configured is skipped without a spawn", async () => {
  const worker = new FakeWorker();
  const cleanups = new TextCleanups({ workers: { [FM]: worker },
    readiness: () => ({ [FM]: { ready: false, reason: "runtime or model files need setup" } }) });
  assert.deepEqual(await cleanups.clean(request(FM)),
    { status: "skipped", code: "unavailable", reason: "runtime or model files need setup" });
  assert.deepEqual(await cleanups.clean(request(S1)),
    { status: "skipped", code: "unavailable", reason: "runtime not configured" });
  assert.equal(worker.generations, 0);
});

test("failures skip with kinds and fixed text only, and the next request uses a fresh worker generation", async () => {
  const worker = new FakeWorker();
  const cleanups = service({ [FM]: worker });
  worker.failLoad = "unavailable: Apple Intelligence is off";
  assert.deepEqual(await cleanups.clean(request(FM)), { status: "skipped", code: "unavailable", reason: "Apple Intelligence is off" });
  worker.failLoad = "Traceback with take text: um so ship it";
  assert.deepEqual(await cleanups.clean(request(FM)), { status: "skipped", code: "load", reason: "the model did not load" });
  worker.failLoad = undefined;
  worker.failNext = "cleanup timed out after 20000 ms";
  assert.deepEqual(await cleanups.clean(request(FM)), { status: "skipped", code: "execution", reason: "timed out after 20 s" });
  worker.failNext = "limit: the take is too long for the model's context";
  assert.deepEqual(await cleanups.clean(request(FM)),
    { status: "skipped", code: "limit", reason: "the take is too long for the model's context" });
  worker.failNext = "model said: um so ship it";
  assert.deepEqual(await cleanups.clean(request(FM)), { status: "skipped", code: "execution", reason: "the model failed" });
  const next = await cleanups.clean(request(FM));
  assert.equal(next.status, "complete");
  assert.equal(next.status === "complete" && next.loadMs, 40, "the next request loads a fresh generation");
  assert.equal(worker.generations, 6);
});

test("one request runs per model and one waits; a third is refused busy, another model is not", async () => {
  const worker = new FakeWorker();
  const other = new FakeWorker();
  const cleanups = service({ [FM]: worker, [S1]: other });
  let open!: () => void;
  worker.hold = new Promise((resolve) => { open = resolve; });
  const first = cleanups.clean(request(FM, "one"));
  const second = cleanups.clean(request(FM, "two"));
  await assert.rejects(cleanups.clean(request(FM, "three")), CleanupBusyError);
  assert.equal((await cleanups.clean(request(S1))).status, "complete", "models do not share a lane");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(worker.inputs.map((input) => input.text), ["one"], "the second waits for the first");
  open();
  assert.deepEqual([(await first).status, (await second).status], ["complete", "complete"]);
  assert.deepEqual(worker.inputs.map((input) => input.text), ["one", "two"]);
  await cleanups.close();
});

test("a cancelled request skips as cancelled, and close cancels and closes every worker", async () => {
  const worker = new FakeWorker();
  const cleanups = service({ [FM]: worker });
  const controller = new AbortController();
  let open!: () => void;
  worker.hold = new Promise((resolve) => { open = resolve; });
  const pending = cleanups.clean(request(FM), controller.signal);
  controller.abort();
  open();
  assert.deepEqual(await pending, { status: "skipped", code: "cancelled", reason: "cleanup cancelled" });
  await cleanups.close();
  assert.equal(worker.closed, 1);
  assert.equal((await cleanups.clean(request(FM))).status, "skipped");
});

test("a warm model is released after the idle period without use", async () => {
  const worker = new FakeWorker();
  const cleanups = service({ [FM]: worker }, 20);
  await cleanups.clean(request(FM));
  await new Promise((resolve) => setTimeout(resolve, 5));
  await cleanups.clean(request(FM));
  assert.equal(worker.released, 0, "use restarts the idle period");
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(worker.released, 1);
  assert.equal(worker.loaded, false);
  await cleanups.close();
});
