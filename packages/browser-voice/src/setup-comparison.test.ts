import assert from "node:assert/strict";
import { setImmediate as settle, setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { SetupComparison, ComparisonBusyError } from "./setup-comparison.ts";
import {
  assertSetupExecutionResult, assertSetupLoadResult, CLEANUP_SETUP_ID, createCanonicalPcm,
  DEFAULT_SETUP_ID, MAX_PCM_BYTES, MAX_TEXT_BYTES, pcmIdentity, SETUP_IDS, sha256, ULTRA_SETUP_ID,
  type ComparisonEvent, type SetupExecution, type SetupExecutionContext, type SetupExecutionResult,
  type SetupExecutor, type SetupId, type SetupLoadResult,
} from "./setup-protocol.ts";
import { assertSetupManifest, getSetup, parseSetupSelection, SETUPS, SETUP_MANIFEST } from "./setup-registry.ts";

const V3 = DEFAULT_SETUP_ID;
const QWEN: SetupId = "qwen/context";
const CPU: SetupId = "sweep/score-2";
function barrier() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => { resolve = r; });
  return { promise, resolve };
}
function output(text = "raw text"): SetupExecutionResult { return { text, processingMs: 3, workerWallMs: 7 }; }

class FakeExecutor implements SetupExecutor {
  readonly calls: string[] = [];
  readonly requests: SetupExecution[] = [];
  readonly used = new Set<SetupId>();
  active = 0;
  maxActive = 0;
  closed = false;
  onLoad?: (id: SetupId, context: SetupExecutionContext) => Promise<void>;
  onExecute?: (request: SetupExecution, context: SetupExecutionContext) => Promise<SetupExecutionResult>;
  onRelease?: (id: SetupId) => Promise<void>;

  async guard<T>(label: string, run: () => Promise<T>): Promise<T> {
    this.calls.push(label);
    this.maxActive = Math.max(this.maxActive, ++this.active);
    try { return await run(); } finally { this.active -= 1; }
  }
  load(id: SetupId, context: SetupExecutionContext): Promise<SetupLoadResult> {
    return this.guard(`load:${id}`, async () => {
      await this.onLoad?.(id, context);
      return { startupMs: this.used.has(id) ? 0 : 2, loadMs: this.used.has(id) ? 0 : 5,
        firstUse: !this.used.has(id) };
    });
  }
  execute(request: SetupExecution, context: SetupExecutionContext): Promise<SetupExecutionResult> {
    return this.guard(`execute:${request.setupId}`, async () => {
      this.requests.push(request);
      this.used.add(request.setupId);
      return this.onExecute ? this.onExecute(request, context) : output();
    });
  }
  release(id: SetupId): Promise<void> {
    return this.guard(`release:${id}`, async () => { await this.onRelease?.(id); this.used.delete(id); });
  }
  async close(): Promise<void> { this.calls.push("close"); this.closed = true; }
}

function harness(options: { now?(): number; idleReleaseMs?: number } = {}) {
  const executor = new FakeExecutor();
  const service = new SetupComparison({ executor, ...options });
  const take = (setupIds: readonly SetupId[] = [V3], extra = {}) => service.compare({
    samples: new Float32Array([0.25, -0, 2]), setupIds, ...extra,
  });
  return { executor, service, take };
}

test("registry pins five exact immutable identities and rejects arbitrary ids or control edits", () => {
  assert.deepEqual(SETUPS.map((s) => s.id), SETUP_IDS);
  assert.equal(SETUP_MANIFEST.defaultSetupId, "mlx-parakeet/v3-greedy");
  assert.equal(SETUP_MANIFEST.idleReleaseMs, 300000);
  assert.deepEqual(SETUPS.map((s) => s.revision), [
    "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15", "b554592c50b2a48471add2daa3d46fa9f00fef5e",
    "ee78f8709547f39bea3b924a16c72403826cac92", "b86784a3a5356cbd126f710b61cd90c5fe5898f1",
    "8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
  ]);
  assert.equal(getSetup(QWEN).controls.prompt, "launch, claude, grok");
  assert.equal(getSetup(CPU).controls.hotwordsScore, 2);
  assert.equal(getSetup(CLEANUP_SETUP_ID).baseSetup, ULTRA_SETUP_ID);
  assert.equal(getSetup(CPU).maxSamples, 960000);
  assertSetupManifest(SETUP_MANIFEST);
  const tampered = structuredClone(SETUP_MANIFEST);
  (tampered.setups[0].controls as Record<string, unknown>).seed = 1;
  assert.throws(() => assertSetupManifest(tampered), /identity\/controls/);
  assert.throws(() => { (getSetup(V3).controls as Record<string, unknown>).seed = 1; }, TypeError);
  assert.throws(() => getSetup("alternative-model"), /unknown/);
  assert.deepEqual(parseSetupSelection([CPU, V3]), [V3, CPU]);
  assert.deepEqual(parseSetupSelection(), [V3]);
  for (const bad of [[], [V3, V3], ["missing"], [{ id: V3, seed: 4 }], "all"]) {
    assert.throws(() => parseSetupSelection(bad), /unique local/);
  }
});

test("the shared setup id order is immutable at runtime", () => {
  const original = [...SETUP_IDS];
  try {
    assert.throws(() => { (SETUP_IDS as unknown as string[])[0] = "custom/model"; }, TypeError);
    assert.deepEqual(parseSetupSelection(), [V3]);
    assert.deepEqual(SETUP_IDS, original);
  } finally {
    if (!Object.isFrozen(SETUP_IDS)) Object.assign(SETUP_IDS, original);
  }
});

test("canonical PCM owns offset bytes exactly, preserves signed zero and never clamps or pads", () => {
  const source = new Float32Array([99, -0, 3.5, -2.25, 99]);
  const pcm = createCanonicalPcm(source.subarray(1, 4));
  const expected = new Uint8Array(12);
  const view = new DataView(expected.buffer);
  [-0, 3.5, -2.25].forEach((value, i) => view.setFloat32(i * 4, value, true));
  assert.deepEqual(pcm.bytes, expected);
  assert.equal(pcm.pcmSha256, sha256(expected));
  assert.equal(pcm.sampleCount, 3);
  assert.equal(pcm.samples.length, 3);
  assert.ok(Object.is(pcm.samples[0], -0));
  source.fill(0);
  assert.deepEqual(pcm.bytes, expected);
  const framed = new Uint8Array(20).fill(255);
  framed.set(expected, 3);
  assert.deepEqual(createCanonicalPcm(framed.subarray(3, 15)).bytes, expected);
  assert.deepEqual(Object.keys(pcmIdentity(pcm)).sort(), ["channels", "pcmBytes", "pcmSha256", "sampleCount", "sampleRate"]);
});

test("canonical PCM rejects nonfinite, partial, empty and oversized payloads before execution", () => {
  for (const bad of [new Float32Array(), new Uint8Array(3), new Uint8Array(MAX_PCM_BYTES + 4),
    new Float32Array([NaN]), new Float32Array([Infinity]), new Float32Array([-Infinity])]) {
    assert.throws(() => createCanonicalPcm(bad), /finite|whole Float32/);
  }
  assert.equal(createCanonicalPcm(new Uint8Array(MAX_PCM_BYTES)).pcmBytes, MAX_PCM_BYTES);
});

test("executor boundary accepts empty raw text and nullable metrics, rejects malformed and unbounded replies", () => {
  assertSetupExecutionResult({ text: "", processingMs: null, workerWallMs: null });
  assertSetupLoadResult({ startupMs: null, loadMs: null, firstUse: true });
  for (const bad of [output("x".repeat(MAX_TEXT_BYTES + 1)), { ...output(), processingMs: NaN },
    { ...output(), workerWallMs: -1 }, { ...output(), rawText: 3 }, { ...output(), decodeMs: Infinity },
    { ...output(), diagnostics: { blob: "x".repeat(65536) } }, { text: "ok", processingMs: 2 }]) {
    assert.throws(() => assertSetupExecutionResult(bad));
  }
  assert.throws(() => assertSetupLoadResult({ startupMs: 0, loadMs: 0 }));
});

test("all selected rows run serially in manifest order and reuse one Ultra decode with exact cleanup input", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  const events: ComparisonEvent[] = [];
  const raw = "\n  launch ☃ \"raw\"  \n";
  h.executor.onExecute = async (r) => output(r.setupId === ULTRA_SETUP_ID ? raw : r.setupId);
  const result = await h.take([...SETUP_IDS].reverse(), {
    onEvent: (event: ComparisonEvent) => events.push(event), expectedPrompt: "must never execute",
  }).done;
  assert.equal(result.status, "complete");
  assert.deepEqual(h.executor.requests.map((r) => r.setupId), SETUP_IDS);
  assert.equal(h.executor.maxActive, 1);
  const acoustic = h.executor.requests.filter((r) => r.operation === "transcribe");
  assert.equal(acoustic.length, 4);
  assert.ok(acoustic.every((r) => r.pcm === acoustic[0].pcm));
  const cleanup = h.executor.requests.at(-1)!;
  assert.equal(cleanup.operation, "cleanup");
  if (cleanup.operation !== "cleanup") throw new Error("missing cleanup request");
  assert.equal(cleanup.A, raw);
  assert.equal(cleanup.B, null);
  assert.equal(cleanup.baseTextSha256, sha256(raw));
  assert.deepEqual(cleanup.base, pcmIdentity(acoustic[0].pcm));
  assert.ok(h.executor.requests.every((r) => !Object.hasOwn(r, "expectedPrompt")));
  assert.ok(result.rows.every((r) => r.status === "complete" && r.timings.deliveryMs === null));
  assert.deepEqual(events.filter((e) => e.type === "row" && e.row.setupId === V3).map((e) =>
    e.type === "row" && e.row.status), ["queued", "loading", "processing", "complete"]);
  const firstComplete = events.findIndex((e) => e.type === "row" && e.row.setupId === V3 && e.row.status === "complete");
  const qwenLoad = events.findIndex((e) => e.type === "row" && e.row.setupId === QWEN && e.row.status === "loading");
  assert.ok(firstComplete < qwenLoad);
  assert.equal(events.at(-1)?.type, "done");
});

test("cleanup alone hides its dependency row, including successful empty Ultra text", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  const events: ComparisonEvent[] = [];
  h.executor.onExecute = async () => output("");
  const result = await h.take([CLEANUP_SETUP_ID], { onEvent: (e: ComparisonEvent) => events.push(e) }).done;
  assert.deepEqual(h.executor.requests.map((r) => r.setupId), [ULTRA_SETUP_ID, CLEANUP_SETUP_ID]);
  assert.equal(result.rows.length, 1);
  assert.equal(result.rows[0].text, "");
  assert.equal(result.rows[0].status, "complete");
  assert.ok(events.some((e) => e.type === "row" && e.row.activeSetupId === ULTRA_SETUP_ID));
  assert.ok(events.every((e) => e.type !== "row" || e.row.setupId === CLEANUP_SETUP_ID));
  const cleanup = h.executor.requests[1];
  assert.ok(cleanup.operation === "cleanup" && cleanup.A === "" && cleanup.B === null);
  await h.take([CLEANUP_SETUP_ID]).done;
  assert.equal(h.executor.requests.filter((r) => r.setupId === ULTRA_SETUP_ID).length, 2,
    "Ultra outcomes never survive into another take");
});

test("Ultra failure skips only cleanup and a later explicit comparison recovers without retries", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  h.executor.onExecute = async (r) => {
    if (r.setupId === ULTRA_SETUP_ID) throw new Error("backend crashed");
    return output();
  };
  const failed = await h.take([ULTRA_SETUP_ID, QWEN, CPU, CLEANUP_SETUP_ID]).done;
  assert.deepEqual(failed.rows.map((r) => r.status), ["error", "complete", "complete", "skipped"]);
  assert.equal(h.executor.requests.filter((r) => r.setupId === ULTRA_SETUP_ID).length, 1);
  assert.ok(!h.executor.calls.includes(`load:${CLEANUP_SETUP_ID}`));
  h.executor.onExecute = async () => output("new take");
  const recovered = await h.take([CLEANUP_SETUP_ID]).done;
  assert.equal(recovered.rows[0].status, "complete");
  assert.equal(h.executor.requests.filter((r) => r.setupId === ULTRA_SETUP_ID).length, 2);
});

test("cleanup errors preserve the completed Ultra row and bounded partial raw output", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  let partial = "partial\n " + "x".repeat(3000);
  h.executor.onExecute = async (r) => {
    if (r.operation === "cleanup") throw Object.assign(new Error("generation failed"), { rawText: partial });
    return output("exact base");
  };
  const result = await h.take([ULTRA_SETUP_ID, CLEANUP_SETUP_ID]).done;
  assert.equal(result.rows[0].text, "exact base");
  assert.equal(result.rows[0].status, "complete");
  assert.equal(result.rows[1].status, "error");
  assert.equal(result.rows[1].text, null);
  assert.equal(result.rows[1].rawText, partial);
  partial = "☃".repeat(3000);
  const oversized = await h.take([CLEANUP_SETUP_ID]).done;
  assert.ok(Buffer.byteLength(oversized.rows[0].rawText!, "utf8") <= MAX_TEXT_BYTES);
  assert.ok(!oversized.rows[0].rawText!.includes("�"), "bounded raw prefix retains whole UTF8 characters");
  assert.equal(oversized.rows[0].error?.rawTextTruncated, true);
});

test("a CPU length refusal leaves the shared take untouched and other selected rows execute", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  const result = await h.service.compare({ samples: new Float32Array(960001), setupIds: [CPU, V3, QWEN] }).done;
  assert.deepEqual(result.rows.map((r) => r.status), ["complete", "complete", "error"]);
  assert.equal(result.rows[2].error?.code, "limit");
  assert.deepEqual(h.executor.requests.map((r) => r.setupId), [V3, QWEN]);
  assert.ok(h.executor.requests.every((r) => r.operation === "transcribe" && r.pcm.sampleCount === 960001));
});

test("load and protocol failures isolate the row and retire it before unrelated work", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  h.executor.onLoad = async (id) => { if (id === V3) throw new Error("assets unavailable"); };
  h.executor.onExecute = async (r) => r.setupId === QWEN ? { ...output(), processingMs: NaN } : output();
  const result = await h.take([V3, QWEN, CPU]).done;
  assert.deepEqual(result.rows.map((r) => r.error?.code), ["load", "protocol", undefined]);
  assert.deepEqual(result.rows.map((r) => r.status), ["error", "error", "complete"]);
  assert.ok(h.executor.calls.indexOf(`release:${V3}`) < h.executor.calls.indexOf(`load:${QWEN}`));
  assert.ok(h.executor.calls.indexOf(`release:${QWEN}`) < h.executor.calls.indexOf(`load:${CPU}`));
});

test("a stopped generation reporting a release error cannot poison remaining rows or the queue", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  h.executor.onExecute = async (r) => {
    if (r.setupId === V3) throw new Error("operation failed");
    return output();
  };
  h.executor.onRelease = async () => { throw new Error("stopped with cleanup diagnostic"); };
  const result = await h.take([V3, QWEN]).done;
  assert.deepEqual(result.rows.map((r) => r.status), ["error", "complete"]);
  assert.match(result.rows[0].error!.message, /operation failed; release: stopped/);
  assert.equal((await h.take([CPU]).done).rows[0].status, "complete");
});

test("one active plus one queued bounds admission, and cancelling queued work frees capacity before load", async (t) => {
  const h = harness();
  const entered = barrier();
  const gate = barrier();
  t.after(async () => { gate.resolve(); await h.service.close(); });
  h.executor.onExecute = async () => { entered.resolve(); await gate.promise; return output(); };
  const first = h.take([V3]);
  await entered.promise;
  const controller = new AbortController();
  const queued = h.take([QWEN], { signal: controller.signal });
  assert.throws(() => h.take([CPU]), ComparisonBusyError);
  controller.abort();
  assert.equal((await queued.done).status, "cancelled");
  const replacement = h.take([CPU]);
  gate.resolve();
  await Promise.all([first.done, replacement.done]);
  assert.deepEqual(h.executor.requests.map((r) => r.setupId), [V3, CPU]);
  assert.equal(h.executor.maxActive, 1);
});

test("queued time ends before load and is excluded from the setup request wall duration", async (t) => {
  let now = 0;
  const h = harness({ now: () => now });
  const entered = barrier();
  const gate = barrier();
  t.after(async () => { gate.resolve(); await h.service.close(); });
  h.executor.onExecute = async (r) => {
    if (r.setupId === V3) { entered.resolve(); await gate.promise; }
    else now += 7;
    return output();
  };
  const first = h.take([V3]);
  await entered.promise;
  now = 10;
  const queued = h.take([QWEN]);
  now = 30;
  gate.resolve();
  await first.done;
  const result = await queued.done;
  assert.equal(result.rows[0].timings.queueMs, 20);
  assert.equal(result.rows[0].timings.requestWallMs, 7);
  assert.equal(result.wallMs, 27);
});

test("active cancellation holds the serial slot until adapter exit and blocks remaining rows", async (t) => {
  const h = harness();
  const entered = barrier();
  const aborted = barrier();
  const exited = barrier();
  t.after(async () => { exited.resolve(); await h.service.close(); });
  h.executor.onExecute = async (r, { signal }) => {
    if (r.setupId === V3) {
      entered.resolve();
      await new Promise<void>((resolve) => signal.addEventListener("abort", () => { aborted.resolve(); resolve(); }, { once: true }));
      await exited.promise;
      throw new Error("child exited after abort");
    }
    return output();
  };
  const first = h.take([V3, QWEN]);
  await entered.promise;
  const second = h.take([CPU]);
  first.cancel();
  await aborted.promise;
  assert.deepEqual(h.executor.calls, [`load:${V3}`, `execute:${V3}`]);
  exited.resolve();
  assert.equal((await first.done).status, "cancelled");
  assert.equal((await second.done).status, "complete");
  assert.deepEqual(h.executor.requests.map((r) => r.setupId), [V3, CPU]);
  assert.equal(h.executor.maxActive, 1);
});

test("cancellation during load prevents execution, and already aborted requests never load", async (t) => {
  const h = harness();
  t.after(() => h.service.close());
  const entered = barrier();
  h.executor.onLoad = async (_id, { signal }) => {
    entered.resolve();
    await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
    throw new Error("load stopped");
  };
  const loading = h.take([V3]);
  await entered.promise;
  loading.cancel();
  assert.equal((await loading.done).status, "cancelled");
  assert.equal(h.executor.requests.length, 0);
  const before = h.executor.calls.length;
  const cancelled = await h.take([QWEN], { signal: AbortSignal.abort() }).done;
  assert.equal(cancelled.status, "cancelled");
  assert.equal(h.executor.calls.length, before);
});

test("close cancels active and queued work, waits for exit, and closes every adapter once", async () => {
  const h = harness();
  const entered = barrier();
  const aborted = barrier();
  const exit = barrier();
  h.executor.onExecute = async (_r, { signal }) => {
    entered.resolve();
    await new Promise<void>((resolve) => signal.addEventListener("abort", () => { aborted.resolve(); resolve(); }, { once: true }));
    await exit.promise;
    throw new Error("stopped");
  };
  const active = h.take();
  await entered.promise;
  const queued = h.take([QWEN]);
  const closing = h.service.close();
  assert.equal(closing, h.service.close());
  await aborted.promise;
  assert.equal(h.executor.closed, false);
  assert.equal((await queued.done).status, "cancelled");
  exit.resolve();
  await closing;
  assert.equal((await active.done).status, "cancelled");
  assert.equal(h.executor.closed, true);
  assert.equal(h.executor.calls.filter((c) => c === "close").length, 1);
  assert.throws(() => h.take(), /closed/);
});

test("phase metrics preserve explicit boundaries, missing values, first use and dependency processing", async (t) => {
  let now = 100;
  const h = harness({ now: () => now });
  t.after(() => h.service.close());
  h.executor.onLoad = async () => { now += 10; };
  h.executor.onExecute = async () => { now += 7; return { ...output(), decodeMs: 1 }; };
  const cold = await h.take([ULTRA_SETUP_ID, CLEANUP_SETUP_ID]).done;
  assert.deepEqual(cold.rows[0].timings, { queueMs: 0, startupMs: 2, loadMs: 5, firstUse: true,
    processingMs: 3, workerWallMs: 7, requestWallMs: 17, serviceWallMs: 17,
    dependencyProcessingMs: null, deliveryMs: null });
  assert.equal(cold.rows[1].timings.dependencyProcessingMs, 3);
  assert.equal(cold.rows[0].decodeMs, 1, "legacy decode timing keeps its own scope");
  assert.equal(cold.wallMs, 34);
  const warm = await h.take([ULTRA_SETUP_ID]).done;
  assert.equal(warm.rows[0].timings.firstUse, false);
  assert.equal(warm.rows[0].timings.loadMs, 0);
});

test("idle release waits for comparison work and its exit before a newly admitted model load", async (t) => {
  let now = 0;
  const h = harness({ now: () => now, idleReleaseMs: 10 });
  const releasing = barrier();
  const released = barrier();
  t.after(async () => { released.resolve(); await h.service.close(); });
  await h.take([V3]).done;
  h.executor.onRelease = async () => { releasing.resolve(); await released.promise; };
  now = 11;
  // The production timer is unrefed; this test keeps the event loop alive while observing it.
  await Promise.race([releasing.promise, delay(500).then(() => { throw new Error("idle release did not run"); })]);
  const queued = h.take([QWEN]);
  await delay(1);
  assert.ok(!h.executor.calls.includes(`load:${QWEN}`));
  released.resolve();
  assert.equal((await queued.done).status, "complete");
  assert.equal((await h.take([V3]).done).rows[0].timings.firstUse, true);
  assert.equal(h.executor.maxActive, 1);
});

test("queued comparisons release expired unused models between jobs and wait for exit", async (t) => {
  let now = 0;
  const h = harness({ now: () => now });
  const entered = barrier();
  const finished = barrier();
  const released = barrier();
  t.after(async () => { finished.resolve(); released.resolve(); await h.service.close(); });
  await h.take([V3]).done;
  h.executor.onExecute = async (request) => {
    if (request.setupId === QWEN) { entered.resolve(); await finished.promise; }
    return output();
  };
  h.executor.onRelease = async (id) => { if (id === V3) await released.promise; };
  const active = h.take([QWEN]);
  await entered.promise;
  const queued = h.take([CPU]);
  now = SETUP_MANIFEST.idleReleaseMs + 1;
  assert.ok(!h.executor.calls.includes(`release:${V3}`), "release waits for the active operation");
  finished.resolve();
  await active.done;
  await settle();
  assert.ok(h.executor.calls.includes(`release:${V3}`), "expiration is checked before queued work");
  assert.ok(!h.executor.calls.includes(`load:${CPU}`), "queued load waits for release exit");
  released.resolve();
  assert.equal((await queued.done).status, "complete");
  assert.ok(h.executor.calls.indexOf(`release:${V3}`) < h.executor.calls.indexOf(`load:${CPU}`));
  assert.ok(!h.executor.calls.includes(`release:${QWEN}`), "recently used models remain resident");
  assert.equal((await h.take([V3]).done).rows[0].timings.firstUse, true);
  assert.equal(h.executor.maxActive, 1);
});
