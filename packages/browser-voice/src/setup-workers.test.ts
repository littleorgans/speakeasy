import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { encodeWavFloat32 } from "../../speech-io/src/bench/wav.ts";
import { pcmDigest, PythonSetupWorker, SetupWorkers, MAX_WORKER_PCM_BYTES, MAX_WORKER_TEXT_BYTES } from "./setup-workers.ts";
import { readPinnedPick } from "./voice-references.ts";
import { createWorkerVoiceSynth } from "./voice-workers.ts";
import { createCanonicalPcm, sha256, type SetupExecution, type SetupLoadResult } from "./setup-protocol.ts";
import { CPU_PRESETS, type CpuReady } from "./transcriber.ts";
import { DEFAULT_NAMES_SNAPSHOT, namesSnapshot } from "./names.ts";

let PYTHON: string;
const RUNTIME = fileURLToPath(new URL("../runtime/", import.meta.url));
let directory: string;
let fixture: string;

// This launches the production framing loop and fd isolation with fake backends.
const FAKE = `
import sys, os, json, hashlib, struct, time, signal
sys.path.insert(0, ${JSON.stringify(RUNTIME)})
import worker
mode = ""
original_write = worker.write_frame
class Fake:
    def __init__(self, options):
        global mode
        self.options = options
        self.calls = 0
        mode = options.get("mode", "")
        # A voice copy fake takes settings and a reference; every other voice refuses both in the worker.
        self.takes_settings = self.takes_reference = mode == "copy"
        # A text cleanup fake takes context only for the model whose backend does.
        self.takes_context = options.get("setupId") == "apple/foundation-models"
    def mark(self):
        path = self.options.get("marker")
        if path:
            with open(path, "w") as target:
                target.write(str(os.getpid()))
    def load(self):
        print("python diagnostic", flush=True)
        os.write(1, b"native diagnostic\\n")
        if mode == "missingReady":
            self.mark()
            time.sleep(30)
        if mode == "loadError":
            raise RuntimeError("load failed")
        if mode in ("sendRace", "exitAfterReady"):
            self.mark()
        return {"revision": "pinned", "setupId": self.options.get("setupId"), "loadMs": 2, "pid": os.getpid()}
    def transcribe(self, samples, settings=None):
        self.calls += 1
        if mode == "ignoreTerm":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.mark()
        if mode in ("blocked", "ignoreTerm"):
            time.sleep(30)
        if mode == "crash":
            os._exit(12)
        if mode == "error":
            error = RuntimeError("backend failed")
            error.rawText = "partial\\n"
            raise error
        if mode == "stderr":
            os.write(2, b"x" * 50000)
        raw = samples.tobytes()
        return {"text": hashlib.sha256(raw).hexdigest(), "processingMs": 3, "decodeMs": 1,
                "diagnostics": {"hex": raw.hex(), "count": len(samples), "pid": os.getpid(), "calls": self.calls,
                                **({"settings": settings} if settings is not None else {}),
                                **({"names": self.options["names"]} if "names" in self.options else {})}}
    def cleanup(self, text, **context):
        if context:
            return {"text": json.dumps({"text": text, **context}, sort_keys=True), "processingMs": 1}
        return {"text": text, "rawText": text, "processingMs": 0}
    def feed(self, samples, start, settings):
        if mode == "badPartial":
            return {"stable": "", "tentative": "", "processingMs": -1}
        self.fed = (0 if start else getattr(self, "fed", 0)) + len(samples)
        chosen = " " + json.dumps(settings, sort_keys=True) if settings else ""
        return {"stable": " heard %d" % self.fed, "tentative": " maybe" + chosen, "processingMs": 2}
    def speak(self, text, voice, speed, settings=None, reference=None):
        if mode == "copy":
            pcm = struct.pack("<4f", (settings or {}).get("seed", -1), reference["sampleRate"] if reference else -1,
                              len(reference["samples"]) if reference else -1, len((reference or {}).get("transcript") or ""))
            return {"samples": pcm, "sampleRate": 24000, "processingMs": 1}
        if mode == "badAudio":
            return {"samples": b"\\0\\0\\xc0\\x7f", "sampleRate": 24000, "processingMs": 1}
        pcm = struct.pack("<4f", len(text), speed, -0.0, len(voice))
        return {"samples": pcm, "sampleRate": 24000, "processingMs": 2}
    def close(self):
        path = self.options.get("closeMarker")
        if path:
            with open(path, "w") as target:
                target.write("closed")
def factory(setup, options):
    return Fake({**options, "setupId": setup})
def write(target, header, pcm=b""):
    header = dict(header)
    if header["type"] == "audio" and mode == "audioHashMismatch":
        header["audio"] = {**header["audio"], "sha256": "0" * 64}
    if header["type"] == "partial" and mode == "badPartialReply":
        header["partial"] = {**header["partial"], "stable": 7}
    if header["type"] == "ready" and mode == "badReady":
        header["metadata"] = {"revision": "wrong"}
    if header["type"] == "result":
        if mode == "wrongId": header["id"] += 1
        if mode == "wrongGeneration": header["generation"] = "stale"
        if mode == "wrongSetup": header["setupId"] = "other"
        if mode == "wrongOperation": header["operation"] = "load"
        if mode == "wrongVersion": header["version"] = 2
        if mode == "bodyReply": header["pcmBytes"] = 4
        if mode == "badResult": header["result"]["processingMs"] = -1
        if mode == "badDecode": header["result"]["decodeMs"] = -1
        if mode == "badType": header["type"] = "ready"
        if mode == "oversize":
            target.write(struct.pack("<I", worker.MAX_HEADER_BYTES + 1))
            target.flush()
            return
        if mode in ("invalidUtf8", "badJson"):
            target.write(struct.pack("<I", 1) + (b"\\xff" if mode == "invalidUtf8" else b"{"))
            target.flush()
            return
    if mode == "fragmented":
        data = json.dumps(header, separators=(",", ":")).encode()
        data = struct.pack("<I", len(data)) + data
        for byte in data[:8]:
            target.write(bytes([byte]))
            target.flush()
            time.sleep(0.002)
        target.write(data[8:])
        target.flush()
    elif mode == "extraReply" and header["type"] == "result":
        data = json.dumps(header, separators=(",", ":")).encode()
        packet = struct.pack("<I", len(data)) + data
        target.write(packet + packet)
        target.flush()
    else:
        original_write(target, header, pcm)
    if header["type"] == "ready" and mode == "exitAfterReady":
        os._exit(12)
    if header["type"] == "ready" and mode == "sendRace":
        os.close(0)
        time.sleep(30)
worker.backend_factory = factory
worker.write_frame = write
sys.exit(worker.main())
`;

before(async () => {
  // Resolve the interpreter once so per request deadlines exclude a version manager shim.
  PYTHON = execFileSync("python3", ["-c", "import sys; print(sys.executable)"], { encoding: "utf8" }).trim();
  directory = await mkdtemp(join(tmpdir(), "speakeasy-worker-transport-"));
  fixture = join(directory, "fake_worker.py");
  await writeFile(fixture, FAKE);
});
after(async () => { await rm(directory, { recursive: true, force: true }); });

function makeWorker(options: Partial<ConstructorParameters<typeof PythonSetupWorker>[0]> = {}) {
  return new PythonSetupWorker({ setupId: "mlx-parakeet/v3-greedy", pythonPath: PYTHON,
    workerPath: fixture, expectedIdentity: { revision: "pinned" },
    loadTimeoutMs: 3000, operationTimeoutMs: 3000, ...options });
}
function digest(bytes: Uint8Array): string { return createHash("sha256").update(bytes).digest("hex"); }
async function markerPid(path: string): Promise<number> {
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
function assertExited(pid: number): void {
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
}

test("real persistent process preserves offset PCM bytes, hash, signed zero and caller ownership", async (t) => {
  const worker = makeWorker({ backendOptions: { mode: "fragmented" } });
  t.after(() => worker.close());
  const source = new Float32Array([99, 0, -0, 1.5, -2, 0.125, 88]);
  const samples = source.subarray(1, 6);
  const expected = Buffer.from(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
  const pending = worker.transcribe(samples);
  source.fill(7);
  const first = await pending;
  assert.equal(first.text, digest(expected));
  assert.equal((first.diagnostics as { hex: string }).hex, expected.toString("hex"));
  assert.equal((first.diagnostics as { count: number }).count, 5);
  assert.equal(first.firstUse, true);
  assert.equal(first.decodeMs, 1);
  assert.ok(first.startupLoadWallMs > 0);
  assert.ok(first.workerWallMs >= first.startupLoadWallMs);
  const second = await worker.transcribe(new Float32Array());
  assert.equal(second.text, digest(Buffer.alloc(0)));
  assert.equal(second.firstUse, false);
  assert.equal(second.startupLoadWallMs, 0);
  assert.equal(second.metadata.pid, first.metadata.pid);
  assert.equal((second.diagnostics as { calls: number }).calls, 2);
  assert.match(worker.stderr, /python diagnostic/);
  assert.match(worker.stderr, /native diagnostic/);
});

test("production Python process reads split prefixes and coalesced requests then sends closed", async (t) => {
  const generation = "coalesced-generation";
  const setupId = "mlx-parakeet/v3-greedy";
  const child = spawn(PYTHON, ["-B", "-u", fixture, "--setup", setupId, "--generation", generation], { timeout: 3000 });
  t.after(() => { if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); });
  const output: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => output.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  const completion = new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  function packet(id: number, operation: string, fields: Record<string, unknown>, body = Buffer.alloc(0)) {
    const header = Buffer.from(JSON.stringify({ version: 1, generation, setupId, id, operation,
      pcmBytes: body.length, ...fields }));
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32LE(header.length);
    return Buffer.concat([prefix, header, body]);
  }
  const body = Buffer.from(new Float32Array([-0, 1.5]).buffer);
  const closeMarker = join(directory, "protocol-close.txt");
  const stream = Buffer.concat([packet(1, "load", { options: { closeMarker } }), packet(2, "transcribe", {
    sampleRate: 16000, channels: 1, sampleCount: 2, sha256: digest(body),
  }, body), packet(3, "close", {})]);
  for (let index = 0; index < 4; index++) {
    child.stdin.write(stream.subarray(index, index + 1));
    await new Promise((resolve) => setTimeout(resolve, 3));
  }
  child.stdin.end(stream.subarray(4));
  assert.equal(await completion, 0);
  assert.equal(await readFile(closeMarker, "utf8"), "closed");
  let frames = Buffer.concat(output);
  const replies: { type: string; result?: { text: string; diagnostics: { hex: string } } }[] = [];
  while (frames.length) {
    const size = frames.readUInt32LE();
    replies.push(JSON.parse(frames.subarray(4, 4 + size).toString("utf8")));
    frames = frames.subarray(4 + size);
  }
  assert.deepEqual(replies.map((reply) => reply.type), ["ready", "result", "closed"]);
  assert.equal(replies[1].result!.text, digest(body));
  assert.equal(replies[1].result!.diagnostics.hex, body.toString("hex"));
  assert.match(Buffer.concat(stderr).toString(), /native diagnostic/);
});

test("cleanup keeps exact Unicode, whitespace and completed empty text with zero PCM", async (t) => {
  const worker = makeWorker({ setupId: "mlx-cleanup/qwen-targeted" });
  t.after(() => worker.close());
  const base = { sampleRate: 16000 as const, channels: 1 as const, sampleCount: 0,
    pcmBytes: 0, pcmSha256: digest(Buffer.alloc(0)), samples: new Float32Array(65_536) };
  for (const text of ["", "  café\n{\"A\":\"launch\"}  "]) {
    const result = await worker.cleanup(text, base);
    assert.equal(result.text, text);
    assert.equal(result.rawText, text);
  }
  await assert.rejects(worker.cleanup("x".repeat(MAX_WORKER_TEXT_BYTES + 1), base), /limit/);
  await assert.rejects(worker.cleanup("", { ...base, sampleCount: 1 }), /identity/);
});

test("cleanText sends a final's text and context with zero PCM, loads on first use, and only for cleanup models", async (t) => {
  const s1 = makeWorker({ setupId: "superwhisper/s1-mini" });
  const fm = makeWorker({ setupId: "apple/foundation-models" });
  t.after(() => Promise.all([s1.close(), fm.close()]));
  const first = await s1.cleanText({ text: "  um café  " });
  assert.equal(first.text, "  um café  ");
  assert.equal(first.firstUse, true);
  assert.ok(first.startupLoadWallMs > 0, "the first call spawns and loads");
  assert.equal((await s1.cleanText({ text: "again" })).firstUse, false);
  const context = { instruction: "Clean it.", names: ["Sean", "Michaela"], earlier: ["Before."] };
  assert.deepEqual(JSON.parse((await fm.cleanText({ text: "um tell shawn", ...context })).text),
    { text: "um tell shawn", ...context });
  await assert.rejects(s1.cleanText({ text: "x", names: ["Sean"] }), /takes text only/, "a text only model refuses context");
  await assert.rejects(fm.cleanText({ text: "x".repeat(MAX_WORKER_TEXT_BYTES + 1) }), /limit/);
  await assert.rejects(makeWorker().cleanText({ text: "x" }), /not a text cleanup worker/);
  await assert.rejects(makeWorker({ setupId: "superwhisper/s1-mini" }).transcribe(new Float32Array(4)), /does not transcribe/);
});

test("nonfinite samples and oversized takes fail before spawning", async (t) => {
  const worker = makeWorker({ pythonPath: "/nonexistent-transport-test-python" });
  t.after(() => worker.close());
  for (const value of [NaN, Infinity, -Infinity]) {
    await assert.rejects(worker.transcribe(new Float32Array([value])), /finite/);
  }
  await assert.rejects(worker.transcribe(new Float32Array(MAX_WORKER_PCM_BYTES / 4 + 1)), /length/);
  await assert.rejects(worker.transcribe(new Float32Array()), /ENOENT/);
});

test("real process malformed replies retire their generation before rejection", async (t) => {
  const modes = ["wrongId", "wrongGeneration", "wrongSetup", "wrongOperation", "wrongVersion", "bodyReply",
    "badResult", "badDecode", "badType", "oversize", "invalidUtf8", "badJson", "extraReply"];
  for (const mode of modes) {
    await t.test(mode, async (s) => {
      const marker = join(directory, `${mode}.pid`);
      const worker = makeWorker({ backendOptions: { mode, marker } });
      s.after(() => worker.close());
      await assert.rejects(worker.transcribe(new Float32Array([0.5])));
      assertExited(await markerPid(marker));
    });
  }
});

test("ready attestation is checked and missing ready has its own deadline", async (t) => {
  const invalid = makeWorker({ backendOptions: { mode: "badReady" } });
  t.after(() => invalid.close());
  await assert.rejects(invalid.transcribe(new Float32Array()), /identity mismatch/);
  const marker = join(directory, "missing-ready.pid");
  const missing = makeWorker({ loadTimeoutMs: 1500, backendOptions: { mode: "missingReady", marker } });
  t.after(() => missing.close());
  await assert.rejects(missing.transcribe(new Float32Array()), /load timed out/);
  assertExited(await markerPid(marker));
});

test("backend error preserves partial output and load failure ends its child", async (t) => {
  const closeMarker = join(directory, "error-close.txt");
  const worker = makeWorker({ backendOptions: { mode: "error", closeMarker } });
  t.after(() => worker.close());
  await assert.rejects(worker.transcribe(new Float32Array()), (error: Error & { rawText?: string }) => {
    assert.equal(error.message, "backend failed");
    assert.equal(error.rawText, "partial\n");
    return true;
  });
  assert.equal(await readFile(closeMarker, "utf8"), "closed");
  const loadCloseMarker = join(directory, "load-error-close.txt");
  const loading = makeWorker({ backendOptions: { mode: "loadError", closeMarker: loadCloseMarker } });
  t.after(() => loading.close());
  await assert.rejects(loading.transcribe(new Float32Array()), /load failed/);
  assert.equal(await readFile(loadCloseMarker, "utf8"), "closed");
});

test("active cancellation holds the slot until SIGTERM resistant process exits; only next request restarts", async (t) => {
  const marker = join(directory, "cancel.pid");
  const closeMarker = join(directory, "forced-cancel-close.txt");
  const backendOptions = { mode: "ignoreTerm", marker, closeMarker };
  const worker = makeWorker({ backendOptions });
  t.after(() => worker.close());
  const controller = new AbortController();
  const outcome = worker.transcribe(new Float32Array([0.5]), controller.signal);
  const rejected = assert.rejects(outcome, /cancelled/);
  const pid = await markerPid(marker);
  controller.abort();
  await assert.rejects(worker.transcribe(new Float32Array()), /busy/);
  await rejected;
  assertExited(pid);
  await assert.rejects(readFile(closeMarker), { code: "ENOENT" });
  // A later explicit request is the sole restart trigger.
  backendOptions.mode = "";
  const next = await worker.transcribe(new Float32Array([0.25]));
  assert.notEqual(next.metadata.pid, pid);
  assert.equal(next.firstUse, true);
});

test("operation timeout, process crash and explicit release recover on a later request", async (t) => {
  for (const mode of ["blocked", "crash"]) {
    await t.test(mode, async (s) => {
      const marker = join(directory, `${mode}.pid`);
      const backendOptions = { mode, marker };
      const worker = makeWorker({ operationTimeoutMs: 150, backendOptions });
      s.after(() => worker.close());
      await assert.rejects(worker.transcribe(new Float32Array()), mode === "blocked" ? /transcribe timed out/ : /exited|pipe closed/);
      const pid = await markerPid(marker);
      assertExited(pid);
      backendOptions.mode = "";
      const next = await worker.transcribe(new Float32Array());
      assert.notEqual(next.metadata.pid, pid);
      await worker.release();
      assertExited(next.metadata.pid as number);
      const afterRelease = await worker.transcribe(new Float32Array());
      assert.notEqual(afterRelease.metadata.pid, next.metadata.pid);
    });
  }
});

test("cancel before load, cancel during load and close do not leave a live child", async (t) => {
  const controller = new AbortController();
  controller.abort();
  const beforeLoad = makeWorker({ pythonPath: "/nonexistent-transport-test-python" });
  t.after(() => beforeLoad.close());
  await assert.rejects(beforeLoad.transcribe(new Float32Array(), controller.signal), /cancelled/);
  for (const action of ["cancel", "close"]) {
    const marker = join(directory, `load-${action}.pid`);
    const worker = makeWorker({ backendOptions: { mode: "missingReady", marker } });
    t.after(() => worker.close());
    const active = new AbortController();
    const outcome = worker.transcribe(new Float32Array(), active.signal);
    const rejected = assert.rejects(outcome, action === "cancel" ? /cancelled/ : /closed/);
    const pid = await markerPid(marker);
    if (action === "cancel") active.abort();
    else await worker.close();
    await rejected;
    assertExited(pid);
    await worker.close();
    await assert.rejects(worker.transcribe(new Float32Array()), /closed/);
  }
});

test("stderr retained for diagnostics is bounded", async (t) => {
  const worker = makeWorker({ backendOptions: { mode: "stderr" } });
  t.after(() => worker.close());
  await worker.transcribe(new Float32Array());
  assert.ok(Buffer.byteLength(worker.stderr) <= 16_384);
});

test("exit and closed input during send settle once and await actual exit", async (t) => {
  for (const mode of ["exitAfterReady", "sendRace"]) {
    const marker = join(directory, `${mode}.pid`);
    const backendOptions = { mode, marker };
    const worker = makeWorker({ backendOptions });
    t.after(() => worker.close());
    await assert.rejects(worker.transcribe(new Float32Array(256_000)));
    const pid = await markerPid(marker);
    assertExited(pid);
    backendOptions.mode = "";
    assert.notEqual((await worker.transcribe(new Float32Array())).metadata.pid, pid);
  }
});

test("scheduler executor loads explicitly, verifies canonical identity, and preserves cleanup dependency text", async (t) => {
  const options = { pythonPath: PYTHON, workerPath: fixture, expectedIdentity: { revision: "pinned" } };
  const executor = new SetupWorkers({ python: {
    "mlx-parakeet/ultra-greedy": options, "mlx-cleanup/qwen-targeted": options,
  } });
  t.after(() => executor.close());
  const context = { signal: new AbortController().signal };
  const pcm = createCanonicalPcm(new Float32Array([-0, 0.5, 1.5]));
  const request: SetupExecution = { requestId: "take-1", setupId: "mlx-parakeet/ultra-greedy",
    operation: "transcribe", pcm };
  await assert.rejects(executor.execute(request, context), /explicit load required/);
  const loaded: SetupLoadResult = await executor.load(request.setupId, context);
  assert.equal(loaded.firstUse, true);
  assert.equal(loaded.loadMs, 2);
  assert.equal(loaded.startupMs, null);
  const result = await executor.execute(request, context);
  assert.equal(result.text, pcm.pcmSha256);
  assert.equal(result.processingMs, 3);
  assert.equal(result.decodeMs, 1);
  assert.ok(result.workerWallMs! >= 0);
  const warm = await executor.load(request.setupId, context);
  assert.deepEqual(warm, { firstUse: false, startupMs: 0, loadMs: 0 });
  await assert.rejects(executor.execute({ ...request, pcm: { ...pcm, pcmSha256: "0".repeat(64) } }, context), /identity/);
  await executor.load("mlx-cleanup/qwen-targeted", context);
  const cleanup: SetupExecution = { operation: "cleanup", requestId: "take-1",
    setupId: "mlx-cleanup/qwen-targeted", base: pcm, baseTextSha256: sha256(""), A: "", B: null };
  const cleaned = await executor.execute(cleanup, context);
  assert.equal(cleaned.text, "");
  await assert.rejects(executor.execute({ ...cleanup, baseTextSha256: "0".repeat(64) }, context), /identity/);
  assert.equal(result.diagnostics!.settings, undefined, "a take without final settings sends none");
  const biased = await executor.execute({ ...request, settings: { namesBias: 2.5 } }, context);
  assert.deepEqual(biased.diagnostics!.settings, { namesBias: 2.5 }, "a take's final settings reach the backend");
  await executor.release(request.setupId);
  await assert.rejects(executor.execute(request, context), /explicit load required/);
  await executor.load(request.setupId, context);
  assert.equal((await executor.execute(request, context)).text, pcm.pcmSha256);
  await assert.rejects(executor.load("qwen/context", context), /not configured/);
  await executor.close();
  await assert.rejects(executor.load(request.setupId, context), /closed/);
});

test("a names setup reuses its generation for the same list, reloads at load for another, and refuses a stale execute", async (t) => {
  const executor = new SetupWorkers({ python: { "cohere/transcribe": { pythonPath: PYTHON, workerPath: fixture,
    expectedIdentity: { revision: "pinned" } } } });
  t.after(() => executor.close());
  const signal = new AbortController().signal;
  const eight = DEFAULT_NAMES_SNAPSHOT;
  const two = namesSnapshot(["Qwen", "TM"]);
  const request: SetupExecution = { operation: "transcribe", setupId: "cohere/transcribe", requestId: "names-1",
    pcm: createCanonicalPcm(new Float32Array([0.5])), names: eight };
  assert.equal((await executor.load(request.setupId, { signal, names: eight })).startupMs, null);
  const first = await executor.execute(request, { signal });
  assert.deepEqual(first.diagnostics!.names, eight.names, "the load option carries the list");
  assert.deepEqual(await executor.load(request.setupId, { signal, names: eight }), { firstUse: false, startupMs: 0, loadMs: 0 },
    "the same digest keeps the generation");
  await assert.rejects(executor.execute({ ...request, names: two }, { signal }), /names changed since load/);
  const reloaded = await executor.load(request.setupId, { signal, names: two });
  assert.deepEqual([reloaded.startupMs, reloaded.firstUse], [null, true], "the reload counts as load time");
  assertExited(first.diagnostics!.pid as number);
  const second = await executor.execute({ ...request, names: two }, { signal });
  assert.deepEqual(second.diagnostics!.names, ["Qwen", "TM"]);
  await assert.rejects(executor.execute({ ...request, names: eight }, { signal }), /names changed since load/);
  await assert.rejects(executor.execute({ ...request, names: undefined }, { signal }), /names changed since load/);
});

test("a CPU setup prepares the take's names, counts a list change as a cold load and refuses a stale execute", async () => {
  const prepared: (readonly string[])[] = [];
  const sent: (readonly string[])[] = [];
  const ready = { type: "ready", version: 1, load: { loadMs: 10, modelConstructionMs: 8 }, assets: {} } as unknown as CpuReady;
  const executor = new SetupWorkers({ cpu: { "tm/today": {
    async prepare(names: readonly string[]) { prepared.push(names); return ready; },
    async transcribe(_samples: Float32Array, names: readonly string[]) {
      sent.push(names);
      return { text: "", decodeMs: 1, model: "cpu", hotwords: { path: "names", count: names.length, revision: "r" },
        comparison: { loadMs: 10, modelConstructionMs: 8, processingMs: 1, workerWallMs: 1, firstUse: true,
          decodingMethod: "modified_beam_search" } };
    },
    async closeAndWait() {},
  } } });
  const context = { signal: new AbortController().signal };
  const two = namesSnapshot(["Qwen", "TM"]);
  const request: SetupExecution = { operation: "transcribe", setupId: "tm/today", requestId: "cpu-names",
    pcm: createCanonicalPcm(new Float32Array([0.5])), names: DEFAULT_NAMES_SNAPSHOT };
  await executor.load("tm/today", { ...context, names: DEFAULT_NAMES_SNAPSHOT });
  assert.deepEqual((await executor.execute(request, context)).diagnostics, { modelConstructionMs: 8, names: 8, namesRevision: "r",
    decodingMethod: "modified_beam_search" }, "the row reports the count applied and the decoding used");
  assert.deepEqual((await executor.load("tm/today", { ...context, names: DEFAULT_NAMES_SNAPSHOT })).startupMs, 0, "warm");
  await assert.rejects(executor.execute({ ...request, names: two }, context), /names changed since load/);
  assert.deepEqual(await executor.load("tm/today", { ...context, names: two }), { startupMs: null, loadMs: 10, firstUse: true });
  await executor.execute({ ...request, names: two }, context);
  assert.deepEqual(prepared, [DEFAULT_NAMES_SNAPSHOT.names, DEFAULT_NAMES_SNAPSHOT.names, two.names]);
  assert.deepEqual(sent, [DEFAULT_NAMES_SNAPSHOT.names, two.names]);
  await executor.close();
});

test("scheduler delegates CPU prepare, timing, cancellation and release to the supplied fixed child", async () => {
  const ready: CpuReady = { type: "ready", version: 1, preset: CPU_PRESETS["sweep/score-2"],
    load: { loadMs: 10, modelConstructionMs: 8 }, assets: {} };
  const calls: string[] = [];
  let releaseBlocked!: () => void;
  let started!: () => void;
  const barrier = new Promise<void>((resolve) => { started = resolve; });
  let block = false;
  const executor = new SetupWorkers({ cpu: { "sweep/score-2": {
    async prepare() { calls.push("prepare"); return ready; },
    async transcribe(samples) {
      calls.push("transcribe");
      assert.equal(samples.length, 1);
      if (block) {
        await new Promise<void>((resolve) => { releaseBlocked = resolve; started(); });
        throw new Error("child closed");
      }
      return { text: "", decodeMs: 7, model: CPU_PRESETS["sweep/score-2"].model,
        comparison: { loadMs: 10, modelConstructionMs: 8, processingMs: 3, workerWallMs: 4, firstUse: true } };
    },
    async closeAndWait() { calls.push("closeAndWait"); releaseBlocked?.(); },
  } } });
  const context = { signal: new AbortController().signal };
  const request: SetupExecution = { operation: "transcribe", setupId: "sweep/score-2",
    requestId: "cpu-1", pcm: createCanonicalPcm(new Float32Array([0.5])) };
  await assert.rejects(executor.execute(request, context), /explicit load/);
  assert.deepEqual(await executor.load(request.setupId, context), { startupMs: null, loadMs: 10, firstUse: true });
  assert.deepEqual(await executor.execute(request, context), { text: "", decodeMs: 7,
    processingMs: 3, workerWallMs: 4, diagnostics: { modelConstructionMs: 8 } });
  assert.equal((await executor.load(request.setupId, context)).firstUse, false);
  block = true;
  const controller = new AbortController();
  const cancelled = assert.rejects(executor.execute(request, { signal: controller.signal }), /cancelled/);
  await barrier;
  controller.abort();
  await cancelled;
  await assert.rejects(executor.execute(request, context), /explicit load/);
  await executor.load(request.setupId, context);
  await executor.release(request.setupId);
  await executor.close();
  assert.ok(calls.filter((call) => call === "closeAndWait").length >= 3);
});

test("each CPU setup has its own child, so one list's take never waits on or reuses the other's", async () => {
  const calls: string[] = [];
  const fake = (id: "sweep/score-2" | "tm/today") => ({
    async prepare(): Promise<CpuReady> {
      calls.push(`${id} prepare`);
      return { type: "ready", version: 1, preset: CPU_PRESETS[id], load: { loadMs: 1, modelConstructionMs: 1 }, assets: {} };
    },
    async transcribe() {
      calls.push(`${id} transcribe`);
      return { text: id, decodeMs: 1, model: CPU_PRESETS[id].model,
        comparison: { loadMs: 1, modelConstructionMs: 1, processingMs: 1, workerWallMs: 1, firstUse: true } };
    },
    async closeAndWait() { calls.push(`${id} close`); },
  });
  const executor = new SetupWorkers({ cpu: { "sweep/score-2": fake("sweep/score-2"), "tm/today": fake("tm/today") } });
  const context = { signal: new AbortController().signal };
  const pcm = createCanonicalPcm(new Float32Array([0.5]));
  await executor.load("tm/today", context);
  await assert.rejects(executor.execute({ operation: "transcribe", setupId: "sweep/score-2", requestId: "a", pcm }, context),
    /explicit load/);
  assert.equal((await executor.execute({ operation: "transcribe", setupId: "tm/today", requestId: "b", pcm }, context)).text,
    "tm/today");
  await executor.release("tm/today");
  // A failed take retires only its own setup's child.
  assert.deepEqual(calls, ["tm/today prepare", "sweep/score-2 close", "tm/today transcribe", "tm/today close"]);
  await executor.close();
});

test("close awaits a CPU preparation already in flight before its final child retirement", async () => {
  let release!: () => void;
  const preparing = new Promise<CpuReady>((resolve) => { release = () => resolve({ type: "ready", version: 1,
    preset: CPU_PRESETS["sweep/score-2"], load: { loadMs: 1, modelConstructionMs: 1 }, assets: {} }); });
  let closes = 0;
  const executor = new SetupWorkers({ cpu: { "sweep/score-2": {
    prepare: () => preparing,
    async transcribe() { throw new Error("unused"); },
    async closeAndWait() { closes++; },
  } } });
  const pending = assert.rejects(executor.load("sweep/score-2", { signal: new AbortController().signal }), /closed/);
  let settled = false;
  const closing = executor.close().then(() => { settled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(settled, false);
  release();
  await pending;
  await closing;
  assert.ok(closes >= 2);
});

test("a worker voice speaks one segment and returns its exact PCM, checked against its identity", async (t) => {
  const voice = (mode = "") => new PythonSetupWorker({ setupId: "kokoro-mlx-bf16", pythonPath: PYTHON, workerPath: fixture,
    backendOptions: { mode }, expectedIdentity: { revision: "pinned" }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 });
  const worker = voice();
  t.after(() => worker.close());
  const first = await worker.speak("Hi.", "af_heart", 1.5);
  assert.deepEqual([...first.samples], [3, 1.5, -0, 8]);
  assert.ok(Object.is(first.samples[2], -0));
  assert.equal(first.sampleRate, 24000);
  assert.equal(first.processingMs, 2);
  assert.equal(first.firstUse, true);
  assert.ok(first.startupLoadWallMs > 0, "the first segment carries the load");
  const second = await worker.speak("Again.", "af_heart", 1);
  assert.equal(second.firstUse, false);
  assert.equal(second.startupLoadWallMs, 0);
  await assert.rejects(worker.speak(" ", "af_heart", 1), /empty/);
  for (const mode of ["badAudio", "audioHashMismatch"]) {
    const failing = voice(mode);
    t.after(() => failing.close());
    await assert.rejects(failing.speak("Hi.", "af_heart", 1), mode === "badAudio" ? /finite/ : /invalid voice audio/);
  }
});

test("a voice copy speak carries settings in its header and the reference as a hashed body", async (t) => {
  const voice = (mode: string) => new PythonSetupWorker({ setupId: "qwen3-tts-0.6b-base", pythonPath: PYTHON, workerPath: fixture,
    backendOptions: { mode }, expectedIdentity: { revision: "pinned" }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 });
  const samples = Float32Array.of(0.5, -0.25, 0.125);
  const reference = { sha256: pcmDigest(samples), samples, sampleRate: 48_000, transcript: "Hello." };
  const copy = voice("copy");
  t.after(() => copy.close());
  const read = await copy.speak("Hi.", "recording", 1, undefined, { settings: { seed: 7 }, reference });
  assert.deepEqual([...read.samples], [7, 48_000, 3, 6]);
  assert.deepEqual([...(await copy.speak("Hi.", "ref-3s", 1, undefined, { settings: { seed: 2 } })).samples], [2, -1, -1, 0]);
  await assert.rejects(copy.speak("Hi.", "recording", 1, undefined, { reference: { ...reference, sha256: "0".repeat(64) } }),
    /reference identity mismatch/, "the host refuses a reference whose PCM differs from its id");
  const plain = voice("");
  t.after(() => plain.close());
  await assert.rejects(plain.speak("Hi.", "ref-3s", 1, undefined, { settings: { seed: 2 } }), /takes no settings/);
  await assert.rejects(plain.speak("Hi.", "ref-3s", 1, undefined, { reference }), /takes no reference/);
});

test("a voice copy pick is read only when its bytes match the payload pin", async () => {
  const bytes = encodeWavFloat32(Float32Array.of(0.5), 24_000);
  await mkdir(join(directory, "picks", "references"), { recursive: true });
  await writeFile(join(directory, "picks", "references", "pick.wav"), bytes);
  const payload = { name: "picks", note: "", family: "worker" as const, sampleRate: 24_000, numSpeakers: 1, attribution: [],
    files: [{ name: "references/pick.wav", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }] };
  assert.deepEqual(await readPinnedPick(payload, directory, "references/pick.wav"), Buffer.from(bytes));
  await assert.rejects(readPinnedPick({ ...payload, files: [{ ...payload.files[0]!, sha256: "0".repeat(64) }] }, directory,
    "references/pick.wav"), /differs from its pin/);
  await assert.rejects(readPinnedPick(payload, directory, "references/other.wav"));
});

test("a worker voice synth maps the pipeline's speaker to the registry voice and passes the pins", async (t) => {
  const synth = await createWorkerVoiceSynth("kokoro-mlx-bf16", { pythonPath: PYTHON, loadTimeoutMs: 3000 }, directory, fixture);
  t.after(() => synth.close());
  assert.equal(synth.numSpeakers, 2);
  assert.equal(synth.sampleRate, 24000);
  const michael = await synth.synth({ text: "Hi.", sid: 1, speed: 1 });
  assert.deepEqual([...michael.samples], [3, 1, -0, "am_michael".length]);
  assert.equal(michael.audioDurationMs, 4 / 24);
  assert.equal((await synth.synth({ text: "Hi." })).samples[3], "af_heart".length, "speaker 0 is the default voice");
  await assert.rejects(synth.synth({ text: "Hi.", sid: 2 }), /Invalid TTS speaker/);
});

test("live words feed a loaded worker, mark it used, and reject a malformed partial", async (t) => {
  const worker = makeWorker();
  t.after(() => worker.close());
  await assert.rejects(worker.feed(new Float32Array(4), true), /explicit load required/);
  assert.equal((await worker.load()).firstUse, true);
  assert.deepEqual(await worker.feed(new Float32Array([0.5, -0, 1]), true), { stable: " heard 3", tentative: " maybe", processingMs: 2 });
  assert.equal((await worker.feed(new Float32Array(0), false)).stable, " heard 3", "an empty feed still answers");
  assert.equal((await worker.feed(new Float32Array(2), false)).stable, " heard 5");
  assert.equal((await worker.transcribe(new Float32Array(4))).firstUse, false, "the final pass after live words is warm");
  for (const [mode, message] of [["badPartial", /finite processingMs/], ["badPartialReply", /invalid worker partial/]] as const) {
    const bad = makeWorker({ backendOptions: { mode } });
    t.after(() => bad.close());
    await bad.load();
    await assert.rejects(bad.feed(new Float32Array(2), true), message);
  }
  const executor = new SetupWorkers({ python: { "hex/unified": { pythonPath: PYTHON, workerPath: fixture,
    expectedIdentity: { revision: "pinned" }, loadTimeoutMs: 3000, operationTimeoutMs: 3000 } } });
  t.after(() => executor.close());
  const signal = new AbortController().signal;
  await executor.load("hex/unified", { signal });
  assert.equal((await executor.feed("hex/unified", new Float32Array(7), true, { signal })).stable, " heard 7");
  assert.equal((await executor.feed("hex/unified", new Float32Array(1), true, { signal }, { lookaheadMs: 480 })).tentative,
    ' maybe {"lookaheadMs": 480}', "a start's settings reach the backend");
  assert.equal((await executor.feed("hex/unified", new Float32Array(1), false, { signal }, { lookaheadMs: 480 })).tentative,
    " maybe", "settings go only with a start");
  await assert.rejects(executor.feed("sweep/score-2", new Float32Array(1), true, { signal }), /final text only/);
});
