import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createHash } from "node:crypto";
import { startBrowserVoiceServer, type BrowserVoiceServer, type BrowserVoiceServerOptions } from "./host.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { createFileInputAdapters, createHostProviderConfig, type InputApiAdapter } from "./setup-routes.ts";
import { SETUPS } from "./setup-registry.ts";
import { createCanonicalPcm, type SetupExecution, type SetupExecutor } from "./setup-protocol.ts";

class FakeExecutor implements SetupExecutor {
  calls: SetupExecution[] = [];
  loads = 0;
  closed = 0;
  block?: (request: SetupExecution, signal: AbortSignal) => Promise<{ text: string; processingMs: number; workerWallMs: number }>;
  async load() { this.loads++; return { startupMs: 0, loadMs: 0, firstUse: true }; }
  async execute(request: SetupExecution, { signal }: { signal: AbortSignal }) {
    this.calls.push(request);
    return this.block?.(request, signal) ?? { text: " exact text ", processingMs: 1, workerWallMs: 2 };
  }
  async release() {}
  async close() { this.closed++; }
}

const readiness = () => SETUPS.map((setup) => ({ setupId: setup.id, ready: true }));
const config = { schema: 1 as const, provider: { kind: "cartesia" as const, model: "fake-model" } };

async function fixture(t: TestContext, options: BrowserVoiceServerOptions = {}) {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-host-setup-"));
  const promptsPath = join(root, "prompts.json");
  const outputDir = join(root, "takes");
  await writeFile(promptsPath, JSON.stringify({ set: "test", items: [
    { id: "p001", stratum: "test", text: "Expected words must never reach an executor." },
  ] }));
  const executor = new FakeExecutor();
  const host = await startBrowserVoiceServer({ port: 0, setupExecutor: executor, setupReadiness: readiness,
    transcriber: { async transcribe() { throw new Error("legacy must not be called"); }, close() {} },
    promptReader: { promptsPath, outputDir }, ...options });
  t.after(async () => { await host.close(); await rm(root, { recursive: true, force: true }); });
  return { host, root, outputDir, executor };
}

function post(host: BrowserVoiceServer, path: string, body: object | Uint8Array, cookie?: string) {
  const binary = body instanceof Uint8Array;
  return fetch(host.url + path, { method: "POST", headers: { origin: host.url,
    "content-type": binary ? "application/octet-stream" : "application/json", ...(cookie ? { cookie } : {}) },
    body: binary ? new Blob([new Uint8Array(body).buffer]) : JSON.stringify(body) });
}

function take(): Float32Array<ArrayBuffer> {
  const samples = Float32Array.from({ length: 321 }, (_, index) => Math.sin(index));
  samples[0] = -0;
  samples[2] = 1.25;
  return samples;
}

async function events(response: Response): Promise<Array<Record<string, any>>> {
  assert.equal(response.status, 200, await (response.status === 200 ? Promise.resolve("") : response.text()));
  assert.match(response.headers.get("content-type")!, /application\/x-ndjson/);
  return (await response.text()).trim().split("\n").map((line) => JSON.parse(line));
}

test("setup metadata is fixed ordered and cannot load or execute models", async (t) => {
  const { host, executor } = await fixture(t);
  const response = await fetch(host.url + "/prompt-reader/setups");
  const metadata = await response.json() as any;
  assert.deepEqual(metadata.setups.map((setup: any) => setup.id), SETUPS.map((setup) => setup.id));
  assert.equal(metadata.defaultSetupId, "mlx-parakeet/v3-greedy");
  assert.equal(metadata.setups.find((setup: any) => setup.id === "sweep/score-2").maxAudioMs, 60_000);
  assert.equal(executor.loads, 0);
  assert.equal(executor.calls.length, 0);
  const forbidden = await fetch(host.url + "/prompt-reader/setups", { headers: { origin: "https://foreign.example" } });
  assert.equal(forbidden.status, 403);
});

test("local comparison streams rows progressively, replays exact saved samples and leaves corpus unchanged", async (t) => {
  const { host, outputDir, executor } = await fixture(t);
  const samples = take();
  const bytes = new Uint8Array(samples.buffer);
  const live = await events(await post(host, "/prompt-reader/comparisons?requestId=live", bytes));
  assert.equal(live[0]!.type, "accepted");
  assert.equal(live.at(-1)!.type, "done");
  assert.equal(live[0]!.pcm.pcmSha256, createCanonicalPcm(samples).pcmSha256);
  assert.ok(live.some((event) => event.type === "row" && event.row.text === " exact text "));
  assert.ok(live.every((event, index) => event.requestId === "live" && event.sequence === index));
  const saved = await post(host, "/prompt-reader/recordings?promptId=p001", bytes);
  assert.equal(saved.status, 201);
  const { audio } = await saved.json() as { audio: string };
  const files = await readdir(outputDir);
  const before = await Promise.all(files.map((file) => readFile(join(outputDir, file))));
  const replay = await events(await post(host, "/prompt-reader/comparisons", {
    requestId: "saved", source: { kind: "saved", takeName: audio },
  }));
  assert.deepEqual(replay[0]!.pcm, live[0]!.pcm);
  assert.equal(replay[0]!.source.kind, "saved-take");
  assert.equal(replay[0]!.source.takeId, audio);
  assert.deepEqual(executor.calls.length, 2);
  for (const request of executor.calls) {
    assert.equal(request.operation, "transcribe");
    if (request.operation !== "transcribe") throw new Error();
    assert.deepEqual(request.pcm.bytes, bytes);
    assert.equal(Object.is(request.pcm.samples[0], -0), true);
    assert.equal(request.pcm.samples[2], 1.25);
    assert.equal("expected" in request, false);
  }
  assert.deepEqual(await Promise.all(files.map((file) => readFile(join(outputDir, file)))), before);
  for (const name of ["../prompts.json", "/etc/hosts", "unlisted.wav"]) {
    assert.equal((await post(host, "/prompt-reader/comparisons", { source: { kind: "saved", takeName: name } })).status, 404);
  }
  await writeFile(join(outputDir, audio), "corrupt WAV");
  assert.equal((await post(host, "/prompt-reader/comparisons", { source: { kind: "saved", takeName: audio } })).status, 400);
  assert.equal(executor.calls.length, 2);
});

test("comparison ingress rejects malformed metadata and nonfinite PCM before dispatch", async (t) => {
  const { host, executor } = await fixture(t);
  const bytes = new Uint8Array(take().buffer);
  for (const query of ["setupIds=[]", "setupIds=broken", "setupIds=%5B%22sweep%2Fscore-2%22%2C%22sweep%2Fscore-2%22%5D",
    "requestId=bad%20id", "requestId=a&requestId=b", "source=saved", "approved=true"]) {
    assert.equal((await post(host, "/prompt-reader/comparisons?" + query, bytes)).status, 400, query);
  }
  for (const body of [new Uint8Array(), new Uint8Array(3), new Uint8Array(new Float32Array([NaN]).buffer),
    new Uint8Array(new Float32Array([Infinity]).buffer)]) {
    assert.equal((await post(host, "/prompt-reader/comparisons", body)).status, 400);
  }
  assert.equal((await post(host, "/prompt-reader/comparisons", { setupIds: [SETUPS[0]!.id],
    source: { kind: "saved", takeName: "unknown.wav" }, samples: [1] })).status, 400);
  assert.equal(executor.loads, 0);
});

test("one active and one queued comparison stream immediately, disconnect cancels and shutdown awaits executor", async (t) => {
  const { host, executor } = await fixture(t);
  let aborted = false;
  executor.block = async (_request, signal) => {
    await new Promise<void>((resolve) => signal.addEventListener("abort", () => { aborted = true; resolve(); }, { once: true }));
    throw new Error("cancelled");
  };
  const bytes = new Uint8Array(take().buffer);
  const first = await post(host, "/prompt-reader/comparisons?requestId=first", bytes);
  const firstReader = first.body!.getReader();
  assert.match(new TextDecoder().decode((await firstReader.read()).value), /accepted/);
  const second = await post(host, "/prompt-reader/comparisons?requestId=second", bytes);
  const secondReader = second.body!.getReader();
  assert.match(new TextDecoder().decode((await secondReader.read()).value), /"queued":true/);
  const full = await post(host, "/prompt-reader/comparisons?requestId=third", bytes);
  assert.equal(full.status, 429);
  await secondReader.cancel();
  await firstReader.cancel();
  await waitFor(() => aborted);
  executor.block = undefined;
  await events(await post(host, "/prompt-reader/comparisons?requestId=later", bytes));
  const active = await post(host, "/prompt-reader/comparisons?requestId=shutdown", bytes);
  await active.body!.cancel();
  await host.close();
  assert.equal(executor.closed, 1);
});

test("legacy transcription retains its serial request group and excludes comparison work", async (t) => {
  let entered = 0;
  let finish!: () => void;
  const pending = new Promise<void>((resolve) => { finish = resolve; });
  const { host } = await fixture(t, { transcriber: {
    async transcribe() { entered++; await pending; return { model: "fake", text: "legacy", decodeMs: 1 }; }, close() { finish(); },
  } });
  const bytes = new Uint8Array(take().buffer);
  const first = post(host, "/prompt-reader/transcriptions", bytes);
  const second = post(host, "/prompt-reader/transcriptions", bytes);
  try {
    await waitFor(() => entered === 2);
    assert.equal((await post(host, "/prompt-reader/comparisons", bytes)).status, 409);
  } finally { finish(); }
  assert.equal((await first).status, 200);
  assert.equal((await second).status, 200);
  await events(await post(host, "/prompt-reader/comparisons", bytes));
});

async function session(host: BrowserVoiceServer) {
  const response = await post(host, "/speech-providers/session", {});
  assert.equal(response.status, 201);
  const body = await response.json() as { sessionId: string };
  return { id: body.sessionId, cookie: response.headers.get("set-cookie")!.split(";")[0]! };
}

test("server assigned sessions and exact selection protect one live API dispatch", async (t) => {
  let calls = 0;
  let source: unknown;
  const adapter: InputApiAdapter = { kind: "cartesia", model: "fake-model", label: "Fake API",
    async transcribe(samples, context) {
      context.recheck();
      assert.equal(context.apiKey!(), "fake-key");
      source = context.authorization.source;
      calls++;
      assert.deepEqual(samples, take());
      return { text: "  unchanged  ", providerNetworkMs: null };
    } };
  const providers = createHostProviderConfig({ credentials: createProviderCredentials(() => ({ CARTESIA_API_KEY: "fake-key" })),
    readiness, inputAdapters: [adapter] });
  const { host } = await fixture(t, { providers });
  const catalogue = providers.catalogue();
  assert.equal(calls, 0);
  assert.equal((await post(host, "/speech-providers/session", { sessionId: "forged" })).status, 400);
  const first = await session(host);
  const second = await session(host);
  assert.notEqual(first.id, second.id);
  assert.match(first.cookie, /^speakeasy-session=/);
  const selectedResponse = await post(host, "/speech-providers/selection", {
    direction: "input", catalogueRevision: catalogue.revision, config,
  }, first.cookie);
  assert.equal(selectedResponse.status, 200);
  const selected = await selectedResponse.json() as { selectionRevision: number };
  const query = new URLSearchParams({ config: JSON.stringify(config), catalogueRevision: catalogue.revision,
    selectionRevision: String(selected.selectionRevision) });
  const path = "/prompt-reader/api-transcriptions?" + query;
  const bytes = new Uint8Array(take().buffer);
  assert.equal((await post(host, path, bytes, second.cookie)).status, 403);
  assert.equal((await post(host, path, bytes, "speakeasy-session=forged")).status, 403);
  const live = await post(host, path, bytes, first.cookie);
  assert.equal(live.status, 200);
  assert.equal(((await live.json()) as any).result.text, "  unchanged  ");
  assert.deepEqual(source && (source as any).kind, "live-capture");
  assert.equal((source as any).sessionId, first.id);
  assert.equal(calls, 1);
  query.set("source", "saved");
  assert.equal((await post(host, "/prompt-reader/api-transcriptions?" + query, bytes, first.cookie)).status, 400);
  assert.equal(calls, 1);
});

test("actual file adapters receive only the explicitly selected live take with fake transports", async (t) => {
  const files: Uint8Array[] = [];
  const adapters = createFileInputAdapters({ fetch: async (_url, init) => {
    const form = init!.body as FormData;
    files.push(new Uint8Array(await (form.get("file") as Blob).arrayBuffer()));
    return Response.json({ type: "transcript", text: "unmodified" });
  } });
  const providers = createHostProviderConfig({ credentials: createProviderCredentials(() => ({
    CARTESIA_API_KEY: "fake-cartesia", OPENAI_API_KEY: "fake-openai" })), readiness, inputAdapters: adapters });
  const { host } = await fixture(t, { providers });
  const identity = await session(host);
  assert.equal(files.length, 0);
  for (const adapter of adapters) {
    const config = { schema: 1, provider: { kind: adapter.kind, model: adapter.model } };
    const catalogueRevision = providers.catalogue().revision;
    const selected = await post(host, "/speech-providers/selection", { direction: "input", catalogueRevision, config }, identity.cookie);
    const { selectionRevision } = await selected.json() as { selectionRevision: number };
    const query = new URLSearchParams({ config: JSON.stringify(config), catalogueRevision, selectionRevision: String(selectionRevision) });
    const response = await post(host, "/prompt-reader/api-transcriptions?" + query, new Uint8Array(take().buffer), identity.cookie);
    assert.equal(response.status, 200);
    const body = await response.json() as any;
    assert.equal(body.result.text, "unmodified");
    assert.equal(body.pcm.pcmSha256, createCanonicalPcm(take()).pcmSha256);
  }
  assert.equal(files.length, 2, "one request per explicit action; no fanout");
  assert.deepEqual(files[0], new Uint8Array(take().buffer));
  assert.deepEqual(files[1]!.slice(44), new Uint8Array(take().buffer));
});

test("response disconnect aborts API dispatch and a changed selection cancels the old request", async (t) => {
  let started = 0;
  let aborted = 0;
  const inputAdapters: InputApiAdapter[] = [{ kind: "cartesia", model: "fake-model", label: "Fake API",
    async transcribe(_samples, _context, signal) {
      started++;
      await new Promise<void>((resolve) => signal.addEventListener("abort", () => { aborted++; resolve(); }, { once: true }));
      throw new Error("cancelled");
    } }];
  const providers = createHostProviderConfig({ credentials: createProviderCredentials(() => ({ CARTESIA_API_KEY: "fake-key" })),
    readiness, inputAdapters });
  const { host } = await fixture(t, { providers });
  const identity = await session(host);
  const select = () => post(host, "/speech-providers/selection", { direction: "input",
    catalogueRevision: providers.catalogue().revision, config }, identity.cookie);
  for (const cause of ["disconnect", "selection"] as const) {
    const { selectionRevision } = await (await select()).json() as { selectionRevision: number };
    const controller = new AbortController();
    const query = new URLSearchParams({ config: JSON.stringify(config), catalogueRevision: providers.catalogue().revision,
      selectionRevision: String(selectionRevision) });
    const request = fetch(host.url + "/prompt-reader/api-transcriptions?" + query, { method: "POST",
      headers: { origin: host.url, cookie: identity.cookie, "content-type": "application/octet-stream" },
      body: new Blob([take().buffer]), signal: controller.signal });
    const failure = assert.rejects(request);
    await waitFor(() => started > aborted);
    if (cause === "disconnect") controller.abort();
    else assert.equal((await select()).status, 200);
    await failure;
    await waitFor(() => aborted === started);
  }
  assert.equal(aborted, 2);
});

test("saved API replay denies before reading corrupt WAV bytes without any client approval", async (t) => {
  let calls = 0;
  const inputAdapters: InputApiAdapter[] = [{ kind: "cartesia", model: "fake-model", label: "Fake API",
    async transcribe() { calls++; return { text: "allowed" }; } }];
  const credentials = createProviderCredentials(() => ({ CARTESIA_API_KEY: "fake-key" }));
  const providers = createHostProviderConfig({ credentials, readiness, inputAdapters });
  const { host, outputDir } = await fixture(t, { providers });
  const saved = await post(host, "/prompt-reader/recordings?promptId=p001", new Uint8Array(take().buffer));
  const { audio } = await saved.json() as { audio: string };
  await writeFile(join(outputDir, audio), "corrupt, must not be decoded before denial");
  const identity = await session(host);
  const selection = await post(host, "/speech-providers/selection", {
    direction: "input", catalogueRevision: providers.catalogue().revision, config,
  }, identity.cookie);
  const { selectionRevision } = await selection.json() as { selectionRevision: number };
  const value = { config, catalogueRevision: providers.catalogue().revision, selectionRevision,
    source: { kind: "saved", takeName: audio } };
  const denied = await post(host, "/prompt-reader/api-transcriptions", value, identity.cookie);
  assert.equal(denied.status, 403);
  assert.equal(((await denied.json()) as any).code, "SAVED_REPLAY_NOT_AUTHORIZED");
  assert.equal((await post(host, "/prompt-reader/api-transcriptions", { ...value, approved: true }, identity.cookie)).status, 400);
  assert.equal(calls, 0);
});

test("only trusted matching startup grant permits a strict saved API dispatch", async (t) => {
  let calls = 0;
  const inputAdapters: InputApiAdapter[] = [{ kind: "cartesia", model: "fake-model", label: "Fake API",
    async transcribe() { calls++; return { text: "allowed" }; } }];
  const credentials = createProviderCredentials(() => ({ CARTESIA_API_KEY: "fake-key" }));
  const { host, outputDir } = await fixture(t);
  const saved = await post(host, "/prompt-reader/recordings?promptId=p001", new Uint8Array(take().buffer));
  const { audio } = await saved.json() as { audio: string };
  const runId = "trusted-test-run";
  const corpusId = createHash("sha256").update(await realpath(outputDir)).digest("hex");
  const allowedProviders = createHostProviderConfig({ credentials, readiness, inputAdapters, serverRunId: runId,
    replayGrants: [{ grantId: "test-only", ownerDecisionRef: "test-owner-decision", serverRunId: runId,
      scope: { corpusId, takeIds: [audio], inputConfigs: [config], outputConfigs: [] } }] });
  const allowed = await startBrowserVoiceServer({ port: 0, promptReader: {
    promptsPath: join(outputDir, "..", "prompts.json"), outputDir }, providers: allowedProviders });
  t.after(() => allowed.close());
  const allowedIdentity = await session(allowed);
  const allowedSelection = await post(allowed, "/speech-providers/selection", {
    direction: "input", catalogueRevision: allowedProviders.catalogue().revision, config,
  }, allowedIdentity.cookie);
  const revision = await allowedSelection.json() as { selectionRevision: number };
  const accepted = await post(allowed, "/prompt-reader/api-transcriptions", { config, source: { kind: "saved", takeName: audio },
    catalogueRevision: allowedProviders.catalogue().revision, selectionRevision: revision.selectionRevision }, allowedIdentity.cookie);
  assert.equal(accepted.status, 200, await accepted.text());
  assert.equal(calls, 1);
});

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 2000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("host event timeout");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
