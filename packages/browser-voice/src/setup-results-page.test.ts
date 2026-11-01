import assert from "node:assert/strict";
import { test } from "node:test";
import { pageHarness, okReply, readyTake, enableComparison, streamReply, rowEvent, acceptStream, type FetchInit } from "./prompt-reader-page-harness.ts";

const configs = [
  { schema: 1, provider: { kind: "cartesia", model: "ink-whisper" } },
  { schema: 1, provider: { kind: "openai", model: "gpt-transcribe" } },
];
const clone = (value: any) => JSON.parse(JSON.stringify(value));
const settle = () => new Promise((resolve) => setImmediate(resolve));

function apiHarness({ mutate = (_body: any) => {}, selection = (_body: any) => {}, session = (_body: any) => {}, hold = false } = {}) {
  const requests: Array<{ url: string; init?: FetchInit }> = [];
  const stream = streamReply();
  let localStream = stream;
  let release: (() => void) | undefined;
  const h = pageHarness({ respond: async (url, init) => {
    requests.push({ url, init });
    if (url.startsWith("/prompt-reader/comparisons")) return localStream.reply;
    if (url === "/speech-providers/session") {
      const body = { sessionId: "protected-session" };
      session(body);
      return okReply(body);
    }
    if (url === "/speech-providers/selection") {
      const body = { ...JSON.parse(init!.body as string), sessionId: "protected-session", selectionRevision: 3 };
      selection(body);
      return okReply(body);
    }
    if (url.startsWith("/prompt-reader/api-transcriptions")) {
      const query = new URLSearchParams(url.split("?")[1]);
      const count = (init!.body as ArrayBuffer).byteLength / 4;
      const body = {
        requestId: query.get("requestId"), config: JSON.parse(query.get("config")!),
        source: { kind: "live-capture", sessionId: "protected-session", operationId: "host-operation" },
        pcm: { sampleRate: 16000, channels: 1, sampleCount: count, pcmBytes: count * 4, pcmSha256: "a".repeat(64) },
        result: { text: "raw API text\n", timings: {
          providerRequestWallMs: { status: "measured", ms: 0, clock: "server", boundary: "provider request" },
          providerProcessingMs: { status: "provider-reported", ms: 12, metric: "processing", boundary: "provider compute", evidenceRef: "fixture" },
          providerNetworkMs: { status: "unavailable", reason: "not exposed" },
        }, opaqueAdapterField: { preserved: true } },
      };
      mutate(body);
      if (hold) await new Promise<void>((resolve) => { release = resolve; });
      return okReply(body);
    }
    return okReply({ audio: "utt-p001-a.wav" });
  } });
  enableComparison(h.reader);
  h.reader.comparison.providers = { revision: "provider-revision", input: [
    ...configs.map((config) => ({ id: config.provider.kind, label: config.provider.kind, selectable: true, config: clone(config) })),
    { id: "hidden", selectable: false, config: { schema: 1, provider: { kind: "openai", model: "unavailable" } } },
  ] };
  h.reader.comparison.renderProviders();
  const select = (kind: string) => {
    h.reader.comparison.elements["input-provider"].value = kind;
    h.dispatch("input-provider:change");
  };
  return { ...h, requests, stream, select, nextLocal: () => (localStream = streamReply()), release: () => release!() };
}

for (const config of configs) test(`${config.provider.kind} API identity replaces a preceding local result and retains the exact adapter result`, async () => {
  const h = apiHarness();
  await readyTake(h);
  h.reader.comparison.mode = "compare";
  const local = h.reader.comparison.run();
  const localId = h.reader.comparison.job.requestId;
  acceptStream(h.stream, localId);
  h.stream.send({ ...rowEvent(localId, "mlx-parakeet/v3-greedy"), manifestSha256: "old-manifest" });
  h.stream.send({ type: "done", requestId: localId, comparisonId: "server-job", status: "complete" });
  h.stream.end();
  await local;
  assert.equal(h.reader.comparison.rows[0].result.text, "raw words\n");
  h.select(config.provider.kind);
  // The second take is an offset view. Never upload the surrounding backing bytes.
  const backing = new Float32Array([9, 0.25, -0.5, 8]);
  h.reader.take = { ...h.reader.take, samples: backing.subarray(1, 3) };
  const running = h.reader.comparison.run();
  backing.fill(7);
  await running;
  const row = h.reader.comparison.rows[0];
  const upload = h.requests.find((entry) => entry.url.startsWith("/prompt-reader/api-transcriptions"))!;
  const query = new URLSearchParams(upload.url.split("?")[1]);
  assert.deepEqual(Array.from(new Float32Array(upload.init!.body as ArrayBuffer)), [0.25, -0.5]);
  assert.deepEqual(clone(row.identity), {
    requestId: query.get("requestId"), config, source: { kind: "live-capture", sessionId: "protected-session", operationId: "host-operation" },
    pcm: { sampleRate: 16000, channels: 1, sampleCount: 2, pcmBytes: 8, pcmSha256: "a".repeat(64) },
    catalogueRevision: "provider-revision", selectionRevision: 3,
  });
  assert.equal(row.result.opaqueAdapterField.preserved, true);
  assert.equal(row.result.text, "raw API text\n");
  const details = JSON.parse(row.view.identity.textContent);
  assert.equal(details.manifestSha256, null);
  assert.deepEqual(details.identity, clone(row.identity));
  assert.doesNotMatch(row.view.identity.textContent, /same-digest|old-manifest/);
  assert.match(row.view.timing.textContent, /0 ms.*measured.*server/);
  assert.match(row.view.timing.textContent, /provider reported.*processing/);
  assert.match(row.view.timing.textContent, /Provider network: Unavailable: not exposed/);
  assert.equal(row.view.text.children.map((node: any) => node.textContent).join(""), "raw API text\n");
  h.select("local");
  const nextStream = h.nextLocal();
  const nextLocal = h.reader.comparison.run();
  const nextId = h.reader.comparison.job.requestId;
  assert.equal(h.reader.comparison.resultIdentity, null);
  assert.equal(h.reader.comparison.resultManifest, null);
  nextStream.send({ type: "accepted", requestId: nextId, comparisonId: "server-job", pcm: { sampleCount: 2, pcmSha256: "next-local-digest" }, manifestSha256: "next-local-manifest" });
  nextStream.send(rowEvent(nextId, "mlx-parakeet/v3-greedy"));
  nextStream.send({ type: "done", requestId: nextId, comparisonId: "server-job", status: "complete" });
  nextStream.end();
  await nextLocal;
  const localDetails = JSON.parse(h.reader.comparison.rows[0].view.identity.textContent);
  assert.equal(localDetails.manifestSha256, "next-local-manifest");
  assert.equal(localDetails.identity, undefined);
  assert.equal(localDetails.adapterResult, undefined);
  h.reader.comparison.invalidate();
  assert.equal(h.reader.comparison.resultIdentity, null);
  assert.equal(h.reader.comparison.resultManifest, null);
});

test("API metadata, capture and selection remain inert; deliberate input dispatch is single and saved takes stay denied", async () => {
  const h = apiHarness();
  assert.deepEqual(h.reader.comparison.elements["input-provider"].children.map((option: any) => option.value), ["local", "cartesia", "openai"]);
  h.select("cartesia");
  await readyTake(h);
  assert.equal(h.requests.length, 0);
  await h.reader.comparison.run();
  assert.equal(h.requests.length, 3);
  assert.deepEqual(JSON.parse(h.requests[0]!.init!.body as string), {});
  assert.equal(JSON.parse(h.requests[1]!.init!.body as string).direction, "input");
  h.reader.take = null;
  h.reader.prompts[0].takes = ["saved.wav"];
  await h.reader.comparison.run();
  assert.equal(h.requests.length, 3);
  assert.match(h.reader.comparison.message, /not authorized/);
});

for (const [name, mutate] of Object.entries({
  request: (body: any) => { body.requestId = "other"; },
  config: (body: any) => { body.config.provider.model = "other"; },
  source: (body: any) => { body.source.kind = "saved-take"; },
  session: (body: any) => { body.source.sessionId = "other"; },
  operation: (body: any) => { body.source.operationId = "  "; },
  rate: (body: any) => { body.pcm.sampleRate = 48000; },
  channels: (body: any) => { body.pcm.channels = 2; },
  count: (body: any) => { body.pcm.sampleCount++; },
  bytes: (body: any) => { body.pcm.pcmBytes++; },
  digest: (body: any) => { body.pcm.pcmSha256 = "wrong"; },
  text: (body: any) => { body.result.text = null; },
})) test(`mismatched API ${name} fails without borrowing local identity or blocking Play/Save`, async () => {
  const h = apiHarness({ mutate });
  h.select("openai");
  const take = await readyTake(h);
  h.reader.comparison.resultIdentity = { pcmSha256: "old-local" };
  h.reader.comparison.resultManifest = "old-manifest";
  await h.reader.comparison.run();
  assert.equal(h.reader.comparison.rows[0].phase, "error");
  assert.equal(h.reader.comparison.rows[0].result, null);
  assert.equal(h.reader.comparison.rows[0].identity, undefined);
  assert.doesNotMatch(h.reader.comparison.rows[0].view.identity.textContent, /old-local|old-manifest/);
  await h.reader.play();
  assert.deepEqual(Array.from(h.played[0]!.samples), Array.from(take.samples));
  assert.equal(h.reader.elements.save.disabled, false);
  await h.reader.save();
  assert.equal(h.requests.at(-1)!.init!.body, take.samples.buffer);
});

test("cancelled API response cannot replace a later local job", async () => {
  const h = apiHarness({ hold: true });
  h.select("cartesia");
  await readyTake(h);
  const api = h.reader.comparison.run();
  await settle();
  h.reader.comparison.cancel();
  h.select("local");
  const local = h.reader.comparison.run();
  const id = h.reader.comparison.job.requestId;
  acceptStream(h.stream, id);
  h.stream.send(rowEvent(id, "mlx-parakeet/v3-greedy"));
  h.stream.send({ type: "done", requestId: id, comparisonId: "server-job", status: "complete" });
  h.stream.end();
  await local;
  h.release();
  await api;
  assert.equal(h.reader.comparison.rows[0].result.text, "raw words\n");
  assert.equal(h.reader.comparison.rows[0].identity, undefined);
  assert.equal(h.requests[2]!.init!.signal!.aborted, true);
});

for (const [name, selection] of Object.entries({
  direction: (body: any) => { body.direction = "output"; },
  config: (body: any) => { body.config.provider.model = "other"; },
  session: (body: any) => { body.sessionId = "other-session"; },
  catalogue: (body: any) => { body.catalogueRevision = "stale"; },
  revision: (body: any) => { body.selectionRevision = 0; },
})) test(`invalid input selection ${name} stops before upload`, async () => {
  const h = apiHarness({ selection });
  h.select("cartesia");
  await readyTake(h);
  await h.reader.comparison.run();
  assert.equal(h.requests.length, 2);
  assert.equal(h.reader.comparison.rows[0].phase, "error");
  assert.equal(h.reader.comparison.rows[0].identity, undefined);
});

test("invalid protected session stops before selecting or uploading", async () => {
  const h = apiHarness({ session: (body) => { body.sessionId = " "; } });
  h.select("cartesia");
  await readyTake(h);
  await h.reader.comparison.run();
  assert.equal(h.requests.length, 1);
  assert.equal(h.reader.comparison.rows[0].phase, "error");
});

test("an empty API transcript remains a completed result with its own identity", async () => {
  const h = apiHarness({ mutate: (body) => { body.result.text = ""; } });
  h.select("openai");
  await readyTake(h);
  await h.reader.comparison.run();
  assert.equal(h.reader.comparison.rows[0].phase, "complete");
  assert.equal(h.reader.comparison.rows[0].adapterResult.text, "");
  assert.equal(h.reader.comparison.rows[0].view.text.textContent, "No speech detected");
  assert.equal(h.reader.comparison.rows[0].identity.config.provider.kind, "openai");
});
