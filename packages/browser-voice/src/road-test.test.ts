import assert from "node:assert/strict";
import { test, type TestContext } from "node:test";
import { WebSocket } from "ws";
import { startBrowserVoiceServer, type BrowserVoiceServerOptions } from "./host.ts";
import { SETUPS } from "./setup-registry.ts";
import type { SetupExecution, SetupExecutor, SetupId, SetupLoadContext } from "./setup-protocol.ts";
import { buildSpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { LOCAL_OUTPUT_MODELS, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";
import { TextCleanups, type CleanupWorker } from "./text-cleanup.ts";
import type { TextCleanupInput } from "./setup-workers.ts";
import { DEFAULT_NAMES, NamesStore } from "./names.ts";

class FakeExecutor implements SetupExecutor {
  calls: SetupExecution[] = [];
  fed: [SetupId, number, boolean][] = [];
  settings: unknown[] = [];
  loads: [SetupId, SetupLoadContext["names"]][] = [];
  async feed(setupId: SetupId, samples: Float32Array, start: boolean, _context?: unknown, settings?: unknown) {
    this.fed.push([setupId, samples.length, start]);
    this.settings.push(settings);
    return { stable: " Clear the", tentative: " scream", processingMs: 3 };
  }
  async load(setupId: SetupId, { names }: SetupLoadContext) {
    this.loads.push([setupId, names]);
    return { startupMs: 0, loadMs: 0, firstUse: true };
  }
  async execute(request: SetupExecution) {
    this.calls.push(request);
    return { text: "Clear the screen.", processingMs: 1, workerWallMs: 2 };
  }
  async release() {}
  async close() {}
}

async function host(t: TestContext, options: BrowserVoiceServerOptions = {}) {
  const executor = new FakeExecutor();
  const server = await startBrowserVoiceServer({ port: 0, setupExecutor: executor,
    setupReadiness: () => SETUPS.map((setup) => ({ setupId: setup.id, ready: true, sizeBytes: 1_250_000_000 })), ...options });
  t.after(() => server.close());
  return { server, executor };
}

test("the road test flag serves the road test page at the root; the room keeps the default", async (t) => {
  const road = (await host(t, { home: "road-test" })).server;
  const room = (await host(t)).server;
  const page = await fetch(road.url + "/");
  assert.equal(page.status, 200);
  assert.match(await page.text(), /<title>Speak Easy · Road test<\/title>/);
  assert.match(await (await fetch(room.url + "/")).text(), /<title>[^<]*<\/title>/);
  assert.doesNotMatch(await (await fetch(room.url + "/")).text(), /Road test/);
  assert.match(await (await fetch(room.url + "/road-test")).text(), /Road test/, "the road test stays reachable on any host");
  for (const asset of ["/road-test.css", "/road-test.js", "/road-test-text.js", "/capture-session.js", "/voice-readback.js", "/talk-over.js",
    "/text-cleanup.js"]) {
    assert.equal((await fetch(road.url + asset)).status, 200, asset);
  }
});

// Each page script, with a module only its nested imports reach.
for (const [page, nested] of [["/road-test.js", "/live-trail.js"], ["/app.js", "/speech-observations.js"], ["/prompt-reader.js", "/speech-observations.js"]])
  test(`the host serves every module ${page} imports`, async (t) => {
    const { server } = await host(t, { home: "road-test" });
    const seen = new Set<string>();
    const visit = async (path: string): Promise<void> => {
      if (seen.has(path)) return;
      seen.add(path);
      const response = await fetch(server.url + path);
      assert.equal(response.status, 200, path);
      for (const [, child] of (await response.text()).matchAll(/^import .* from "(\/[^"]+)";$/gm)) await visit(child!);
    };
    await visit(page);
    assert.ok(seen.has(nested) && seen.size > 3, [...seen].join(" "));
  });

test("a live take transcribes without a prompt reader; saved takes still need one", async (t) => {
  const { server, executor } = await host(t, { home: "road-test" });
  const query = new URLSearchParams({ requestId: "road-1", setupIds: JSON.stringify(["mlx-parakeet/ultra-greedy"]) });
  const samples = Float32Array.from({ length: 1_600 }, (_, index) => Math.sin(index / 8) / 4);
  const response = await fetch(`${server.url}/prompt-reader/comparisons?${query}`, { method: "POST",
    headers: { origin: server.url, "content-type": "application/octet-stream" }, body: new Blob([samples.buffer]) });
  assert.equal(response.status, 200);
  const rows = (await response.text()).trim().split("\n").map((line) => JSON.parse(line)).filter((event) => event.type === "row");
  assert.equal(rows.at(-1).row.status, "complete");
  assert.equal(rows.at(-1).row.text, "Clear the screen.");
  assert.deepEqual(executor.calls.map((call) => call.setupId), ["mlx-parakeet/ultra-greedy"]);
  const saved = await fetch(`${server.url}/prompt-reader/comparisons`, { method: "POST",
    headers: { origin: server.url, "content-type": "application/json" },
    body: JSON.stringify({ requestId: "road-2", setupIds: ["mlx-parakeet/ultra-greedy"], source: { kind: "saved", takeName: "x.wav" } }) });
  assert.equal(saved.status, 404);
});

test("cleanup models are listed and clean a final's text through the host, holding no audio lease", async (t) => {
  let open!: () => void;
  const held = new Promise<void>((resolve) => { open = resolve; });
  const inputs: TextCleanupInput[] = [];
  const worker: CleanupWorker = {
    load: async () => ({ startupMs: null, loadMs: 30, firstUse: true }),
    cleanText: async (input) => {
      inputs.push(input);
      await held;
      return { text: "Ship it.", processingMs: 4, metadata: {}, firstUse: true, startupLoadWallMs: 30, workerWallMs: 35 };
    },
    release: async () => {}, close: async () => {},
  };
  const textCleanups = new TextCleanups({ workers: { "apple/foundation-models": worker },
    readiness: () => ({ "apple/foundation-models": { ready: true } }) });
  const { server, executor } = await host(t, { home: "road-test", textCleanups });
  const listing = await (await fetch(`${server.url}/prompt-reader/cleanups`)).json() as { schema: number;
    models: { id: string; readiness: { ready: boolean; reason?: string }; settings: { default: unknown }[] }[] };
  assert.equal(listing.schema, 1);
  assert.deepEqual(listing.models.map((model) => [model.id, model.readiness.ready]),
    [["superwhisper/s1-mini", false], ["apple/foundation-models", true]]);
  assert.equal(listing.models[0].readiness.reason, "runtime not configured");
  const post = (body: unknown, origin = server.url) => fetch(`${server.url}/prompt-reader/cleanups`, { method: "POST",
    headers: { origin, "content-type": "application/json" }, body: JSON.stringify(body) });
  const pending = post({ requestId: "clean-1", modelId: "apple/foundation-models", text: "um ship it" });
  // The cleanup holds no lease: a comparison runs to completion while it waits.
  const query = new URLSearchParams({ requestId: "road-3", setupIds: JSON.stringify(["mlx-parakeet/ultra-greedy"]) });
  const comparison = await fetch(`${server.url}/prompt-reader/comparisons?${query}`, { method: "POST",
    headers: { origin: server.url, "content-type": "application/octet-stream" }, body: new Blob([new Float32Array(1_600).buffer]) });
  assert.equal(comparison.status, 200);
  await comparison.text();
  assert.equal(executor.calls.length, 1);
  open();
  const reply = await pending;
  assert.equal(reply.status, 200);
  assert.deepEqual(await reply.json(), { requestId: "clean-1", modelId: "apple/foundation-models", status: "complete",
    text: "Ship it.", processingMs: 4, workerWallMs: 35, loadMs: 30, firstUse: true,
    settings: { instruction: listing.models[1].settings[0].default, earlier: false } });
  assert.deepEqual(inputs[0]!.names, DEFAULT_NAMES, "a request without names gets the current list");
  await (await post({ modelId: "apple/foundation-models", text: "um", names: [" Qwen ", "Qwen", "", "TM"] })).text();
  assert.deepEqual(inputs[1]!.names, ["Qwen", "TM"], "the take's names, canonical");
  const refusedNames = await post({ modelId: "apple/foundation-models", text: "um", names: ["Grok, Claude"] });
  assert.equal(refusedNames.status, 400);
  assert.match((await refusedNames.json() as { error: string }).error, /"Grok, Claude" contains a comma/);
  assert.equal((await post({ modelId: "superwhisper/s1-mini", text: "um", names: ["Grok"] })).status, 400, "S1-mini takes no names");
  assert.equal((await post({ modelId: "apple/foundation-models", text: "um", names: Array.from({ length: 64 }, (_, index) => `${index}`.padEnd(128, "n")) })).status, 200, "the route's JSON bound fits names at their bounds");
  const skipped = await (await post({ modelId: "superwhisper/s1-mini", text: "um" })).json() as Record<string, unknown>;
  assert.deepEqual([skipped.status, skipped.code, skipped.reason], ["skipped", "unavailable", "runtime not configured"]);
  assert.equal((await post({ modelId: "superwhisper/s1-mini", text: "um", settings: {} })).status, 400);
  assert.equal((await post({ modelId: "apple/foundation-models", text: "x".repeat(16_384) })).status, 413);
  assert.equal((await post({ modelId: "apple/foundation-models", text: "x" }, "http://evil.example")).status, 403);
});

test("the names route answers the list, saves a valid edit and refuses an invalid one unchanged", async (t) => {
  const { server, executor } = await host(t, { home: "road-test", names: await NamesStore.open() });
  const route = `${server.url}/road-test/names`;
  const put = (body: unknown, origin = server.url) => fetch(route, { method: "PUT",
    headers: { origin, "content-type": "application/json" }, body: JSON.stringify(body) });
  const fresh = await (await fetch(route)).json() as { names: string[]; digest: string; limits: unknown };
  assert.deepEqual(fresh, { names: [...DEFAULT_NAMES],
    digest: "59d6626926c652e50c7e90fea5f95afdea75f6c27e11aa07c2996b89d51fc068", limits: { maxNames: 64, maxNameBytes: 128 } });
  const saved = await put({ names: [" Qwen", "GPT-SOL ", "", "Qwen"] });
  assert.equal(saved.status, 200);
  assert.deepEqual((await saved.json() as { names: string[] }).names, ["Qwen", "GPT-SOL"], "trimmed, blanks and repeats removed");
  for (const names of ["Grok", ["Grok", 3], Array.from({ length: 65 }, (_, index) => `n${index}`), ["é".repeat(65)],
    ["Gr\u0000ok"], ["Gr\u0085ok"], ["Grok, Claude"]]) {
    const refused = await put({ names });
    assert.equal(refused.status, 400, JSON.stringify(names));
    assert.ok((await refused.json() as { error: string }).error);
  }
  assert.equal((await put({ names: [], extra: 1 })).status, 400, "unsupported fields");
  assert.equal((await put({ names: [] }, "http://evil.example")).status, 403);
  assert.equal((await fetch(route, { method: "PUT", headers: { "content-type": "application/json" },
    body: JSON.stringify({ names: [] }) })).status, 403, "a write needs the page's origin");
  assert.equal((await fetch(route, { headers: { origin: "http://evil.example" } })).status, 403);
  const { limits: _limits, ...kept } = await (await fetch(route)).json() as { names: string[]; digest: string; limits: unknown };
  assert.deepEqual(kept.names, ["Qwen", "GPT-SOL"], "a refused edit changes nothing");
  // The next take uses the saved list: accepted echoes it, and only names setups load and run with it.
  const query = new URLSearchParams({ requestId: "names-1", setupIds: JSON.stringify(["mlx-parakeet/v3-greedy", "cohere/transcribe"]) });
  const events = (await (await fetch(`${server.url}/prompt-reader/comparisons?${query}`, { method: "POST",
    headers: { origin: server.url, "content-type": "application/octet-stream" }, body: new Blob([new Float32Array(1_600).buffer]) }))
    .text()).trim().split("\n").map((line) => JSON.parse(line));
  assert.deepEqual(events[0].names, kept);
  assert.deepEqual(executor.loads, [["mlx-parakeet/v3-greedy", undefined], ["cohere/transcribe", kept]]);
  assert.deepEqual(executor.calls.map((call) => call.names), [undefined, kept]);
  const rows = events.filter((event) => event.type === "row" && event.row.status === "complete").map((event) => event.row);
  assert.deepEqual(rows.map((row) => [row.setupId, row.namesDigest]), [["mlx-parakeet/v3-greedy", undefined], ["cohere/transcribe", kept.digest]]);
});

test("a Cohere take carries its names setting to the executor and its row; values not offered are refused", async (t) => {
  const { server, executor } = await host(t, { home: "road-test" });
  const metadata = await (await fetch(server.url + "/prompt-reader/setups")).json() as { setups: any[] };
  assert.deepEqual(metadata.setups.filter((setup) => setup.final).map((setup) => setup.id), ["cohere/transcribe"]);
  assert.equal(metadata.setups.find((setup) => setup.id === "cohere/transcribe").final.settings[0].label, "Names");
  const samples = Float32Array.from({ length: 1_600 }, (_, index) => Math.sin(index / 8) / 4);
  const post = (settings?: unknown, setupIds = ["cohere/transcribe"]) => fetch(`${server.url}/prompt-reader/comparisons?${new URLSearchParams({
    requestId: `road-${executor.calls.length}-${Math.random().toString(36).slice(2)}`, setupIds: JSON.stringify(setupIds),
    ...(settings === undefined ? {} : { settings: JSON.stringify(settings) }) })}`, { method: "POST",
    headers: { origin: server.url, "content-type": "application/octet-stream" }, body: new Blob([samples.buffer]) });
  const gentle = await post({ "cohere/transcribe": { namesBias: 1.25 } });
  assert.equal(gentle.status, 200);
  const rows = (await gentle.text()).trim().split("\n").map((line) => JSON.parse(line)).filter((event) => event.type === "row");
  assert.deepEqual(rows.at(-1).row.settings, { namesBias: 1.25 }, "the row records the value used");
  assert.deepEqual(executor.calls.map((call) => call.operation === "transcribe" && call.settings), [{ namesBias: 1.25 }]);
  await (await post()).text();
  const off = executor.calls.at(-1)!;
  assert.deepEqual(off.operation === "transcribe" && off.settings, { namesBias: 0 }, "Off unless chosen");
  for (const [settings, setupIds] of [[{ "cohere/transcribe": { namesBias: 1 } }, ["cohere/transcribe"]],
    [{ "cohere/transcribe": { namesBias: 2.5 } }, ["hex/unified"]], [[2.5], ["cohere/transcribe"]]] as const) {
    const refused = await post(settings, [...setupIds]);
    assert.equal(refused.status, 400, JSON.stringify(settings));
    await refused.text();
  }
  assert.equal(executor.calls.length, 2, "refused takes never run");
});

test("every speech in setup carries a model card, and metadata adds the measured size", async (t) => {
  for (const setup of SETUPS) {
    assert.ok(setup.card.runtime.trim(), setup.id);
    assert.ok(["final", "partials"].includes(setup.card.transcript), setup.id);
  }
  const { server } = await host(t);
  const metadata = await (await fetch(server.url + "/prompt-reader/setups")).json() as { setups: any[] };
  assert.deepEqual(metadata.setups.map((setup: { card: unknown }) => setup.card), SETUPS.map((setup) => setup.card));
  assert.ok(metadata.setups.every((setup: { readiness: { sizeBytes: number } }) => setup.readiness.sizeBytes === 1_250_000_000));
});

test("local voices publish their runtime and pinned payload size for the card", () => {
  const catalogue = buildSpeechOutputCatalogue({ localRegistry: LOCAL_OUTPUT_REGISTRY, local: {
    default: { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 },
    models: LOCAL_OUTPUT_MODELS.map((model) => ({ ...model, readiness: { state: "ready" as const, reason: null } })),
  } });
  const kristin = catalogue.models.find((model) => model.id === "piper-kristin-medium")!;
  assert.equal(kristin.runtime, "sherpa-onnx VITS, CPU");
  assert.equal(kristin.sizeBytes, 63_531_507 + 940 + 4_968);
  assert.ok(catalogue.models.every((model) => model.runtime && model.sizeBytes! > 0));
});

test("live words stream mic frames to a streaming setup's model and answer with its words, tagged by take", async (t) => {
  const { server, executor } = await host(t, { home: "road-test" });
  const socket = new WebSocket(server.url.replace("http", "ws") + "/live-words", { headers: { origin: server.url } });
  t.after(() => socket.close());
  const events: any[] = [];
  socket.on("message", (data) => events.push(JSON.parse(data.toString())));
  await new Promise((resolve) => socket.once("open", resolve));
  const next = async (count: number) => {
    for (let tries = 0; events.length < count; tries++) {
      if (tries > 200) throw new Error(`waited for ${count} events, have ${JSON.stringify(events)}`);
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
  };
  socket.send(JSON.stringify({ type: "begin", takeId: "t1", setupId: "qwen/context" }));
  await next(1);
  assert.deepEqual(events[0], { type: "error", takeId: null, message: "Qwen3 ASR with known words gives final text only" });
  socket.send(JSON.stringify({ type: "begin", takeId: "t2", setupId: "mlx-cleanup/qwen-targeted" }));
  await next(2);
  assert.equal(events[1].message, "Ultra plus Qwen cleanup gives final text only",
    "the page names the live model itself; a cleanup setup's is its base");
  socket.send(JSON.stringify({ type: "begin", takeId: "t3", setupId: "mlx-parakeet/ultra-greedy", settings: { updateMs: 1_200 } }));
  socket.send(new Float32Array(320));
  await next(6);
  assert.deepEqual(events[2], { type: "stream", takeId: "t3", setupId: "mlx-parakeet/ultra-greedy",
    settings: { updateMs: 1_200 } }, "the host names the stream and every setting it applies");
  assert.deepEqual(events.slice(3).map((event) => [event.type, event.takeId]), [["loading", "t3"], ["ready", "t3"], ["partial", "t3"]]);
  assert.equal(events[5].stable + events[5].tentative, " Clear the scream");
  assert.deepEqual(executor.fed, [["mlx-parakeet/ultra-greedy", 320, true]]);
  assert.deepEqual(executor.settings, [{ updateMs: 1_200 }]);
  socket.send(JSON.stringify({ type: "end", takeId: "t3" }));
  socket.send(new Float32Array(320));
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(executor.fed.length, 1, "frames after end are dropped");
  for (const [settings, message] of [[{ updateMs: 320 }, "unsupported updateMs"], [{ lookaheadMs: 480 }, "unknown live setting lookaheadMs"]]) {
    const count = events.length;
    socket.send(JSON.stringify({ type: "begin", takeId: "t4", setupId: "mlx-parakeet/v3-greedy", settings }));
    await next(count + 1);
    assert.equal(events[count].message, message);
  }
  socket.send("{\"type\":\"begin\"}");
  await next(events.length + 1);
  assert.equal(events.at(-1).message, "takeId required");
});
