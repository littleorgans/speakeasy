import assert from "node:assert/strict";
import { test, type TestContext } from "node:test";
import { startBrowserVoiceServer, type BrowserVoiceServerOptions } from "./host.ts";
import { SETUPS } from "./setup-registry.ts";
import type { SetupExecution, SetupExecutor } from "./setup-protocol.ts";
import { buildSpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { LOCAL_OUTPUT_MODELS, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";

class FakeExecutor implements SetupExecutor {
  calls: SetupExecution[] = [];
  async load() { return { startupMs: 0, loadMs: 0, firstUse: true }; }
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
  for (const asset of ["/road-test.css", "/road-test.js", "/road-test-text.js", "/mic-capture.js", "/voice-readback.js", "/talk-over.js"]) {
    assert.equal((await fetch(road.url + asset)).status, 200, asset);
  }
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
