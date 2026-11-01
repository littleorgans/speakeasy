import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import type { AudioSegment } from "@speakeasy/convo-engine";
import {
  AUDIO_PACKET_KIND,
  decodeMicFrame,
  encodeAudioPacket,
  parseClientCommand,
  RESPONSE_ENGINES,
} from "./protocol.ts";
import { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";

test("room engine options match the protocol engines", async () => {
  const html = await readFile(
    new URL("../public/index.html", import.meta.url),
    "utf8",
  );
  const options = html
    .match(/<select id="engine-select">([\s\S]*?)<\/select>/)?.[1]
    ?.matchAll(/<option value="([^"]+)"/g);
  assert.ok(options);
  assert.deepEqual(
    [...options].map((match) => match[1]),
    [...RESPONSE_ENGINES],
  );
});

test("parseClientCommand accepts the bounded browser command set", () => {
  assert.deepEqual(
    parseClientCommand(
      '{"type":"start","engine":"mercury-instant","mode":"natural","pauseMs":700,"voice":"marin","barge":true,"systemPrompt":"  Be kind.  "}',
    ),
    {
      type: "start",
      engine: "mercury-instant",
      mode: "natural",
      pauseMs: 700,
      voice: "marin",
      barge: true,
      systemPrompt: "Be kind.",
    },
  );
  assert.deepEqual(parseClientCommand('{"type":"interrupt"}'), { type: "interrupt" });
  assert.deepEqual(parseClientCommand('{"type":"commit-input"}'), { type: "commit-input" });
  assert.deepEqual(parseClientCommand('{"type":"playback-drained","playbackId":4,"audioEndMs":375}'), {
    type: "playback-drained",
    playbackId: 4,
    audioEndMs: 375,
  });
  assert.throws(() => parseClientCommand('{"type":"unknown"}'), /unknown command/);
  assert.throws(
    () => parseClientCommand('{"type":"start","engine":"realtime","mode":"natural","pauseMs":199,"voice":"marin"}'),
    /pauseMs must be 200 to 3000/,
  );
  assert.throws(
    () => parseClientCommand('{"type":"start","engine":"realtime","mode":"natural","pauseMs":700,"voice":"unknown"}'),
    /Unsupported Realtime voice/,
  );
  assert.throws(
    () => parseClientCommand('{"type":"start","engine":"unknown","mode":"natural","pauseMs":700,"voice":"marin"}'),
    /requires engine/,
  );
  assert.throws(
    () => parseClientCommand('{"type":"playback-drained","playbackId":-1}'),
    /non-negative playbackId/,
  );
  assert.throws(
    () => parseClientCommand('{"type":"playback-drained","playbackId":4,"audioEndMs":-1}'),
    /non-negative integer/,
  );
});

test("decodeMicFrame owns and validates Float32 PCM", () => {
  const original = new Float32Array([0.1, -0.2, 0.3]);
  const decoded = decodeMicFrame(new Uint8Array(original.buffer));
  original[0] = 0.9;
  assert.ok(Math.abs(decoded[0]! - 0.1) < 0.00001);

  const invalid = new Float32Array([Number.NaN]);
  assert.throws(
    () => decodeMicFrame(new Uint8Array(invalid.buffer)),
    /non-finite sample/,
  );
});

test("encodeAudioPacket carries playback identity, rate, and samples", () => {
  const segment: AudioSegment = {
    index: 0,
    sentence: "hello",
    samples: new Float32Array([0.25, -0.5]),
    sampleRate: 24_000,
    readyAtMs: 4,
    synthMs: 2,
    audioDurationMs: 1,
  };
  const packet = encodeAudioPacket(7, segment);
  const header = new DataView(packet.buffer);
  assert.equal(header.getUint8(0), AUDIO_PACKET_KIND);
  assert.equal(header.getUint32(1, true), 7);
  assert.equal(header.getUint32(5, true), 24_000);
  assert.deepEqual(
    [...new Float32Array(packet.buffer.slice(9))],
    [0.25, -0.5],
  );
});

test("optional room speechOutput uses the shared registry while old start remains compatible", () => {
  const start = { type: "start", engine: "mercury-low", mode: "hold", pauseMs: 700, voice: "cedar" };
  const parsed = parseClientCommand(JSON.stringify({ ...start, speechOutput: DEFAULT_LOCAL_OUTPUT }), LOCAL_OUTPUT_REGISTRY);
  assert.ok(parsed.type === "start");
  assert.deepEqual(parsed.speechOutput, DEFAULT_LOCAL_OUTPUT);
  for (const speechOutput of [{ ...DEFAULT_LOCAL_OUTPUT, voice: "not-a-voice" },
    { ...DEFAULT_LOCAL_OUTPUT, endpoint: "https://foreign.example" }]) {
    assert.throws(() => parseClientCommand(JSON.stringify({ ...start, speechOutput }), LOCAL_OUTPUT_REGISTRY));
  }
});

test("PCM decoder owns offset Buffer views and preserves finite out of range values and signed zero", () => {
  const buffer = Buffer.alloc(17);
  const data = new DataView(buffer.buffer, buffer.byteOffset + 1, 12);
  [-0, 1.25, -2].forEach((sample, index) => data.setFloat32(index * 4, sample, true));
  const result = decodeMicFrame(buffer.subarray(1, 13));
  assert.equal(Object.is(result[0], -0), true);
  assert.deepEqual([...result.slice(1)], [1.25, -2]);
  buffer.fill(0);
  assert.equal(result[1], 1.25);
});

test("room selection reuses strict output parsing and excludes trial commands", () => {
  const command = { type: "select-output", requestId: "room_1", catalogueRevision: "revision",
    selection: DEFAULT_LOCAL_OUTPUT };
  const parse = (value: unknown) => parseClientCommand(JSON.stringify(value), LOCAL_OUTPUT_REGISTRY);
  assert.deepEqual(parse(command), command);
  for (const value of [
    { ...command, source: { kind: "typed-text" } },
    { ...command, requestId: "bad id" },
    { ...command, requestId: "x".repeat(97) },
    { ...command, catalogueRevision: "x".repeat(129) },
    { ...command, selection: { ...DEFAULT_LOCAL_OUTPUT, voice: "missing" } },
    { ...command, type: "speak-sample", text: "hello" },
    { type: "cancel-sample", requestId: "trial" },
  ]) assert.throws(() => parse(value));
});

test("room start binds paired bounded revisions and rejects client authority", () => {
  const command = { type: "start", engine: "mercury-low", mode: "hold", pauseMs: 700,
    voice: "cedar", speechOutput: DEFAULT_LOCAL_OUTPUT, catalogueRevision: "current", selectionRevision: 3 };
  const parse = (value: unknown) => parseClientCommand(JSON.stringify(value), LOCAL_OUTPUT_REGISTRY);
  const parsed = parse(command);
  assert.ok(parsed.type === "start");
  assert.equal(parsed.catalogueRevision, "current");
  assert.equal(parsed.selectionRevision, 3);
  for (const fields of [
    { catalogueRevision: undefined }, { selectionRevision: undefined },
    { catalogueRevision: " " }, { catalogueRevision: "x".repeat(129) },
    { selectionRevision: 0 }, { selectionRevision: 1.1 }, { selectionRevision: Number.MAX_SAFE_INTEGER + 1 },
    { sessionId: "client" }, { source: { kind: "live-response-text" } },
    { grantId: "client" }, { approval: true },
  ]) assert.throws(() => parse({ ...command, ...fields }));
  const hosted = { ...DEFAULT_LOCAL_OUTPUT, provider: { kind: "cartesia" }, model: "sonic-test", voice: "katie" };
  assert.throws(() => parseClientCommand(JSON.stringify({ ...command, speechOutput: hosted,
    catalogueRevision: undefined, selectionRevision: undefined }),
  [{ kind: "cartesia", model: hosted.model, voices: ["katie"], speed: { min: 0.6, max: 1.5 } }]));
});
