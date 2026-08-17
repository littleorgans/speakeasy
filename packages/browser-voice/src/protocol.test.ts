import assert from "node:assert/strict";
import { test } from "node:test";
import type { AudioSegment } from "@speakeasy/convo-engine";
import {
  AUDIO_PACKET_KIND,
  decodeMicFrame,
  encodeAudioPacket,
  parseClientCommand,
} from "./protocol.ts";

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
