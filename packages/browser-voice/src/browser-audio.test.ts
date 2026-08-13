import assert from "node:assert/strict";
import { test } from "node:test";
import type { AudioSegment } from "@speakeasy/convo-engine";
import { BrowserAudioSink } from "./browser-audio.ts";
import type { BrowserHostEvent } from "./protocol.ts";

const segment: AudioSegment = {
  index: 0,
  sentence: "hello",
  samples: new Float32Array([0.1, 0.2]),
  sampleRate: 16_000,
  readyAtMs: 0,
  synthMs: 0,
  audioDurationMs: 1,
};

test("browser sink waits for the browser playback drain acknowledgement", async () => {
  const events: BrowserHostEvent[] = [];
  const packets: Uint8Array[] = [];
  const sink = new BrowserAudioSink({
    playbackId: 3,
    sampleRate: 16_000,
    sendEvent: (event) => events.push(event),
    sendAudio: (packet) => packets.push(packet),
  });
  sink.open();
  sink.write(segment);
  let settled = false;
  const ending = sink.end().then(() => {
    settled = true;
  });
  await Promise.resolve();
  assert.equal(settled, false);
  sink.drained();
  await ending;
  assert.equal(settled, true);
  assert.equal(packets.length, 1);
  assert.deepEqual(
    events.filter((event) => event.type === "playback").map((event) => event.action),
    ["start", "end"],
  );
});

test("interrupt resolves playback immediately and clears the client queue", async () => {
  const events: BrowserHostEvent[] = [];
  const sink = new BrowserAudioSink({
    playbackId: 9,
    sampleRate: 16_000,
    sendEvent: (event) => events.push(event),
    sendAudio: () => {},
  });
  sink.open();
  const interrupted = sink.interrupt();
  sink.drained(375);
  assert.equal(await interrupted, 375);
  await sink.end();
  assert.deepEqual(
    events.filter((event) => event.type === "playback").map((event) => event.action),
    ["start", "clear"],
  );
});
