import assert from "node:assert/strict";
import { test } from "node:test";
import type { WavAudio } from "@speakeasy/speech-io";
import { ConversationLoop } from "../loop.ts";
import type { ConversationEvent } from "../events.ts";
import { CascadeResponder } from "../responder/cascade.ts";
import { FakeLLM, FakeSTT, FakeTTS } from "../test-support.ts";
import { createNullSink } from "./null-sink.ts";
import { ScriptedWavSource } from "./sources.ts";

test("scripted source continues after an empty final transcript", {
  timeout: 500,
}, async () => {
  const source = new ScriptedWavSource(
    [utterance(0.5), utterance(0.7), utterance(0.9)],
    {
      frameMs: 1,
      silenceTailMs: 1,
    },
  );
  const stt = new FakeSTT();
  const transcripts = ["first", "", "third"];
  const metrics: Extract<ConversationEvent, { type: "metrics" }>[] = [];
  let heardSpeech = false;

  const loop = new ConversationLoop(
    {
      stt,
      responder: new CascadeResponder({
        llm: new FakeLLM(["Reply."]),
        tts: new FakeTTS(),
      }),
      mic: source,
      createSink: createNullSink,
    },
    {
      maxTurns: 2,
      onEvent: (event) => {
        source.onEvent(event);
        if (event.type === "metrics") {
          metrics.push(event);
        }
      },
    },
  );

  stt.session.onPushAudio = (frame) => {
    if (frame.some((sample) => sample !== 0)) {
      assert.equal(loop.state, "listening");
      heardSpeech = true;
      return;
    }
    if (heardSpeech) {
      heardSpeech = false;
      const transcript = transcripts.shift();
      if (transcript !== undefined) {
        stt.session.say(transcript);
      }
    }
  };

  await loop.start();
  await loop.done;
  await source.done;

  assert.deepEqual(
    metrics.map((event) => event.metrics.transcript),
    ["first", "third"],
  );
  assert.deepEqual(transcripts, []);
});

test("scripted source drops the rest of an utterance after an early endpoint", async () => {
  const source = new ScriptedWavSource(
    [utteranceFrames(1, 2), utteranceFrames(3)],
    { frameMs: 0.0001, silenceTailMs: 0 },
  );
  const amplitudes: number[] = [];

  source.onEvent({ type: "state", state: "listening" });
  source.start({
    onFrame: (frame) => {
      amplitudes.push(frame[0] ?? 0);
      if (amplitudes.length === 1) {
        queueMicrotask(() => {
          source.onEvent({ type: "state", state: "thinking" });
          setTimeout(() => {
            source.onEvent({ type: "state", state: "listening" });
          }, 10);
        });
      }
    },
    onError: assert.fail,
  });

  await source.done;
  assert.deepEqual(amplitudes, [1, 3]);
});

function utterance(amplitude: number): WavAudio {
  const samples = new Float32Array([amplitude]);
  return {
    sampleRate: 1_000,
    channels: 1,
    bitsPerSample: 32,
    durationMs: 1,
    samples,
    frames: [samples],
  };
}

function utteranceFrames(...amplitudes: number[]): WavAudio {
  const frames = amplitudes.map((amplitude) => new Float32Array([amplitude]));
  return {
    sampleRate: 1_000,
    channels: 1,
    bitsPerSample: 32,
    durationMs: amplitudes.length,
    samples: new Float32Array(amplitudes),
    frames,
  };
}
