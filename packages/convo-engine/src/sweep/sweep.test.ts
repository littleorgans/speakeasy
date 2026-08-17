import assert from "node:assert/strict";
import { test } from "node:test";
import type { WavAudio } from "@speakeasy/speech-io";
import { CascadeResponder } from "../responder/cascade.ts";
import { FakeLLM, FakeSTT, FakeTTS } from "../test-support.ts";
import { expandMatrix } from "./matrix.ts";
import { formatResultRow } from "./results.ts";
import { runSweep } from "./runner.ts";
import type { SweepMatrixRow, SweepTurn } from "./types.ts";

test("expandMatrix covers realtime and every cascade combination", () => {
  assert.deepEqual(
    expandMatrix().map((row) => [row.id, row.config]),
    [
      ["realtime", { responder: "realtime" }],
      [
        "cerebras-sherpa",
        { responder: "cascade", llmProvider: "cerebras", ttsEngine: "sherpa" },
      ],
      [
        "cerebras-cartesia",
        { responder: "cascade", llmProvider: "cerebras", ttsEngine: "cartesia" },
      ],
      [
        "mercury-instant-sherpa",
        {
          responder: "cascade",
          llmProvider: "mercury",
          llmReasoningEffort: "instant",
          ttsEngine: "sherpa",
        },
      ],
      [
        "mercury-instant-cartesia",
        {
          responder: "cascade",
          llmProvider: "mercury",
          llmReasoningEffort: "instant",
          ttsEngine: "cartesia",
        },
      ],
      [
        "mercury-low-sherpa",
        {
          responder: "cascade",
          llmProvider: "mercury",
          llmReasoningEffort: "low",
          ttsEngine: "sherpa",
        },
      ],
      [
        "mercury-low-cartesia",
        {
          responder: "cascade",
          llmProvider: "mercury",
          llmReasoningEffort: "low",
          ttsEngine: "cartesia",
        },
      ],
    ],
  );
});

test("runSweep records a missing key and continues to the next row", async () => {
  const rows: SweepMatrixRow[] = [
    {
      id: "cerebras-sherpa",
      config: { responder: "cascade", llmProvider: "cerebras" },
    },
    {
      id: "mercury-instant-sherpa",
      config: { responder: "cascade", llmProvider: "mercury" },
    },
  ];
  const stt = new FakeSTT();
  stt.session.onFlush = () => stt.session.say("test utterance");

  const results = await runSweep(
    {
      rows,
      utterances: [{ id: "test", audio: utterance() }],
      runs: 1,
      turnGapMs: 0,
    },
    {
      env: {},
      sourceOptions: { frameMs: 1, silenceTailMs: 1 },
      createRuntime: async (config) => {
        if (config.llmProvider === "cerebras") {
          throw new Error("CEREBRAS_API_KEY is required");
        }
        return {
          stt,
          responder: new CascadeResponder({
            llm: new FakeLLM(["Reply."]),
            tts: new FakeTTS(),
          }),
          label: "fake runtime",
        };
      },
    },
  );

  assert.deepEqual(
    results.map((result) => result.status),
    ["skipped", "completed"],
  );
  assert.match(results[0].status === "skipped" ? results[0].reason : "", /CEREBRAS_API_KEY/);
  assert.equal(results[1].turns.length, 1);
  assert.equal(results[1].turns[0]?.metrics.transcript, "test utterance");
});

test("runSweep keeps completed turns when a later turn fails", async () => {
  const stt = new FakeSTT();
  const transcripts = ["first", "second", "third"];
  stt.session.onFlush = () => {
    const transcript = transcripts.shift();
    if (transcript) {
      stt.session.say(transcript);
    }
  };
  const llm = new FakeLLM(["Reply."]);
  llm.failOnCall = 2;

  const [result] = await runSweep(
    {
      rows: [{ id: "partial-row", config: { responder: "cascade" } }],
      utterances: ["first", "second", "third"].map((id) => ({
        id,
        audio: utterance(),
      })),
      runs: 1,
      turnGapMs: 0,
    },
    {
      env: {},
      sourceOptions: { frameMs: 1, silenceTailMs: 0 },
      createRuntime: async () => ({
        stt,
        responder: new CascadeResponder({ llm, tts: new FakeTTS() }),
        label: "fake runtime",
      }),
    },
  );

  assert.equal(result?.status, "partial");
  if (result?.status !== "partial") {
    assert.fail("expected a partial result");
  }
  assert.deepEqual(
    result?.turns.map((turn) => turn.utteranceId),
    ["first", "third"],
  );
  assert.equal(result.failedTurn, 2);
  assert.match(result.reason, /llm exploded/);
  const row = formatResultRow(result);
  assert.match(row, /^partial-row \| partial \|/);
  assert.match(row, /turn 2: error: llm exploded/);
  assert.doesNotMatch(row, /n\/a/);
});

test("formatResultRow separates the cold turn from the warm median", () => {
  const turns = [
    turn(1, 10, 100, 200),
    turn(2, 20, 120, 240),
    turn(3, 30, 130, 260),
  ];
  assert.equal(
    formatResultRow({
      status: "completed",
      id: "cerebras-sherpa",
      label: "fake runtime",
      turns,
      summary: "summary",
    }),
    "cerebras-sherpa | completed | 10.0ms | 25.0ms | 100.0ms | 125.0ms | 200.0ms | 250.0ms | 3 |",
  );
});

function utterance(): WavAudio {
  const samples = new Float32Array([0.5]);
  return {
    sampleRate: 1_000,
    channels: 1,
    bitsPerSample: 32,
    durationMs: 1,
    samples,
    frames: [samples],
  };
}

function turn(
  turnIndex: number,
  endpointToFinalMs: number,
  endpointToFirstTokenMs: number,
  endpointToFirstAudioMs: number,
): SweepTurn {
  return {
    configId: "cerebras-sherpa",
    utteranceId: `utt-${turnIndex}`,
    turnIndex,
    metrics: {
      turn: turnIndex,
      transcript: "test",
      endpointToFinalMs,
      endpointToFirstTokenMs,
      endpointToFirstAudioMs,
      tokenCount: 1,
      spokenMs: 100,
    },
  };
}
