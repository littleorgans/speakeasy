import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSpeechOutputCommand } from "./speech-output-protocol.ts";

const registry = [{ kind: "local" as const, model: "bundle", voices: ["voice", 0], speed: { min: 0.5, max: 2 } }];
const selection = { schema: 1, provider: { kind: "local" }, model: "bundle", voice: "voice", speed: 1 };
const command = { type: "speak-sample", requestId: "sample_1", catalogueRevision: "revision", selection, text: "  One. Two.  " };
const parse = (value: unknown) => parseSpeechOutputCommand(JSON.stringify(value), registry);

test("trial protocol freezes exact validated config and preserves typed text", () => {
  assert.deepEqual(parse(command), command);
  assert.deepEqual(parse({ ...command, selection: { ...selection, voice: 0 } }), {
    ...command, selection: { ...selection, voice: 0 },
  });
  assert.deepEqual(parse({ type: "cancel-sample", requestId: "sample_1" }), { type: "cancel-sample", requestId: "sample_1" });
  assert.deepEqual(parse({ type: "playback-drained", playbackId: 1, audioEndMs: 23 }), {
    type: "playback-drained", playbackId: 1, audioEndMs: 23,
  });
});

test("trial protocol rejects malformed config, text, ids and client authority", () => {
  const variants = [
    { ...command, selection: { ...selection, model: "other" } },
    { ...command, selection: { ...selection, voice: "other" } },
    { ...command, selection: { ...selection, voice: "0" } },
    { ...command, selection: { ...selection, speed: 0 } },
    { ...command, selection: { ...selection, speed: 2.01 } },
    { ...command, selection: { ...selection, endpoint: "forbidden" } },
    { ...command, selection: { ...selection, provider: { kind: "groq" } } },
    { ...command, text: " \n " }, { ...command, text: "x".repeat(2001) },
    { ...command, requestId: "x".repeat(97) }, { ...command, requestId: "../take" },
    { ...command, sessionId: "client" }, { ...command, source: { kind: "saved-take" } },
    { ...command, grant: { approved: true } }, { ...command, approved: true },
    { ...command, selectionRevision: -1 }, { ...command, catalogueRevision: "" },
    { type: "playback-drained", playbackId: -1 },
    { type: "playback-drained", playbackId: 2 ** 32 },
    { type: "playback-drained", playbackId: 1, audioEndMs: -1 },
    { type: "cancel-sample", requestId: "sample_1", grantId: "forged" },
  ];
  for (const value of variants) assert.throws(() => parse(value));
  for (const raw of ["{", "null", "[]", "x".repeat(20_001)]) {
    assert.throws(() => parseSpeechOutputCommand(raw, registry));
  }
});

test("selection is explicit and revisions cannot be forged into a config", () => {
  assert.deepEqual(parse({ type: "select-output", requestId: "choose", catalogueRevision: "revision", selection }), {
    type: "select-output", requestId: "choose", catalogueRevision: "revision", selection,
  });
  assert.throws(() => parse({ type: "select-output", requestId: "choose", catalogueRevision: "revision", selection, selected: true }));
});
