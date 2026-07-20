import assert from "node:assert/strict";
import { test } from "node:test";
import type { AudioSegment, TTSSession, TextToSpeech } from "@speakeasy/speech-io";
import type { ChatModel } from "@speakeasy/llm";
import { CascadeResponder } from "./cascade.ts";

const now = () => performance.now();

class StubLLM implements ChatModel {
  async *stream(): AsyncGenerator<string> {
    yield "One. ";
    yield "Two.";
  }
}

/** Yields one segment per pulled token, mimicking the sentence pipeline shape. */
class StubTTS implements TextToSpeech {
  closed = 0;
  async open(): Promise<TTSSession> {
    const self = this;
    return {
      async *speak(text): AsyncGenerator<AudioSegment> {
        let index = 0;
        for await (const sentence of text as AsyncIterable<string>) {
          yield {
            index,
            sentence,
            samples: new Float32Array(160),
            sampleRate: 16000,
            readyAtMs: 0,
            synthMs: 1,
            audioDurationMs: 10,
          };
          index += 1;
        }
      },
      async close(): Promise<void> {
        self.closed += 1;
      },
    };
  }
}

test("cascade interleaves token and audio events, tokens first", async () => {
  const session = await new CascadeResponder({
    llm: new StubLLM(),
    tts: new StubTTS(),
    now,
  }).open();

  const kinds: string[] = [];
  let reply = "";
  let lastTokenAt = 0;
  for await (const event of session.respond([{ role: "user", content: "go" }])) {
    kinds.push(event.type);
    if (event.type === "token") {
      reply += event.text;
      assert.ok(event.at >= lastTokenAt, "token timestamps are monotonic");
      lastTokenAt = event.at;
    }
  }

  // Each pulled token is followed by its segment; every token precedes audio.
  assert.deepEqual(kinds, ["token", "audio", "token", "audio"]);
  assert.equal(reply, "One. Two.");
});

test("cascade close() closes the underlying TTS session", async () => {
  const tts = new StubTTS();
  const session = await new CascadeResponder({ llm: new StubLLM(), tts, now }).open();
  await session.close();
  assert.equal(tts.closed, 1);
});
