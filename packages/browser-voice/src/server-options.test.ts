import assert from "node:assert/strict";
import { test } from "node:test";
import { readBrowserVoiceOptions } from "./server-options.ts";

test("server options keep the model and use the cascade TTS voice", () => {
  assert.deepEqual(
    readBrowserVoiceOptions([], {
      SPEAKEASY_MODEL: "operator-model",
      SPEAKEASY_TTS: "sherpa",
      SPEAKEASY_TTS_MODEL: "kokoro-v0.19",
      SPEAKEASY_TTS_VOICE: "7",
      SPEAKEASY_VOICE: "marin",
    }),
    {
      port: 4317,
      runtimeConfig: {
        llmModel: "operator-model",
        ttsEngine: "sherpa",
        ttsModel: "kokoro-v0.19",
        voice: "7",
      },
    },
  );
});

test("prompt reader flags come as a pair", () => {
  assert.deepEqual(
    readBrowserVoiceOptions(["--prompts", "set-a.json", "--out", "/tmp/takes"], {})
      .promptReader,
    { promptsPath: "set-a.json", outputDir: "/tmp/takes" },
  );
  assert.equal(readBrowserVoiceOptions([], {}).promptReader, undefined);
  assert.throws(
    () => readBrowserVoiceOptions(["--prompts", "set-a.json"], {}),
    /go together/,
  );
  assert.throws(
    () => readBrowserVoiceOptions(["--out", "--prompts", "x"], {}),
    /--out needs a value/,
  );
});

test("hotwords flag selects an external list and requires its path", () => {
  assert.equal(readBrowserVoiceOptions(["--hotwords", "../speech/words.txt"], {}).hotwordsPath,
    "../speech/words.txt");
  assert.equal(readBrowserVoiceOptions([], {}).hotwordsPath, undefined);
  for (const argv of [["--hotwords"], ["--hotwords", "--port", "0"]]) {
    assert.throws(() => readBrowserVoiceOptions(argv, {}), /--hotwords needs a value/);
  }
});
