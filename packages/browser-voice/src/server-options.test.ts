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
