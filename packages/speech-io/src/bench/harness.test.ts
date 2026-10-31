import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import { FLUSH_FINAL_TIMEOUT_MS } from "./config.ts";
import { runPttOnce } from "./harness.ts";

test("a silent push to talk run completes on empty with a measured latency", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let didFlush!: () => void;
  const flushed = new Promise<void>((resolve) => { didFlush = resolve; });
  class Session extends EventEmitter {
    pushAudio(): void {}
    reset(): void {}
    async end(): Promise<void> {}
    flush(): void {
      this.emit("empty", { audioMs: 0, decodeMs: 0 });
      didFlush();
    }
  }
  const session = new Session();
  const result = runPttOnce({
    run: 1,
    engine: { open: async () => session },
    audio: {
      frames: [], samples: new Float32Array(), sampleRate: 16_000,
      channels: 1, bitsPerSample: 16, durationMs: 0,
    },
    releaseFrame: 0,
    frameMs: 20,
    expected: "",
  });
  await flushed;
  t.mock.timers.tick(FLUSH_FINAL_TIMEOUT_MS);
  const observation = await result;
  assert.ok(Number.isFinite(observation.flushToFinalMs));
  assert.equal(observation.finalText, "");
  assert.equal(observation.textCorrect, true);
});
