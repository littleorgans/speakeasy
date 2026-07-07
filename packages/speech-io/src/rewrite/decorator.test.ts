import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { withRewrite } from "./decorator.ts";

class FakeSession extends EventEmitter implements STTSession {
  calls: string[] = [];
  pushAudio(): void {
    this.calls.push("pushAudio");
  }
  flush(): void {
    this.calls.push("flush");
  }
  reset(): void {
    this.calls.push("reset");
  }
  async end(): Promise<void> {
    this.calls.push("end");
  }
}

class FakeEngine implements VoiceToText {
  last?: FakeSession;
  async open(_config?: STTConfig): Promise<STTSession> {
    this.last = new FakeSession();
    return this.last;
  }
}

const config = {
  rules: [{ from: "crown", to: "chrome" }],
  numbers: "digits" as const,
};

test("rewrites final events through the pipeline", async () => {
  const engine = new FakeEngine();
  const session = await withRewrite(engine, config).open();
  const finals: string[] = [];
  session.on("final", (e: { text: string }) => finals.push(e.text));

  engine.last!.emit("final", { text: "open crown ten" });
  assert.deepEqual(finals, ["open chrome 10"]);
});

test("leaves partial events untouched (no flicker)", async () => {
  const engine = new FakeEngine();
  const session = await withRewrite(engine, config).open();
  const partials: string[] = [];
  session.on("partial", (e: { text: string }) => partials.push(e.text));

  engine.last!.emit("partial", { text: "open crown" });
  assert.deepEqual(partials, ["open crown"]);
});

test("passes endpoint and error events through", async () => {
  const engine = new FakeEngine();
  const session = await withRewrite(engine, config).open();
  let endpoints = 0;
  let errored: unknown;
  session.on("endpoint", () => (endpoints += 1));
  session.on("error", (e: { err: unknown }) => (errored = e.err));

  engine.last!.emit("endpoint", {});
  engine.last!.emit("error", { err: "boom" });
  assert.equal(endpoints, 1);
  assert.equal(errored, "boom");
});

test("forwards session methods to the inner session", async () => {
  const engine = new FakeEngine();
  const session = await withRewrite(engine, config).open();
  session.pushAudio(new Float32Array(1));
  session.flush();
  session.reset();
  await session.end();
  assert.deepEqual(engine.last!.calls, ["pushAudio", "flush", "reset", "end"]);
});
