import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { test } from "node:test";
import type { STTSession } from "../contract.ts";
import { flushCommit } from "./commit.ts";
import { StubEngine } from "./stub.ts";

function assertListenersReleased(session: STTSession): void {
  assert.equal(session.listenerCount("final"), 0);
  assert.equal(session.listenerCount("empty"), 0);
}

for (const kind of ["final", "empty"] as const) {
  test(`a ${kind} during flush() commits after the release stamp`, async () => {
    const session = await new StubEngine().open();
    let flushedAt = 0;
    session.flush = () => {
      assert.equal(session.listenerCount("final"), 1);
      assert.equal(session.listenerCount("empty"), 1);
      flushedAt = performance.now();
      session.emit(kind, { text: " hello ", audioMs: 500 });
    };
    const commit = await flushCommit(session, 1_000);
    assert.equal(commit.observation?.text, kind === "final" ? "hello" : "");
    assert.ok(commit.releaseAt <= flushedAt);
    assert.ok(flushedAt <= commit.observation!.at);
    assert.equal(
      commit.releaseToCommitMs,
      commit.observation!.at - commit.releaseAt,
    );
    assertListenersReleased(session);
  });
}

test("an asynchronous commit is observed", async () => {
  const session = await new StubEngine().open();
  session.flush = () => {
    queueMicrotask(() => session.emit("final", { text: "hello" }));
  };
  const commit = await flushCommit(session, 1_000);
  assert.equal(commit.observation?.text, "hello");
  assertListenersReleased(session);
});

test("a timeout is distinct from empty and releases both listeners", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const session = await new StubEngine().open();
  session.flush = () => {};
  const committed = flushCommit(session, 1_000);
  t.mock.timers.tick(1_000);
  const commit = await committed;
  assert.equal(commit.observation, undefined);
  assert.ok(Number.isNaN(commit.releaseToCommitMs));
  assertListenersReleased(session);
});

for (const emitsFirst of [false, true]) {
  test(`a throwing flush() rethrows and cleans up${emitsFirst ? " even after a final" : ""}`, async (t) => {
    const clearTimeoutSpy = t.mock.method(globalThis, "clearTimeout");
    const session = await new StubEngine().open();
    const failure = new Error("flush failed");
    session.flush = () => {
      if (emitsFirst) {
        session.emit("final", { text: "hello" });
      }
      throw failure;
    };
    await assert.rejects(flushCommit(session, 1_000), failure);
    assert.ok(clearTimeoutSpy.mock.callCount() >= 1);
    assertListenersReleased(session);
  });
}
