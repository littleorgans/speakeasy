import assert from "node:assert/strict";
import { test } from "node:test";
import { StubEngine } from "../engines/stub.ts";
import { nextCommit } from "./commit.ts";

for (const kind of ["final", "empty"] as const) {
  test(`a ${kind} commit settles the waiter and releases both listeners`, async () => {
    const session = await new StubEngine().open();
    const committed = nextCommit(session, 1_000);
    assert.equal(session.listenerCount("final"), 1);
    assert.equal(session.listenerCount("empty"), 1);
    session.emit(kind, { text: " hello ", audioMs: 500 });
    assert.equal((await committed)?.text, kind === "final" ? "hello" : "");
    assert.equal(session.listenerCount("final"), 0);
    assert.equal(session.listenerCount("empty"), 0);
  });
}

test("an asynchronous commit is observed", async () => {
  const session = await new StubEngine().open();
  const committed = nextCommit(session, 1_000);
  queueMicrotask(() => session.emit("final", { text: "hello" }));
  assert.equal((await committed)?.text, "hello");
});

test("a timeout releases both commit listeners", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const session = await new StubEngine().open();
  const committed = nextCommit(session, 1_000);
  t.mock.timers.tick(1_000);
  assert.equal(await committed, undefined);
  assert.equal(session.listenerCount("final"), 0);
  assert.equal(session.listenerCount("empty"), 0);
});
