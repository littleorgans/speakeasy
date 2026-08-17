import assert from "node:assert/strict";
import { test } from "node:test";
import { createNullSink } from "./null-sink.ts";
import { fakeAudioSegment } from "../test-support.ts";

test("null sink reports accumulated audio and ends without waiting", async () => {
  const sink = createNullSink();
  sink.open();
  sink.write(fakeAudioSegment(0));
  sink.write(fakeAudioSegment(1));

  assert.equal(await sink.interrupt(), 200);
  let ended = false;
  const end = sink.end().then(() => {
    ended = true;
  });
  await Promise.resolve();
  assert.equal(ended, true);
  await end;
});
