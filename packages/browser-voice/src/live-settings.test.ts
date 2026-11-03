import assert from "node:assert/strict";
import { test } from "node:test";
import { defaultLiveSetup, LIVE_STREAMS, liveStream, resolveLiveSettings } from "./live-settings.ts";
import { SETUPS } from "./setup-registry.ts";

test("the live streams are exactly the setups whose engine streams, each defaulting to today's values", () => {
  assert.deepEqual(LIVE_STREAMS.map((stream) => stream.setupId).sort(),
    SETUPS.filter((setup) => setup.card.transcript === "partials" && !setup.baseSetup).map((setup) => setup.id).sort());
  for (const stream of LIVE_STREAMS) {
    for (const setting of stream.settings) {
      assert.ok(setting.options.some((option) => option.value === setting.default), `${stream.setupId} ${setting.id}`);
      assert.ok(setting.note.trim() && setting.label.trim());
    }
  }
  assert.deepEqual(resolveLiveSettings(liveStream("mlx-parakeet/ultra-greedy")!), { updateMs: 800 },
    "800 ms chunks, as before");
  assert.deepEqual(resolveLiveSettings(liveStream("hex/unified")!), { lookaheadMs: 2_080 }, "Hex's own 2.08 s lookahead");
  assert.equal(liveStream("hex/unified")!.settings[0]!.options[0]!.phrase, "2.08 s lookahead, written words never change");
});

test("each setup's default live model is today's pairing", () => {
  assert.equal(defaultLiveSetup("hex/unified"), "hex/unified");
  assert.equal(defaultLiveSetup("mlx-parakeet/v3-greedy"), "mlx-parakeet/v3-greedy");
  assert.equal(defaultLiveSetup("mlx-cleanup/qwen-targeted"), "mlx-parakeet/ultra-greedy", "cleanup shows its base's words");
  for (const id of ["qwen/context", "sweep/score-2", "tm/today"] as const) assert.equal(defaultLiveSetup(id), null, id);
});

test("live settings refuse unknown names, values that were not offered and non objects", () => {
  const parakeet = liveStream("mlx-parakeet/v3-greedy")!;
  assert.deepEqual(resolveLiveSettings(parakeet, { updateMs: 1_600 }), { updateMs: 1_600 });
  assert.throws(() => resolveLiveSettings(parakeet, { windowMs: 2_560 }), /unknown live setting/);
  assert.throws(() => resolveLiveSettings(parakeet, { updateMs: 320 }), /unsupported updateMs/);
  assert.throws(() => resolveLiveSettings(parakeet, { updateMs: "800" }), /unsupported updateMs/);
  assert.throws(() => resolveLiveSettings(parakeet, { lookaheadMs: 480 }), /unknown live setting/);
  assert.throws(() => resolveLiveSettings(parakeet, [800]), /object/);
  assert.equal(liveStream("qwen/context"), undefined);
});
