import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const context = vm.createContext({});
vm.runInContext(readFileSync(new URL("../public/speech-observations.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);

test("durations preserve measured zero and reject missing, negative and nonfinite values", () => {
  assert.equal(context.formatMs(0), "0 ms");
  assert.equal(context.formatMs(1.6), "2 ms");
  for (const invalid of [null, undefined, -1, Infinity, NaN, "0"]) assert.equal(context.formatMs(invalid), "Unavailable");
});

test("observations retain measured clock/boundary and provider reported metric/boundary", () => {
  assert.equal(context.formatObservation({ status: "measured", ms: 0, clock: "browser", boundary: "submit to render" }), "0 ms (measured; browser; submit to render)");
  assert.equal(context.formatObservation({ status: "provider-reported", ms: 12.5, metric: "compute", boundary: "provider header" }), "13 ms (provider reported; compute; provider header)");
  assert.equal(context.formatObservation({ status: "unavailable", reason: "not exposed" }), "Unavailable: not exposed");
  for (const value of [null, {}, { status: "other", ms: 0, boundary: "unknown" }, { status: "measured", ms: -1, clock: "server", boundary: "bad" }]) assert.equal(context.formatObservation(value), "Unavailable");
});

test("transcript comparison is exact and distinguishes empty from unavailable", () => {
  assert.equal(context.transcriptStatus("", ""), "same");
  assert.equal(context.transcriptStatus("Say this.", ""), "different");
  for (const returned of [null, undefined]) assert.equal(context.transcriptStatus("", returned), "unavailable");
  for (const returned of ["say this.", "Say this", " Say this."]) assert.equal(context.transcriptStatus("Say this.", returned), "different");
});
