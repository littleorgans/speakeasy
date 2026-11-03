import assert from "node:assert/strict";
import { test } from "node:test";
import * as ort from "onnxruntime-node";
import { ORT_VERSION, sessionOptions } from "./ort-runtime.ts";

test("each session profile keeps its runtime's exact options", () => {
  assert.deepEqual(sessionOptions(4, "tuned"), {
    executionProviders: ["cpu"],
    intraOpNumThreads: 4,
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  assert.deepEqual(sessionOptions(1, "reference"), {
    executionProviders: ["cpu"],
    intraOpNumThreads: 1,
  });
});

test("the label version is the common version Granite labels read", () => {
  assert.equal(ORT_VERSION, ort.env.versions.common);
});
