import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildTurnMetrics,
  formatSessionSummary,
  formatTurnLine,
  type TurnMetrics,
} from "./metrics.ts";

test("buildTurnMetrics anchors every interval at the endpoint", () => {
  const metrics = buildTurnMetrics(
    1,
    "hello there",
    { endpointAt: 1000, finalAt: 1030, firstTokenAt: 1120, firstAudioAt: 1280 },
    7,
    2400,
  );
  assert.equal(metrics.endpointToFinalMs, 30);
  assert.equal(metrics.endpointToFirstTokenMs, 120);
  assert.equal(metrics.endpointToFirstAudioMs, 280);
  assert.equal(metrics.tokenCount, 7);
  assert.equal(metrics.spokenMs, 2400);
});

test("formatTurnLine reports the three latencies and totals", () => {
  const line = formatTurnLine({
    turn: 2,
    transcript: "x",
    endpointToFinalMs: 30,
    endpointToFirstTokenMs: 120,
    endpointToFirstAudioMs: 280,
    tokenCount: 7,
    spokenMs: 2400,
  });
  assert.match(line, /turn 2/);
  assert.match(line, /stt-final=30\.0ms/);
  assert.match(line, /first-token=120\.0ms/);
  assert.match(line, /first-audio=280\.0ms/);
  assert.match(line, /tokens=7/);
});

test("formatSessionSummary medians the recorded turns", () => {
  const turns: TurnMetrics[] = [
    metric(10, 100, 200),
    metric(20, 140, 300),
    metric(30, 120, 250),
  ];
  const summary = formatSessionSummary(turns);
  assert.match(summary, /3 turn\(s\)/);
  assert.match(summary, /endpoint->stt-final\s+20\.0ms/);
  assert.match(summary, /endpoint->first-token\s+120\.0ms/);
  assert.match(summary, /endpoint->first-audio\s+250\.0ms/);
});

test("formatSessionSummary handles an empty session", () => {
  assert.match(formatSessionSummary([]), /no completed turns/);
});

function metric(final: number, token: number, audio: number): TurnMetrics {
  return {
    turn: 1,
    transcript: "x",
    endpointToFinalMs: final,
    endpointToFirstTokenMs: token,
    endpointToFirstAudioMs: audio,
    tokenCount: 1,
    spokenMs: 0,
  };
}
