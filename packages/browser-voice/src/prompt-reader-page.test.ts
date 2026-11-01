import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as preRoll from "../public/pre-roll.js";
import { pageHarness, okReply, mic, waitForRequests, readyTake, comparisonCatalog, enableComparison, streamReply, rowEvent, acceptStream, type PagePort, type FetchInit, type FetchReply } from "./prompt-reader-page-harness.ts";
import { feed, loadCapture } from "./capture-worklet-harness.ts";

test("a late mic acquisition cannot attach to a later take", async () => {
  const { reader, pending, connected } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  reader.cancelTake("cancelled");
  const retaking = reader.press();
  await waitForRequests(pending, 2);
  const stale = mic();
  const current = mic();
  pending[0]!(stale);
  await opening;
  pending[1]!(current);
  await retaking;
  assert.equal(stale.track.stopped, true);
  assert.deepEqual(connected, [current]);
  reader.cancelTake("done");
  assert.equal(current.track.stopped, true);
});

test("review keeps worklet frame values and shared mic constraints", async () => {
  const { reader, pending, constraints, expectedConstraints } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  pending[0]!(mic());
  await opening;
  assert.equal(JSON.stringify(constraints), JSON.stringify([{ audio: expectedConstraints }]));
  const frames = [new Float32Array(3_200).fill(0.123456), new Float32Array(3_200).fill(-0.654321)];
  for (const samples of frames) reader.onFrame({ kind: "frame", samples, level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(reader.take.samples.length, 6_400);
  assert.equal(reader.take.samples[0], frames[0]![0]);
  assert.equal(reader.take.samples[3_200], frames[1]![0]);
});

test("each take drains its worklet tail, so the next take holds none of it", async () => {
  // Page and real worklet, connected through asynchronous ports at 48 kHz.
  let page: PagePort | undefined;
  const { context, Processor } = loadCapture(48_000, (message) =>
    queueMicrotask(() => page?.onmessage?.({ data: message })),
  );
  const processor = new Processor();
  const { reader, pending, posted } = pageHarness({
    worklet: (_port, data) => queueMicrotask(() => processor.port.onmessage?.({ data })),
  });
  const timers: Array<() => void> = [];
  reader.gate.schedule = (callback: () => void) => timers.push(callback);
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const takes: Float32Array[] = [];
  let block = 0;
  for (const value of [1, 0]) {
    const opening = reader.press();
    await waitForRequests(pending, 1);
    if (!takes.length) pending[0]!(mic());
    await opening;
    page = reader.capture.port;
    feed(context, processor, block, (block += 55), value);
    await settle();
    reader.gate.release();
    feed(context, processor, block, (block += 75), value);
    await settle();
    timers.shift()!();
    // Saved audio must stay blocked until the new take's flush reply arrives.
    reader.prompts[0].takes = ["previous.wav"];
    reader.render();
    assert.equal(reader.elements.play.disabled, true);
    await reader.play();
    assert.equal(reader.playback, null);
    await settle();
    takes.push(reader.take.samples);
  }
  // 130 blocks of 128 source samples: 17 frames of 320 plus a 107 sample tail.
  assert.ok(takes[1]!.every((sample) => sample === 0), "second take holds no earlier samples");
  assert.equal(takes[1]!.length, 17 * 320 + 107);
  assert.equal(takes[0]!.length, 17 * 320 + 107, "first take keeps its drained tail");
  assert.ok(takes[0]!.every((sample) => sample === 1));
  assert.equal(posted.filter((data) => (data as { kind?: string }).kind === "flush").length, 2);
});

test("a stale mic denial cannot cancel the current take", async () => {
  const { reader, pending, denied } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  reader.cancelTake("cancelled");
  const retaking = reader.press();
  await waitForRequests(pending, 2);
  const current = mic();
  pending[1]!(current);
  await retaking;
  denied[0]!(new Error("late denial"));
  await opening;
  assert.equal(reader.gate.capturing, true);
  assert.equal(current.track.stopped, false);
  reader.cancelTake("done");
});

test("play replays the unsaved take without opening the mic or changing it", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  pending[0]!(mic());
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(0.3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  const take = reader.take;
  const before = Array.from(take.samples);
  await reader.play();
  assert.equal(pending.length, 1, "playback must not request the mic");
  assert.equal(reader.gate.capturing, false);
  assert.equal(reader.take, take);
  assert.deepEqual(Array.from(take.samples), before);
  assert.deepEqual(Array.from(played[0]!.samples), before);
  assert.deepEqual(fetched, ["/prompt-reader/transcriptions"], "only the display transcript is fetched");
  assert.equal(reader.elements.play.disabled, false);
});

test("play on a revisited prompt fetches the latest saved take; T steps to earlier ones", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: true, takes: ["utt-p001-a.wav", "utt-p001-b.wav"] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, false);
  assert.equal(reader.elements.playLabel.textContent, "Play take 2 of 2");
  assert.equal(reader.elements.earlier.hidden, false);
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav"]);
  assert.equal(played.length, 1);
  reader.stepTake();
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav", "/prompt-reader/takes/utt-p001-a.wav"]);
  assert.equal(pending.length, 0, "playback must not request the mic");
  assert.equal(reader.take, null);
});

test("play is disabled on a prompt with no take", () => {
  const { reader } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, true);
  assert.equal(reader.elements.earlier.hidden, true);
});

const settle = () => new Promise((resolve) => setImmediate(resolve));
const heardWords = (reader: { elements: { heardText: { children?: Array<{ textContent: string; className: string }> } } }) =>
  (reader.elements.heardText.children ?? []).map(({ textContent, className }) => [textContent, className]);

test("a ready take shows its transcript under the prompt with differing words marked", async () => {
  const posted: Array<{ url: string; init?: FetchInit }> = [];
  const harness = pageHarness({
    respond: async (url, init) => {
      posted.push({ url, init });
      return okReply({ text: "Read, that THIS!", decodeMs: 155.3, model: "Canary 180M Flash" });
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  const take = await readyTake(harness);
  assert.equal(posted[0]!.url, "/prompt-reader/transcriptions");
  assert.equal(posted[0]!.init!.body, take.samples.buffer, "the take's own samples are sent");
  await settle();
  const { reader } = harness;
  assert.equal(reader.elements.heard.hidden, false);
  assert.deepEqual(heardWords(reader), [["Read,", ""], [" that", "differs"], [" THIS!", ""]]);
  assert.equal(reader.elements.heardNote.textContent, "Canary 180M Flash · 155 ms");
});

test("save works while the transcript is pending and never carries it", async () => {
  const saves: Array<{ url: string; init?: FetchInit }> = [];
  const harness = pageHarness({
    respond: (url, init) => {
      if (url === "/prompt-reader/transcriptions") return new Promise(() => {});
      saves.push({ url, init });
      return Promise.resolve(okReply({ audio: "utt-p001-a.wav" }));
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  const take = await readyTake(harness);
  const { reader } = harness;
  assert.equal(reader.elements.heardNote.textContent, "Transcribing");
  await reader.save();
  assert.equal(saves.length, 1);
  assert.match(saves[0]!.url, /^\/prompt-reader\/recordings\?promptId=p001&device=test\+mic&preRollMs=0$/);
  assert.equal(saves[0]!.init!.body, take.samples.buffer);
  assert.equal(reader.take, null);
  assert.equal(reader.index, 1);
  assert.equal(reader.elements.heard.hidden, true, "the transcript leaves with its take");
});

test("a failed transcript shows a quiet note and save still works", async () => {
  let saved = 0;
  const harness = pageHarness({
    respond: async (url) => {
      if (url === "/prompt-reader/transcriptions") return { ok: false, status: 503, json: async () => ({ error: "child exited" }) };
      saved += 1;
      return okReply({ audio: "utt-p001-a.wav" });
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  await readyTake(harness);
  await settle();
  const { reader } = harness;
  assert.equal(reader.elements.heard.hidden, false);
  assert.equal(reader.elements.heardNote.textContent, "Transcript unavailable");
  assert.deepEqual(heardWords(reader), []);
  await reader.save();
  assert.equal(saved, 1);
  assert.equal(reader.index, 1);
});

test("a late transcript never lands on a newer take", async () => {
  const replies: Array<(reply: FetchReply) => void> = [];
  const harness = pageHarness({
    respond: () => new Promise((resolve) => replies.push(resolve)),
  });
  const first = await readyTake(harness);
  const second = await readyTake(harness);
  replies[0]!(okReply({ text: "stale words", decodeMs: 1, model: "M" }));
  await settle();
  assert.equal(first.transcript.state, "done");
  assert.equal(second.transcript.state, "pending");
  assert.equal(harness.reader.elements.heardNote.textContent, "Transcribing");
});

test("each take shows its own applied list count and model after later file changes", async () => {
  let count = 2;
  const harness = pageHarness({ respond: async () => okReply({
    text: "read this", decodeMs: 3, model: "en-parakeet-ultra-int8",
    hotwords: { path: "/owner/hotwords.txt", count, revision: String(count) },
  }) });
  const first = await readyTake(harness);
  await settle();
  assert.match(harness.reader.elements.heardHotwords.textContent, /2 entries applied/);
  assert.match(harness.reader.elements.heardNote.textContent, /en-parakeet-ultra-int8/);
  count = 0;
  const second = await readyTake(harness);
  await settle();
  assert.match(harness.reader.elements.heardHotwords.textContent, /No list.*greedy/);
  assert.equal(first.transcript.hotwords.count, 2);
  assert.equal(second.transcript.hotwords.count, 0);
});

test("a rejected list displays exact reasons and keeps the original take saveable", async () => {
  const reason = 'hotword "a☃": unencodable UTF8 coverage at byte 4\nhotword "a :2": reserved native score syntax';
  let saved: ArrayBuffer | string | undefined;
  const harness = pageHarness({ respond: async (url, init) => {
    if (url === "/prompt-reader/transcriptions") return {
      ok: false, status: 503, json: async () => ({ error: reason, model: "en-parakeet-ultra-int8",
        hotwords: { path: "/owner/hotwords.txt", count: 3, revision: "bad" } }),
    };
    saved = init?.body;
    return okReply({ audio: "utt-p001-a.wav" });
  } });
  const take = await readyTake(harness);
  await settle();
  assert.equal(harness.reader.elements.heardError.textContent, reason);
  assert.match(harness.reader.elements.heardHotwords.textContent, /3 entries.*failed/);
  assert.match(harness.reader.elements.heardNote.textContent, /en-parakeet-ultra-int8/);
  assert.equal(harness.reader.take, take);
  await harness.reader.play();
  assert.equal(harness.played[0]?.samples.length, take.samples.length);
  await harness.reader.save();
  assert.equal(saved, take.samples.buffer);
});

const idleFrames = (count: number, value: number) => Array.from({ length: count }, () => new Float32Array(320).fill(value));

test("a take starts with the 300 ms before its press", async () => {
  const saves: string[] = [];
  const harness = pageHarness({
    respond: async (url) => {
      if (url.startsWith("/prompt-reader/recordings")) saves.push(url);
      return okReply({ audio: "utt-a.wav" });
    },
  });
  const { reader, pending } = harness;
  reader.prompts = [
    { id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] },
    { id: "p002", stratum: "short", text: "read that", recorded: false, takes: [] },
  ];
  const opening = reader.press();
  await waitForRequests(pending, 1);
  const device = mic();
  pending[0]!(device);
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(1), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(device.track.stopped, false, "the mic stays open after commit");
  assert.equal(reader.take.preRollMs, 0, "the page's first take opens a cold mic");
  await reader.save();
  // Idle frames older than 300 ms fall out of the ring; the idle level stays hidden.
  for (const samples of [...idleFrames(10, 9), ...idleFrames(15, 2)]) reader.onFrame({ kind: "frame", samples, level: 0.4 });
  assert.equal(reader.elements.level.value, 0);
  const second = reader.press();
  await second;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(pending.length, 1, "one getUserMedia across both takes");
  const { samples } = reader.take;
  assert.equal(samples.length, 4_800 + 6_400);
  assert.ok(samples.subarray(0, 4_800).every((sample: number) => sample === 2), "pre roll is the idle ring");
  assert.ok(samples.subarray(4_800).every((sample: number) => sample === 3));
  assert.equal(reader.take.preRollMs, 300);
  await reader.save();
  assert.deepEqual(saves.map((url) => new URLSearchParams(url.split("?")[1]).get("preRollMs")), ["0", "300"]);
});

for (const [event, hide] of [["document:visibilitychange", true], ["window:pagehide", false]] as const) {
  test(`${event.split(":")[1]} stops the warm mic and the next press reopens it cold`, async () => {
    const harness = pageHarness();
    const { reader, pending, document, dispatch } = harness;
    await readyTake(harness);
    const device = reader.stream.getAudioTracks()[0];
    for (const samples of idleFrames(15, 2)) reader.onFrame({ kind: "frame", samples, level: 0.4 });
    document.hidden = hide;
    dispatch(event);
    assert.equal(device.stopped, true);
    document.hidden = false;
    const take = await readyTake(harness);
    assert.equal(pending.length, 2, "the next press opens the mic again");
    assert.equal(take.preRollMs, 0);
    assert.equal(take.samples.length, 6_400, "no idle frame from before the stop leads the take");
  });
}

test("a tap on a warm mic stays too short despite its pre roll", async () => {
  const harness = pageHarness();
  const { reader } = harness;
  await readyTake(harness);
  for (const samples of idleFrames(15, 2)) reader.onFrame({ kind: "frame", samples, level: 0.4 });
  await reader.press();
  reader.onFrame({ kind: "frame", samples: new Float32Array(3_200).fill(3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(reader.take, null);
  assert.match(reader.elements.status.textContent, /too short/);
});

test("comparison defaults to only v3 in exact manifest order and keeps unavailable identity visible", () => {
  const { reader } = pageHarness();
  enableComparison(reader);
  assert.equal(reader.comparison.checks.length, 5);
  assert.deepEqual(reader.comparison.checks.map((entry: any) => entry.setup.id), comparisonCatalog().setups.map((setup: any) => setup.id));
  assert.deepEqual(reader.comparison.checks.filter((entry: any) => entry.input.checked).map((entry: any) => entry.setup.id), ["mlx-parakeet/v3-greedy"]);
  assert.match(reader.comparison.checks[0].status.textContent, /Needs setup: test runtime unavailable/);
  assert.equal(reader.comparison.checks[0].tags.textContent, "Recommended");
  assert.equal(reader.comparison.checks[2].tags.textContent, "Uses fixed vocabulary");
  assert.match(reader.comparison.checks[3].fact.textContent, /Sherpa Ultra on CPU/);
  assert.equal(reader.comparison.checks.some((entry: any) => /Streams|Accuracy/.test(entry.tags.textContent)), false);
});

test("one canonical buffer is submitted once, rows arrive progressively in selection order, save and play remain independent", async () => {
  const stream = streamReply();
  const posts: Array<{ url: string; init?: FetchInit }> = [];
  const h = pageHarness({ respond: async (url, init) => {
    posts.push({ url, init });
    return url.startsWith("/prompt-reader/comparisons") ? stream.reply : okReply({ audio: "utt-p001-a.wav" });
  } });
  const ids = ["mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy"];
  enableComparison(h.reader, ids);
  const take = await readyTake(h);
  const request = posts[0]!;
  assert.equal(posts.length, 1);
  assert.match(request.url, /^\/prompt-reader\/comparisons\?/);
  assert.equal(request.init?.body, take.samples.buffer);
  const query = new URLSearchParams(request.url.split("?")[1]);
  assert.deepEqual(JSON.parse(query.get("setupIds")!), ids);
  assert.equal(query.has("prompt"), false);
  const requestId = query.get("requestId")!;
  acceptStream(stream, requestId);
  stream.send(rowEvent(requestId, ids[1]!));
  await settle();
  assert.equal(h.reader.comparison.rows[0].result, null);
  assert.equal(h.reader.comparison.rows[1].result.text, "raw words\n");
  assert.equal(h.reader.elements.save.disabled, false);
  assert.equal(h.reader.elements.play.disabled, false);
  await h.reader.play();
  assert.deepEqual(Array.from(h.played[0]!.samples), Array.from(take.samples));
  await h.reader.save();
  assert.equal(posts[1]!.init?.body, take.samples.buffer);
  assert.equal(request.init?.signal?.aborted, true);
  stream.send(rowEvent(requestId, ids[0]!, "complete", "late text"));
  stream.end();
  await settle();
  assert.equal(h.reader.comparison.rows.length, 0);
  assert.equal(h.reader.index, 1);
});

test("selected saved comparison posts only its filename and never decodes or uploads playback samples", async () => {
  const stream = streamReply();
  const posts: Array<{ url: string; init?: FetchInit }> = [];
  const h = pageHarness({ respond: async (url, init) => { posts.push({ url, init }); return stream.reply; } });
  h.reader.prompts[0].takes = ["first.wav", "second.wav"];
  enableComparison(h.reader);
  const running = h.reader.comparison.run();
  await settle();
  const posted = JSON.parse(posts[0]!.init?.body as string);
  assert.deepEqual(posted.source, { kind: "saved", takeName: "second.wav" });
  assert.deepEqual(posted.setupIds, ["mlx-parakeet/v3-greedy"]);
  assert.equal(h.pending.length, 0);
  assert.equal(h.played.length, 0);
  h.reader.stepTake();
  assert.equal(posts[0]!.init?.signal?.aborted, true);
  stream.end();
  await running;
  assert.equal(h.reader.savedTake().name, "first.wav");
});

test("cancel retains completed peers; abrupt EOF does not report all success", async () => {
  const stream = streamReply();
  const h = pageHarness({ respond: async () => stream.reply });
  const ids = ["mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy"];
  enableComparison(h.reader, ids);
  await readyTake(h);
  const job = h.reader.comparison.job;
  acceptStream(stream, job.requestId);
  stream.send(rowEvent(job.requestId, ids[0]!));
  await settle();
  h.reader.comparison.cancel();
  assert.equal(h.reader.comparison.rows[0].result.text, "raw words\n");
  assert.equal(h.reader.comparison.rows[1].phase, "cancelled");
  stream.end();
  await settle();
  assert.equal(h.reader.comparison.rows[1].phase, "cancelled");

  const incomplete = streamReply();
  const second = pageHarness({ respond: async () => incomplete.reply });
  enableComparison(second.reader);
  await readyTake(second);
  acceptStream(incomplete, second.reader.comparison.job.requestId);
  incomplete.end();
  await settle();
  assert.match(second.reader.comparison.message, /incomplete/);
});

test("native controls retain Space, Enter, arrows and letters without reader shortcuts", () => {
  const { reader, dispatch } = pageHarness();
  let actions = 0;
  reader.press = reader.save = reader.play = reader.stepTake = reader.move = () => { actions++; };
  for (const key of [" ", "Enter", "ArrowLeft", "ArrowRight", "p", "t"]) {
    const target = { closest: () => ({}) };
    dispatch("window:keydown", { key, code: key === " " ? "Space" : key, target, preventDefault: () => { actions++; } });
    dispatch("window:keyup", { key, code: key === " " ? "Space" : key, target, preventDefault: () => { actions++; } });
  }
  assert.equal(actions, 0);
});

test("only a reader owned Space keyup releases recording", () => {
  const { reader, dispatch } = pageHarness();
  let releases = 0;
  reader.press = () => { reader.gate.press(); };
  reader.gate.release = () => { releases++; };
  const background = { closest: () => null };
  const event = { key: " ", code: "Space", target: background, preventDefault() {} };
  dispatch("window:keyup", event);
  assert.equal(releases, 0);
  dispatch("window:keydown", event);
  dispatch("window:keyup", { ...event, target: { closest: () => ({}) } });
  assert.equal(releases, 1);
});

for (const status of ["complete", "cancelled"]) test(`empty/partial output and ${status} comparison termination retain their meanings`, async () => {
  const stream = streamReply();
  const h = pageHarness({ respond: async () => stream.reply });
  enableComparison(h.reader, ["mlx-parakeet/v3-greedy", "mlx-cleanup/qwen-targeted"]);
  await readyTake(h);
  const job = h.reader.comparison.job;
  acceptStream(stream, job.requestId);
  stream.send(rowEvent(job.requestId, "mlx-parakeet/v3-greedy", "complete", ""));
  stream.send(rowEvent(job.requestId, "mlx-cleanup/qwen-targeted", "complete", ""));
  stream.send({ type: "done", requestId: job.requestId, comparisonId: "server-job", status, wallMs: 100 });
  await settle();
  assert.equal(h.reader.comparison.rows[0].view.text.textContent, "No speech detected");
  assert.equal(h.reader.comparison.rows[1].view.text.textContent, "Empty output");
  assert.equal(h.reader.comparison.message, status === "cancelled" ? "Comparison cancelled" : "Comparison finished");
  assert.equal(h.reader.comparison.job, null, "terminal event settles the job without waiting for EOF");
  const failed = rowEvent("next", "mlx-cleanup/qwen-targeted", "error", null);
  (failed.row as any).error = { code: "execution", message: "cleanup failed", rawText: "  partial\n" };
  const next = { requestId: "next", comparisonId: "server-job", controller: new AbortController() };
  h.reader.comparison.job = next;
  h.reader.comparison.rows[1].result = null;
  h.reader.comparison.accept(failed, next);
  assert.equal(h.reader.comparison.rows[1].view.phase.textContent, "Incomplete output");
  assert.equal(h.reader.comparison.rows[1].view.text.textContent, "  partial\n");
  h.reader.comparison.cancel();
});
