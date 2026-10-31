import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, realpath, rm, stat, symlink, writeFile } from "node:fs/promises";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { readCorpusEntries, readWavFrames } from "@speakeasy/speech-io";
import { startBrowserVoiceServer, type BrowserVoiceServer } from "./host.ts";
import { PROMPT_MAX_TAKE_BYTES, type PromptListing } from "./prompt-reader.ts";
import { ChildTranscriber, type Transcriber } from "./transcriber.ts";

const PROMPTS = {
  set: "T",
  items: [
    { id: "p002", stratum: "long", source: "written", edited: false, text: "Open .lilo/docs/index.md first." },
    { id: "p001", stratum: "short", source: "past", edited: false, text: "yes do 561 and 563 now" },
  ],
};

async function withReader(
  run: (host: BrowserVoiceServer, outputDir: string, root: string) => Promise<void>,
  options: { transcriber?: (root: string) => Promise<Transcriber> } = {},
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-reader-"));
  const promptsPath = join(root, "prompts.json");
  const outputDir = join(root, "takes");
  await writeFile(promptsPath, JSON.stringify(PROMPTS));
  const transcriber = await options.transcriber?.(root);
  const host = await startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir }, transcriber });
  try {
    await run(host, outputDir, root);
  } finally {
    await host.close();
    await rm(root, { recursive: true, force: true });
  }
}

/** POST through node:http so the test controls Origin and Host exactly. */
function post(
  host: BrowserVoiceServer,
  path: string,
  options: { body: Uint8Array; headers?: Record<string, string> },
): Promise<{ status: number; body: Record<string, unknown> }> {
  const url = new URL(path, host.url);
  return new Promise((resolve, reject) => {
    const outgoing = request(
      url,
      {
        method: "POST",
        headers: {
          origin: host.url,
          "content-type": "application/octet-stream",
          ...options.headers,
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () =>
          resolve({
            status: response.statusCode ?? 0,
            body: JSON.parse(Buffer.concat(chunks).toString() || "{}"),
          }),
        );
      },
    );
    outgoing.on("error", reject);
    outgoing.end(options.body);
  });
}

/** GET a raw, unnormalized path so traversal attempts reach the host as sent. */
function get(
  host: BrowserVoiceServer,
  path: string,
  headers: Record<string, string> = {},
): Promise<{ status: number; type: string | undefined; body: Buffer }> {
  const { hostname, port } = new URL(host.url);
  return new Promise((resolve, reject) => {
    const outgoing = request({ hostname, port, path, method: "GET", headers }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.on("end", () =>
        resolve({ status: response.statusCode ?? 0, type: response.headers["content-type"], body: Buffer.concat(chunks) }),
      );
    });
    outgoing.on("error", reject);
    outgoing.end();
  });
}

/** Requests every take route refuses, with the status each must answer. */
function refusedTakes(host: BrowserVoiceServer): Array<[string, Parameters<typeof post>[2], number]> {
  const port = new URL(host.url).port;
  return [
    ["foreign origin", { body: takeBytes(320), headers: { origin: "https://example.com" } }, 403],
    ["missing origin", { body: takeBytes(320), headers: { origin: "" } }, 403],
    ["rebound host", { body: takeBytes(320), headers: { host: `evil.test:${port}`, origin: `http://evil.test:${port}` } }, 403],
    ["wrong content type", { body: takeBytes(320), headers: { "content-type": "text/plain" } }, 415],
    ["empty body", { body: new Uint8Array(0) }, 400],
    ["partial sample", { body: new Uint8Array(6) }, 400],
    ["oversize take", { body: new Uint8Array(PROMPT_MAX_TAKE_BYTES + 4) }, 413],
  ];
}

function takeBytes(samples: number, value = 0.25): Uint8Array {
  return new Uint8Array(new Float32Array(samples).fill(value).buffer);
}

test("the prompt reader page and its scripts are served", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  try {
    const page = await fetch(`${host.url}/prompt-reader`);
    assert.equal(page.status, 200);
    assert.match(await page.text(), /Speak Easy · Prompt reader/);
    for (const asset of ["/prompt-reader.js", "/prompt-reader.css", "/capture-constraints.js", "/capture-worklet.js", "/hold-release.js"]) {
      assert.equal((await fetch(`${host.url}${asset}`)).status, 200, asset);
    }
    const prompts = await fetch(`${host.url}/prompt-reader/prompts`);
    assert.equal(prompts.status, 404);
    assert.match(((await prompts.json()) as { error: string }).error, /--prompts <file> --out <dir>/);
  } finally {
    await host.close();
  }
});

test("prompts list in id order and a saved take becomes a schema 2 corpus pair", async () => {
  await withReader(async (host, outputDir) => {
    const before = (await (await fetch(`${host.url}/prompt-reader/prompts`)).json()) as PromptListing;
    assert.equal(before.set, "T");
    assert.equal(before.outputDir, join(await realpath(dirname(outputDir)), basename(outputDir)));
    assert.deepEqual(
      before.prompts.map((prompt) => [prompt.id, prompt.recorded]),
      [["p001", false], ["p002", false]],
    );
    assert.deepEqual(before.prompts.map((prompt) => prompt.takes), [[], []]);

    const saved = await post(host, "/prompt-reader/recordings?promptId=p002&device=Desk%20mic", {
      body: takeBytes(16_000),
    });
    assert.equal(saved.status, 201);
    assert.match(String(saved.body.audio), /^utt-p002-.*\.wav$/);

    const [entry] = await readCorpusEntries(outputDir);
    assert.equal(entry?.sidecar.schema, 2);
    assert.equal(entry?.sidecar.promptId, "p002");
    assert.equal(entry?.sidecar.stratum, "long");
    assert.equal(entry?.sidecar.expected, "Open .lilo/docs/index.md first.");
    assert.equal(entry?.sidecar.flushToFinalMs, null);
    assert.equal(entry?.sidecar.device, "Desk mic");
    assert.equal(entry?.sidecar.peakLevel, 0.25);
    // 16 kHz mono Float32: 44 byte header plus 4 bytes per sample.
    assert.equal((await stat(entry!.wavPath)).size, 44 + 16_000 * 4);
    assert.equal((await readFile(entry!.wavPath)).readUInt32LE(24), 16_000);

    const after = (await (await fetch(`${host.url}/prompt-reader/prompts`)).json()) as PromptListing;
    assert.deepEqual(
      after.prompts.map((prompt) => prompt.recorded),
      [false, true],
    );
    assert.deepEqual(after.prompts[1]?.takes, [saved.body.audio]);
  });
});

test("the save route refuses foreign pages, bad bodies and unknown prompts", async () => {
  await withReader(async (host, outputDir) => {
    const path = "/prompt-reader/recordings?promptId=p001";
    for (const [label, options, status] of refusedTakes(host)) {
      assert.equal((await post(host, path, options)).status, status, label);
    }
    assert.equal(
      (await post(host, "/prompt-reader/recordings?promptId=p999", { body: takeBytes(320) })).status,
      404,
    );
    assert.equal(
      (await post(host, "/prompt-reader/recordings?promptId=..%2Fescape", { body: takeBytes(320) })).status,
      404,
    );
    await assert.rejects(readdir(outputDir), { code: "ENOENT" }, "nothing was written");
  });
});

test("an output folder inside the repo is refused before the host starts", async () => {
  const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
  const root = await mkdtemp(join(tmpdir(), "speakeasy-reader-"));
  const promptsPath = join(root, "prompts.json");
  const inside = join(repoRoot, "corpus-should-not-exist");
  try {
    await writeFile(promptsPath, JSON.stringify(PROMPTS));
    for (const outputDir of [repoRoot, inside, join(repoRoot, "..takes")]) {
      const starting = startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir } });
      await starting.then((host) => host.close(), () => {});
      await assert.rejects(starting, /must be outside the repo/);
    }
    await assert.rejects(stat(inside), { code: "ENOENT" });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("malformed prompt sets are refused", async () => {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-reader-"));
  const promptsPath = join(root, "prompts.json");
  try {
    for (const [content, pattern] of [
      [{ prompts: [] }, /items array/],
      [{ items: [{ id: "p001", stratum: "short", text: " " }] }, /non-empty text/],
      [{ items: [PROMPTS.items[0], PROMPTS.items[0]] }, /unique/],
      [{ items: [{ id: "p.001", stratum: "short", text: "read this" }] }, /items\[0\] id "p\.001" must match/],
    ] as const) {
      await writeFile(promptsPath, JSON.stringify(content));
      // Close even an incorrectly accepted host, so the regression terminates.
      const starting = startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir: join(root, "takes") } });
      await starting.then((host) => host.close(), () => {});
      await assert.rejects(starting, pattern);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("output paths through a symlink into the repo are refused", async () => {
  const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
  const root = await mkdtemp(join(tmpdir(), "speakeasy-reader-"));
  const promptsPath = join(root, "prompts.json");
  try {
    await writeFile(promptsPath, JSON.stringify(PROMPTS));
    await symlink(repoRoot, join(root, "repo-link"));
    for (const outputDir of [join(root, "repo-link"), join(root, "repo-link", "missing", "takes")]) {
      // Close even an incorrectly accepted host, so the regression terminates.
      const starting = startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir } });
      await starting.then((host) => host.close(), () => {});
      await assert.rejects(starting, /must be outside the repo/);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("the saved WAV preserves every captured Float32 sample at 16 kHz", async () => {
  await withReader(async (host, outputDir) => {
    const samples = Float32Array.from({ length: 640 }, (_, index) => Math.sin(index / 13) * 0.8);
    const saved = await post(host, "/prompt-reader/recordings?promptId=p001", {
      body: new Uint8Array(samples.buffer),
    });
    assert.equal(saved.status, 201);
    const [entry] = await readCorpusEntries(outputDir);
    const wav = await readWavFrames(entry!.wavPath);
    assert.equal(wav.sampleRate, 16_000);
    assert.equal(wav.channels, 1);
    assert.equal(wav.samples.length, samples.length);
    assert.equal(wav.samples.findIndex((sample, index) => sample !== samples[index]), -1, "sample values changed");
    assert.equal(wav.bitsPerSample, 32);
  });
});

test("saved takes play back read only; every other path is refused", async () => {
  await withReader(async (host, outputDir) => {
    const first = await post(host, "/prompt-reader/recordings?promptId=p001", { body: takeBytes(320, 0.1) });
    const second = await post(host, "/prompt-reader/recordings?promptId=p001", { body: takeBytes(320, 0.2) });
    const listing = (await (await fetch(`${host.url}/prompt-reader/prompts`)).json()) as PromptListing;
    assert.deepEqual(listing.prompts[0]?.takes, [first.body.audio, second.body.audio]);

    const [entry] = await readCorpusEntries(outputDir);
    const onDisk = await readFile(entry!.wavPath);
    const served = await get(host, `/prompt-reader/takes/${encodeURIComponent(String(first.body.audio))}`);
    assert.equal(served.status, 200);
    assert.equal(served.type, "audio/wav");
    assert.ok(served.body.equals(onDisk), "served bytes match the take");

    // A wav the reader did not write, a sidecar and a file outside the folder.
    await writeFile(join(outputDir, "foreign.wav"), onDisk);
    await writeFile(join(dirname(outputDir), "outside.wav"), onDisk);
    const sidecar = basename(entry!.sidecarPath);
    for (const path of [
      "/prompt-reader/takes/foreign.wav",
      `/prompt-reader/takes/${sidecar}`,
      "/prompt-reader/takes/",
      "/prompt-reader/takes/../outside.wav",
      "/prompt-reader/takes/..%2Foutside.wav",
      "/prompt-reader/takes/%2E%2E%2Foutside.wav",
      `/prompt-reader/takes/${encodeURIComponent(join(dirname(outputDir), "outside.wav"))}`,
      `/prompt-reader/takes/sub%2F..%2F${first.body.audio}`,
      "/prompt-reader/takes/%E0%A4%A",
    ]) {
      const refused = await get(host, path);
      assert.equal(refused.status, 404, path);
      assert.notEqual(refused.type, "audio/wav", path);
    }
    const port = new URL(host.url).port;
    assert.equal(
      (await get(host, `/prompt-reader/takes/${first.body.audio}`, { host: `evil.test:${port}` })).status,
      403,
      "rebound host",
    );
    assert.ok((await readFile(entry!.wavPath)).equals(onDisk), "playback left the take unchanged");
  });
});

test("the take route is closed when the reader is not configured", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  try {
    assert.equal((await get(host, "/prompt-reader/takes/utt-p001-x.wav")).status, 404);
  } finally {
    await host.close();
  }
});

test("the transcription route validates like the save route and writes nothing", async () => {
  const heard: number[] = [];
  const transcriber: Transcriber = {
    async transcribe(samples) {
      heard.push(samples.length);
      return { text: "Read this.", decodeMs: 12, model: "Fake" };
    },
    close() {},
  };
  await withReader(async (host, outputDir) => {
    const path = "/prompt-reader/transcriptions";
    for (const [label, options, status] of refusedTakes(host)) {
      assert.equal((await post(host, path, options)).status, status, label);
    }
    assert.deepEqual(heard, [], "refused takes never reach the transcriber");
    const answered = await post(host, path, { body: takeBytes(320) });
    assert.equal(answered.status, 200);
    assert.deepEqual(answered.body, { text: "Read this.", decodeMs: 12, model: "Fake" });
    assert.deepEqual(heard, [320]);
    await assert.rejects(readdir(outputDir), { code: "ENOENT" }, "nothing was written");
  }, { transcriber: async () => transcriber });
});

test("a crashing transcriber child leaves the host serving and is forked again", async () => {
  // A stand in worker: a take whose first sample is 1 kills the child, as a
  // native sherpa abort would; any other take answers with its length.
  const worker = async (root: string) => {
    const workerPath = join(root, "worker.mjs");
    await writeFile(workerPath, `
      process.on("message", ({ id, samples }) => {
        if (samples[0] === 1) process.kill(process.pid, "SIGKILL");
        else process.send({ id, text: "heard " + samples.length, decodeMs: 3 });
      });
      process.on("disconnect", () => process.exit(0));
    `);
    return new ChildTranscriber({ workerPath, label: "Stand in" });
  };
  await withReader(async (host) => {
    const path = "/prompt-reader/transcriptions";
    const crashed = await post(host, path, { body: takeBytes(320, 1) });
    assert.equal(crashed.status, 503);
    assert.match(String(crashed.body.error), /transcriber exited/);
    assert.equal((await fetch(`${host.url}/health`)).status, 200);
    const answered = await post(host, path, { body: takeBytes(640) });
    assert.equal(answered.status, 200);
    assert.deepEqual(answered.body, { text: "heard 640", decodeMs: 3, model: "Stand in" });
    const saved = await post(host, "/prompt-reader/recordings?promptId=p001", { body: takeBytes(320) });
    assert.equal(saved.status, 201);
  }, { transcriber: worker });
});
