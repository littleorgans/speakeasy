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

const PROMPTS = {
  set: "T",
  items: [
    { id: "p002", stratum: "long", source: "written", edited: false, text: "Open .lilo/docs/index.md first." },
    { id: "p001", stratum: "short", source: "past", edited: false, text: "yes do 561 and 563 now" },
  ],
};

async function withReader(
  run: (host: BrowserVoiceServer, outputDir: string) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-reader-"));
  const promptsPath = join(root, "prompts.json");
  const outputDir = join(root, "takes");
  await writeFile(promptsPath, JSON.stringify(PROMPTS));
  const host = await startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir } });
  try {
    await run(host, outputDir);
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
  });
});

test("the save route refuses foreign pages, bad bodies and unknown prompts", async () => {
  await withReader(async (host, outputDir) => {
    const path = "/prompt-reader/recordings?promptId=p001";
    const port = new URL(host.url).port;
    const cases: Array<[string, Parameters<typeof post>[2], number]> = [
      ["foreign origin", { body: takeBytes(320), headers: { origin: "https://example.com" } }, 403],
      ["missing origin", { body: takeBytes(320), headers: { origin: "" } }, 403],
      ["rebound host", { body: takeBytes(320), headers: { host: `evil.test:${port}`, origin: `http://evil.test:${port}` } }, 403],
      ["wrong content type", { body: takeBytes(320), headers: { "content-type": "text/plain" } }, 415],
      ["empty body", { body: new Uint8Array(0) }, 400],
      ["partial sample", { body: new Uint8Array(6) }, 400],
      ["oversize take", { body: new Uint8Array(PROMPT_MAX_TAKE_BYTES + 4) }, 413],
    ];
    for (const [label, options, status] of cases) {
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
    ] as const) {
      await writeFile(promptsPath, JSON.stringify(content));
      await assert.rejects(
        startBrowserVoiceServer({ port: 0, promptReader: { promptsPath, outputDir: join(root, "takes") } }),
        pattern,
      );
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
