# fast-tts measurement contract

Every engine in this experiment, baseline or candidate, on any runtime, follows this contract so its numbers merge with `summarize.py` and appear on the listening page. `fasttts.py` implements it for Python; `render-baselines.ts` is the TypeScript reference.

## Cohort

`cohort.json` holds the texts; every engine renders all of them, unchanged. Each text has an `id`, a `length` class and the `text`:

- `short`: one sentence, about 8 to 15 words.
- `medium`: two or three sentences.
- `long`: one paragraph of six or more sentences, about 120 words.

Ids shared with the audio8-tts cohort keep its wording, so results compare.

## Clock and endpoints

All times are milliseconds on a monotonic clock (`performance.now` in Node, `time.perf_counter_ns` in Python), measured from **text submit**: the moment the full text is handed to the engine. Never use wall clock time.

- **First audio** (`firstAudioMs`, endpoint a): submit to the first chunk of decoded samples available to the caller. Phonemes, tokens or codec frames do not count; PCM does.
- **Playback start** (`playbackStartMs`, endpoint b): submit to the first buffer of real samples the audio device accepted (in `fasttts.DevicePlayer`, the first output callback that copies queued samples). Only for runs that play audio, with the device opened before submit; otherwise `null`. Name the method in `process.playback`.

Always say which endpoint a number is. "TTFA" alone is ambiguous here.

## Per chunk timing

Each run records every chunk the engine emits: `readyMs` (submit to the chunk being available to the caller) and `audioMs` (its duration as played). Full mode, one call that returns all audio, is a single chunk.

## Derived metrics

`fasttts.derive` is the one definition; engines record raw runs and never compute these themselves.

- **RTF** = synthesis wall time / audio duration, per text: last chunk `readyMs` over the sum of chunk `audioMs`. Below 1 is faster than real time.
- **Underruns**: playback is simulated from the chunk list. It starts at `playbackStartMs`, else at the first chunk's `readyMs`, and plays chunks back to back. A chunk not ready when the previous one ends is an underrun; playback stalls until it is ready (`stallMs` sums the stalls).

## Cold and warm

- **Cold**: a fresh process loads the model, then synthesizes one text with no warmup. Report the load (`process.loadMs`: engine construction until it accepts text) separately from first audio. Only the first request in a process is cold, so a cold process runs exactly one text. Repeat in N fresh processes.
- **Warm**: a fresh process loads the model, runs one discarded warmup synthesis (`process.warmupMs`), then N passes over every cohort text.
- Report median and p90 over the N runs. Timed numbers default to N = 5.

## Processes

One fresh process per engine for timed runs, never two engines in one process. `run-timed.sh` runs them in sequence. Cited numbers come only from a timed pass on a quiet machine slot granted by the Space lead (`process.timed: true`); smoke runs set `timed: false` and are labelled untimed wherever shown.

## Result file

One JSON file per process, `schema: "fast-tts.result.v1"`. File names: `<engine>--cold--<text>--<i>.json`, `<engine>--warm--<mode>.json`.

```jsonc
{
  "schema": "fast-tts.result.v1",
  "engine": {
    "id": "kokoro-heart",                 // letters, digits, - _ . ; names clip files
    "label": "Kokoro v1.0 · af_heart",
    "runtime": "sherpa-onnx-node 2.0.1",  // runtime and version: MLX, CoreML, llama.cpp, PyTorch MPS ...
    "provider": "cpu",                    // execution provider or device
    "model": "kokoro-v1.0",
    "modelRevision": "sha256:b40f62b166ac", // weights digest or upstream revision
    "voice": "af_heart",
    "sampleRate": 24000
    // optional extras: threads, streaming, quantization, ...
  },
  "environment": {
    "machine": "Apple M2 Max", "arch": "arm64", "cpuCount": 12, "memoryGiB": 96,
    "os": "macOS 27.0.1", "versions": {"node": "25.9.0", "sherpa-onnx-node": "2.0.1"},
    "harnessCommit": "974793b17327"
  },
  "process": {
    "temperature": "warm",                // cold | warm
    "timed": false,
    "repeats": 5,
    "startedAt": "2026-11-02T20:49:27Z",
    "clock": "performance.now (monotonic)",
    "playback": null,                     // or how playbackStartMs was observed
    "loadMs": 1171.2,
    "warmupMs": 512.0,                    // null on cold
    "loadDetail": {}                      // optional, engine specific
  },
  "runs": [
    {
      "textId": "status", "mode": "stream", "temperature": "warm", "repeat": 0,
      "playbackStartMs": null,
      "chunks": [
        {"index": 0, "readyMs": 111.9, "audioMs": 1374.8, "text": "The deploy finished and"},
        {"index": 1, "readyMs": 289.6, "audioMs": 1600.2, "text": "the site is live again."}
      ]
    }
  ]
}
```

`mode` is `stream` (audio arrives in chunks before synthesis ends) or `full` (one call, all audio at once). `fasttts.validate` checks a document; `summarize.py` rejects files that fail it.

## Clips

The first warm repeat of each text is saved as mono WAV at the engine's native rate: `<textId>--<engineId>.wav` for stream, `<textId>--<engineId>--full.wav` for full, in the shared `assets/fast-tts/clips/`. `build_listen.py` finds clips by these names.

## Plugging in a candidate

1. Write one script per engine (Python: import `fasttts`; other languages: emit the same JSON).
2. Smoke it untimed, run `summarize.py` on its files and check the page with `build_listen.py`.
3. Add its fresh process blocks to `run-timed.sh` and wait for a quiet slot for the timed pass.
