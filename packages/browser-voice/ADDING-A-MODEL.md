# Adding a model to the road test

The road test page lists every local speech in setup and every local voice
the host knows. Add a model to the registry below and it appears in the
page, and speech in setups also appear in the prompt reader comparison.

## Start the road test

From the repo root:

```sh
pnpm road-test
```

It serves `http://127.0.0.1:4870/` with no further setup:

- Voices load from `models/tts` under the repo root.
- Speech in setups read their paths from `models/road-test-runtime.json`
  (gitignored, one per machine). If it is missing, the page still opens; the
  console names the file and each setup that needs it shows
  "needs setup: setup runtime file missing". The sherpa CPU setups need no
  entry: they read `models/sherpa`.

Flags after `pnpm road-test --` or environment values override either default:
`--setup-runtime` / `SPEAKEASY_SETUP_RUNTIME` and `--output-root` /
`SPEAKEASY_OUTPUT_ROOT`. The port works the same way, `pnpm road-test -- --port 4873`:
a flag given twice takes its last value.

### The setup runtime file

JSON with absolute server paths only; `road-test-runtime.example.json` in this
package shows every entry. Omit a setup to leave it unavailable.

- `python.<setup id>`: `pythonPath` (the interpreter that runs
  `runtime/worker.py`) and `modelPath` (file or snapshot directory), plus
  `libraryPath` for `qwen/context` or `runnerPath` for `hex/unified`.
  Optional `loadTimeoutMs` and `operationTimeoutMs`, positive integers,
  default 120000.
- `cpu.modelRoot`: the directory holding `parakeet-ultra-sherpa-onnx-int8`,
  shared by every sherpa CPU setup; defaults to `models/sherpa`.
- `voices.<voice id>`: `pythonPath` for a worker voice (Speech out below),
  plus `runnerPath` where its backend drives a native binary. Optional
  `loadTimeoutMs` and `operationTimeoutMs` as above. Its weights come from the
  output root, so no model path. Without an entry the voice shows
  "needs setup: Voice runtime is not configured".

Unknown setups or fields are refused at startup.

## Speech in

1. Add an entry to `runtime/setups.json`. Required: `id`, `label`, `card`,
   `model`, `revision`, `runtime`, `controls`, `operation` (`transcribe`) and
   `maxSamples`. `card.runtime` is the plain text runtime line on the model
   card; `card.transcript` is `final` or `partials` (Live words below). The card's size is
   measured from the configured model files. A setup with `baseSetup` also
   loads that setup: it is ready only when its base is, and its size counts both.
2. Add the id to `SETUP_IDS` in `src/setup-protocol.ts`, in manifest order.
3. Recompute the manifest pin in `src/setup-registry.ts`
   (`SETUP_MANIFEST_SHA256`, the SHA256 of `JSON.stringify` of the parsed file).
4. Give it a backend. Python backends live in `runtime/<name>.py` with a
   `Backend` class (`load`, `transcribe`, `close`); register it in `BACKENDS` in
   `runtime/worker.py` and `PYTHON_WORKERS` in `src/setup-workers.ts`. A native
   binary goes behind a Python backend; `runtime/hex_unified.py` is the
   example. Its build script builds into a directory of its own under
   `~/Library/Caches/speakeasy/`; after a change to `runtime/hex/worker.rs`,
   rebuild to a new directory and point `runnerPath` at it, so a running
   server keeps the binary it started with.
5. Teach `src/server-options.ts` its files: a native path beyond the model goes
   in `EXTRA_LOCATIONS`, and `inspectSetupReadiness` checks the files exist.
6. Add its paths to `models/road-test-runtime.json` and to
   `road-test-runtime.example.json`, restart, and try it in the page.

A sherpa CPU setup with another fixed word list (`tm/today` beside
`sweep/score-2`) needs no backend or paths: add its preset, with the list and
its SHA256, to `CPU_PRESETS` in `src/transcriber.ts`, and the same list under
`controls.list` in its manifest entry. Each CPU setup runs in a child of its own.

### Live words

A setup whose engine truly streams sets `card.transcript` to `partials` and
gets an entry in `LIVE_STREAMS` in `src/live-settings.ts`, with the streaming
settings it exposes: each an id, label, one line note, the offered values and
the default, which is today's behaviour. Offer only values a take has shown
to hold. The page's Live words choice sits under Speech in: Off, or any live
stream, paired with any final setup (each runs in its own worker, and the
final pass waits for the live stream to settle). It defaults to the setup's
own stream, a `baseSetup`'s stream, or Off.

While the owner holds the key, the page streams the take's mic frames over
the `/live-words` socket, naming the live model and its settings; the host
refuses values not offered, answers with the stream and settings it applied,
then feeds the worker whenever it is free. The page shows the words one tone
quieter until the final text replaces them, and lists each update in the
take's revision trail. The final is still the final setup's full pass on the
whole take. The backend adds `feed(samples, start, settings)`: `start` opens
a fresh stream with `settings` (absent means the defaults; the backend checks
the same offered values), and it returns `stable` (settled text), `tentative`
(text the model may still change) and `processingMs`. Feed whole engine chunks
and keep a remainder for the next call (`stream_settings` in
`runtime/parakeet.py`, the 2,560 sample feeds in `runtime/hex/worker.rs`).
Never fake partials by decoding the growing take again: an engine that only
gives final text keeps `final`.

## Speech out

Every voice reads back through one sentence pipeline
(`packages/speech-io/src/tts/stream.ts`): it plans a short first part, then
whole sentences, and asks the voice to synthesize one part at a time, so
every voice streams by sentence. A sherpa-onnx `vits` or `kokoro` voice runs
in process on the CPU. Any other runtime (MLX, Core ML, a native binary) is a
worker voice: the same Python worker as speech in, answering `speak`.

1. Pin the payload files (name, bytes, SHA256; names may include a
   subdirectory) in `LOCAL_TTS_MODELS` in `packages/speech-io/src/tts/models.ts`.
   A worker voice uses `family: "worker"` with its `sampleRate`, `numSpeakers`
   and `attribution`; these pins are the only ones, as the worker receives
   them to attest the files before loading. A runtime that takes phonemes
   names the sherpa voice whose Piper Plus frontend the host borrows in
   `phonemesFrom`; readiness then checks that voice's files too.
2. Add the browser choice to `LOCAL_OUTPUT_MODELS` in
   `packages/speech-io/src/tts/catalogue.ts`: `id`, `label`, `runtime` (the
   card's runtime line), voices, sample rate and licence. The card's size is
   the sum of the pinned file bytes. A worker voice numbers its `speakerId`s
   from 0 in list order; the worker receives the voice `id`.
3. Put the payload under the output root and restart.

A worker voice also needs:

4. A backend: a `Backend` taking `setupId`, `modelPath` and `files`, with
   `load` (attest with `runtime/pins.py`, then load) and
   `speak(text, voice, speed)`, returning little endian Float32 `samples`
   bytes, `sampleRate` and `processingMs` for one part. Register it in
   `BACKENDS` and `VOICES` in `runtime/worker.py`. A Python runtime gets its
   own `runtime/<name>.py` (`runtime/kokoro_mlx.py` is the example). A native
   runner that speaks the fast-tts frame protocol needs only a `spec` line in
   `runtime/native_voice.py`, its source and a `build.py` under
   `runtime/<name>/` that builds into `~/Library/Caches/speakeasy/<name>/`,
   and `runnerPath` in the runtime file.
5. For a Python runtime, a pinned environment: freeze it to
   `runtime/requirements-<name>.txt`, build it under
   `~/Library/Caches/speakeasy/browser-voice-switch/environments/` with `uv`,
   then run `python -m compileall` on its site packages once; the worker
   starts Python with `-B`, so without compiled bytecode every start
   recompiles the libraries (30 s instead of 3 s for Kokoro MLX).
6. Its interpreter under `voices` in `models/road-test-runtime.json` and in
   `road-test-runtime.example.json`; a native runner also needs its id in
   `VOICE_EXTRA_LOCATIONS` in `src/server-options.ts`. The bridge needs only
   the standard library of Python 3.11 or later.

Every voice streams by sentence, as the pipeline does. A runtime that could
stream within a sentence (Pocket's frames, libpiper's decoder chunks) still
returns each part whole: the worker replies once per request.

## Check

`pnpm typecheck && pnpm test` from the repo root. The registry tests pin the
setup ids and revisions, so update them with the new entry. Backends have
standard library tests:
`python3 -m unittest discover -s packages/browser-voice/runtime`.
