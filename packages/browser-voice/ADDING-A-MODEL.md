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
  "needs setup: setup runtime file missing". The sherpa CPU setup needs no
  entry: it reads `models/sherpa`.

Flags after `pnpm road-test --` or environment values override either default:
`--setup-runtime` / `SPEAKEASY_SETUP_RUNTIME` and `--output-root` /
`SPEAKEASY_OUTPUT_ROOT`.

### The setup runtime file

JSON with absolute server paths only; `road-test-runtime.example.json` in this
package shows every entry. Omit a setup to leave it unavailable.

- `python.<setup id>`: `pythonPath` (the interpreter that runs
  `runtime/worker.py`) and `modelPath` (file or snapshot directory), plus
  `libraryPath` for `qwen/context` or `runnerPath` for `hex/unified`.
  Optional `loadTimeoutMs` and `operationTimeoutMs`, positive integers,
  default 120000.
- `cpu.modelRoot`: the directory holding `parakeet-ultra-sherpa-onnx-int8`;
  defaults to `models/sherpa`.

Unknown setups or fields are refused at startup.

## Speech in

1. Add an entry to `runtime/setups.json`. Required: `id`, `label`, `card`,
   `model`, `revision`, `runtime`, `controls`, `operation` (`transcribe`) and
   `maxSamples`. `card.runtime` is the plain text runtime line on the model
   card; `card.transcript` is `final` or `partials`. The card's size is
   measured from the configured model files.
2. Add the id to `SETUP_IDS` in `src/setup-protocol.ts`, in manifest order.
3. Recompute the manifest pin in `src/setup-registry.ts`
   (`SETUP_MANIFEST_SHA256`, the SHA256 of `JSON.stringify` of the parsed file).
4. Give it a backend. Python backends live in `runtime/<name>.py` with a
   `Backend` class (`load`, `transcribe`, `close`); register it in `BACKENDS` in
   `runtime/worker.py` and `PYTHON_SETUPS` in `src/setup-workers.ts`. A native
   binary goes behind a Python backend; `runtime/hex_unified.py` is the
   example.
5. Teach `src/server-options.ts` its files: a native path beyond the model goes
   in `EXTRA_LOCATIONS`, and `inspectSetupReadiness` checks the files exist.
6. Add its paths to `models/road-test-runtime.json` and to
   `road-test-runtime.example.json`, restart, and try it in the page.

## Speech out

1. Pin the payload files (name, bytes, SHA256) in `LOCAL_TTS_MODELS` in
   `packages/speech-io/src/tts/models.ts`.
2. Add the browser choice to `LOCAL_OUTPUT_MODELS` in
   `packages/speech-io/src/tts/catalogue.ts`: `id`, `label`, `runtime` (the
   card's runtime line), voices, sample rate and licence. The card's size is
   the sum of the pinned file bytes.
3. Put the payload under the output root and restart. A family other than
   sherpa-onnx `vits` or `kokoro` also needs synthesis support in
   `packages/speech-io/src/tts/synth.ts`.

## Check

`pnpm typecheck && pnpm test` from the repo root. The registry tests pin the
setup ids and revisions, so update them with the new entry.
