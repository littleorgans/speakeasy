# Native runtime candidates (E3)

Three candidates share the harness's TypeScript driver, [driver.ts](../driver.ts), with `render-baselines.ts`: [kokoro-coreml](kokoro-coreml/README.md) (Kokoro v1.0 on Core ML through FluidAudio), [supertonic3](supertonic3/README.md) (Supertonic 3 ONNX in Node) and [piper-split](piper-split/README.md) (Kristin with libpiper's split decoder). Each `render.ts` supplies only an `Adapter`; the driver gives it the flags, cold and warm shapes, clips and result JSON of `render-baselines.ts` and [CONTRACT.md](../CONTRACT.md). It is TypeScript because the adapters reuse speak-easy's own `planSegments`, `trimSilence` and Piper Plus frontend from `packages/speech-io/src/tts/`, none of which loads a native addon, so no `pnpm install` is needed.

## Native worker protocol

kokoro-coreml and piper-split run their engine in a resident native process (`NativeWorker` in the driver). The worker moves its frame stream off fd 1 before loading anything, so library logging cannot corrupt it.

- Request, one line on stdin: `S<TAB>part<US>part...` synthesizes the parts in order; `Q` quits.
- Reply frames on stdout: little endian int32 kind, int32 count, payload. `READY` 1 and `END` 3 and `ERROR` 4 carry `count` bytes of UTF-8 JSON; `AUDIO` 2 carries `count` float32 samples.
- Timing: the adapter's clock starts before the request is written and a chunk is ready when its `AUDIO` frame has been read, so IPC and the text frontend are inside every number.

## In run-timed.sh

`run-timed.sh` runs the seven engines below by default and exports the artefact paths, defaulting to `$TMPDIR` as the READMEs build them: `KOKORO_ANE_WORKER`, `SUPERTONIC_NODE`, `PIPER_SPLIT_WORKER`. Engines: `kokoro-coreml-heart kokoro-coreml-michael supertonic3-f1 supertonic3-m1 supertonic3-f1-t4 piper-split-t4 piper-split-carved-t4`.

`kokoro-ane-*` ([kokoro-ane](../../kokoro-ane/README.md), mattmireles/kokoro-coreml on the Neural Engine) run only when named in `ENGINES`; `KOKORO_COREML_WORKER` names their worker.

`kokoro-coreml-v3-heart` stays out of `ENGINES`: it aborts on macOS 27.0.1 (kokoro-coreml README). The `RUNS=1` smoke of the seven engines took 5 minutes under load average 8 to 12; at `RUNS=5` expect about 20 to 25 minutes, Supertonic about two thirds of it.
