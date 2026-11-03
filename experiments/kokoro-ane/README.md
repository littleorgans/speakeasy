# kokoro-ane: Kokoro on the Neural Engine beside a busy LLM

Does a Kokoro route that runs on the Mac's Neural Engine (ANE) hold its speed beside Llama 3.2 3B better than Kokoro MLX? Follows the Space's ane-check, which found FluidAudio's legacy Kokoro Core ML (`kokoro-coreml-heart`) runs on the CPU in practice. Engines follow the fast-tts [CONTRACT.md](../fast-tts/CONTRACT.md) through its [driver](../fast-tts/candidates/NATIVE.md), so `run-timed.sh` and `summarize.py` time them like `kokoro-coreml-heart`.

## Routes

- **mattmireles/kokoro-coreml** (`render.ts`, `worker/`): its KokoroTTS SDK in a resident Swift worker. Builds with Xcode 27.2 plus `xcodebuild -downloadComponent MetalToolchain`; Command Line Tools alone stop in mlx-swift's Metal kernels ([kokoro-coreml README](../fast-tts/candidates/kokoro-coreml/README.md)).
- **FluidAudio v3** (`kokoro-coreml-v3-heart` in fast-tts): still aborts on macOS 27.0.1 with Xcode 27.2, in MPSGraph inside its GPU source generator, at every real length tried (`tools/v3sg.swift`). That generator plans 0 of its 1085 ops to the ANE, so no compute unit choice puts it there. Upstream FluidAudio `ee78635` (2026-11-02) skips it on OS 27 for the base Vocoder chain, whose Vocoder ran on the ANE only at its exact compiled length here, as in ane-check. Not adopted; not wrapped here.

## Engines

The starter bundle has one 15 s acoustic bucket and a 128 token duration shape, so every chunk runs at fixed, padded shapes, the shapes Core ML compiled for the ANE. A chunk costs about the same at any length.

| Engine | Generator | Placement observed |
| --- | --- | --- |
| `kokoro-ane-heart` | hybrid, the SDK default: 53 convolutions on the ANE, normalization and activations as Metal kernels | ANE plus a little GPU |
| `kokoro-ane-folded-heart` | row folded, `KOKORO_HYBRID_GENERATOR=0` | ANE only, no GPU |
| `kokoro-ane-cpu-heart` | folded on the CPU, `KokoroComputePolicy.cpuOnly` | CPU only; the placement control |

Duration front, F0 back and decoder pre run on the ANE in all but the CPU engine; duration back and F0 front (LSTMs) run on the CPU. The worker fails a request whose chunks ran another generator than the engine's, so a silent fallback never gets a wrong label.

## Pins

| What | Pin |
| --- | --- |
| SDK | `mattmireles/kokoro-coreml@b23caa862f0585c2c65187d0fc97e0f8311041dc` (SwiftPM revision pin, `worker/Package.resolved`); MisakiSwift `3a27756a`, mlx-swift 0.31.4 through it. Xcode 27.2 (27B52), Metal Toolchain 27B47, macOS 27.0.1 |
| Weights | `mattmireles/kokoro-coreml@034f7741bf938fd6b4b9c75c22bc72cd795f908a` starter bundle (`HostedManifest.json` sha256 `f3abd581...`, the SDK's `KokoroHostedModels` pin, each file hash checked on download); `af_heart` only; 208 MB, plus 198 MB compiled on device |

Assets sit under `MODELS/kokoro-coreml-mattmireles-034f7741/`.

## Licences

SDK and MisakiSwift Apache 2.0 (their `LICENSE` files); weights derive from hexgrad/Kokoro-82M, Apache 2.0. The hosted bundle's own card licence and the full transitive review are not checked.

## Setup

From the speak-easy repo root. `kokoro-say` from the pinned SDK checkout downloads and verifies the bundle once.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
xcodebuild -downloadComponent MetalToolchain   # once per Xcode
git clone https://github.com/mattmireles/kokoro-coreml $TMPDIR/kokoro-coreml && git -C $TMPDIR/kokoro-coreml checkout b23caa862f0585c2c65187d0fc97e0f8311041dc
(cd $TMPDIR/kokoro-coreml && swift run -c release kokoro-say --cache-dir $MODELS/kokoro-coreml-mattmireles-034f7741 --out $TMPDIR/kokoro.wav)
(cd experiments/kokoro-ane/worker && swift build -c release --scratch-path $TMPDIR/kokoro-coreml-worker-build)
export KOKORO_COREML_WORKER=$TMPDIR/kokoro-coreml-worker-build/release/KokoroCoreMLWorker
```

The first synthesis after a build took 10 to 12 s once (Core ML compiling for the ANE, cause not isolated); later cold processes take about 1 s. `timed.sh` primes every engine before it times.

## Run

```bash
node experiments/kokoro-ane/render.ts --root $MODELS --engine kokoro-ane-heart --temperature warm --repeats 1 \
  --json $TMPDIR/kokoro-ane-heart--warm--stream.json
ENGINES="kokoro-ane-heart kokoro-ane-folded-heart" OUT=$TMPDIR/runs bash experiments/fast-tts/run-timed.sh
```

## Timed pass and placement tools

- `timed.sh`: ane-check's slot method. Idle, then beside continuous Llama 3.2 3B generation (`tools/llm_load.py`, mlx_lm, started 20 s before): `run-timed.sh` stream at `RUNS=5` for Kokoro MLX, Kokoro Core ML legacy and both ANE engines, then `KokoroAneProbe` per generator. Samples the top processes by CPU each minute and stamps each phase for LLM tokens/s.
- `worker/` `KokoroAneProbe`: whole text per call, wall and process CPU, stage times; one process per generator.
- `tools/plan.swift`: MLComputePlan per op, each function of a multifunction model with `PLAN_FUNCTIONS`. `tools/trace_sum.py`: ANE and GPU busy time from an Instruments Core ML trace (`xcrun xctrace record --template 'Core ML' --launch -- ...`). `tools/v3sg.swift`: FluidAudio v3's source generator at a given frame count.

## Results (partial timed pass, 2026-11-03)

One slot, stopped by owner direction at its last step; every engine run completed. Warm short first audio, median, idle vs beside Llama 3.2 3B:

| Engine | Idle | Load |
| --- | --- | --- |
| Kokoro MLX | 68 ms | 165 ms |
| `kokoro-ane-heart` | 115 ms | 443 ms |
| `kokoro-ane-folded-heart` | 345 ms | 573 ms |

Cold first audio under load was 2.2 to 4.3 s for the ANE engines, against 134 to 214 ms for MLX. In that window the ANE route did not hold speed better than MLX. Outside load rose sharply in the second half of the load pass, when the ANE engines ran, so the ranking under load is unverified. Full tables live in the kokoro-ane thread's `results.md`.

## For the road test page

The runner is `KokoroCoreMLWorker`, a resident process that speaks the fast-tts native worker protocol ([NATIVE.md](../fast-tts/candidates/NATIVE.md)).

- **Setup:** Xcode 27.2 or later, then `xcodebuild -downloadComponent MetalToolchain` once; fetch the weights and `swift build` the worker as in Setup above. Weights live in `MODELS/kokoro-coreml-mattmireles-034f7741/` (`KokoroRuntimeManifest.json`, `coreml/`, `voices/af_heart.bin`; Core ML compiles into `compiled-v2/` there on first use).
- **Start:** `KokoroCoreMLWorker <bundle root> af_heart <hybrid|folded>`. The third argument picks the generator. `hybrid` is `kokoro-ane-heart`: faster when idle, but it uses some GPU. `folded` is `kokoro-ane-folded-heart`: ANE only, no GPU, about 3x slower when idle. `cpu` exists as a control only.
- **Requests,** one line on stdin: `S<TAB>part<US>part...` (US is `\u001f`) synthesizes the parts in order; `Q` quits. Parts must not contain tab, newline or US.
- **Replies** on stdout, as frames: little endian int32 kind, int32 count, then the payload.
  - `READY` 1: JSON, sent once after load.
  - `AUDIO` 2: `count` float32 samples, 24 kHz mono, one frame per part.
  - `END` 3: JSON `{"generators": [...]}` after the last part.
  - `ERROR` 4: JSON `{"error": ...}`, including a part that ran another generator than requested.
  - Logs go to stderr.
- **Warm up:** send one short request before the first visible one. The first synthesis in a process loads the Core ML models (about 1 s); the first ever after a build compiles them for the ANE (10 to 12 s once).
- **Text:** the SDK runs its own Misaki G2P on raw text; `render.ts` splits with speak-easy `planSegments` and applies `trimSilence` to each part.

## Stream policy

- Segments: speak-easy `planSegments`, all submitted in one worker request; one `KokoroTTS.synthesize` per segment (Misaki G2P, the SDK's chunking past 15 s); speak-easy `trimSilence` on each. Each yield is one completed segment waveform.
- Full mode: one `synthesize` call on the whole text.
- Cancellation: between segments; an active Core ML chain runs to completion.
- Load (`process.loadMs`): worker spawn plus `KokoroTTS.load` (manifest, digests, vocab). Core ML models load on the first synthesis, so cold first audio carries them.
- Frontend: the SDK's MisakiSwift, not speak-easy's, so pronunciation can differ from `kokoro-heart`.
