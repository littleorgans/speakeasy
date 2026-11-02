# Candidate: Kokoro v1.0 on native Core ML

Kokoro 82M v1.0 split into Core ML stages, run by FluidAudio's `KokoroAneManager` in a resident Swift worker (`worker/`). Engines `kokoro-coreml-heart` (af_heart) and `kokoro-coreml-michael` (am_michael) use FluidAudio's legacy seven stage route; `kokoro-coreml-v3-heart` selects the v3 hybrid route, which aborts on this OS (below) and stays out of the timed pass. `render.ts` follows [CONTRACT.md](../../CONTRACT.md) through the shared [driver](../NATIVE.md).

## Why FluidAudio, not mattmireles/kokoro-coreml

The shortlist's pick, `mattmireles/kokoro-coreml@b23caa862f0585c2c65187d0fc97e0f8311041dc` with weights `034f7741`, does not build here. This Mac has Command Line Tools only (Swift 6.4, macOS SDK 27.0, no Xcode; `xcodebuild` and `xcrun --find metal` both fail), and its README asks for Xcode 26+. `swift build -c release --product kokoro-say` stops in its MisakiSwift frontend dependency (pinned `3a27756a`), which pulls in mlx-swift, whose `.metal` kernels need Xcode's Metal compiler: `error: unable to spawn process 'metal' (No such file or directory)`. Replacing MisakiSwift with the package's own `KokoroG2P` would mean patching the SDK's text processor and sourcing separate G2P assets, past a bounded probe. Installing Xcode, or `xcodebuild -downloadComponent MetalToolchain`, which itself needs Xcode, would unblock it; that is a machine change for the owner.

The shortlist fallback, FluidAudio, has no package dependencies and no `.metal` sources, and builds with Command Line Tools. Its v3 route uses a CPU/GPU generator adapted from kokoro-coreml.

## v3 blocker

`kokoro-coreml-v3-heart` downloads and loads (`FluidInference/kokoro-82m-coreml@6c0750dd`, base `006395f6`), then aborts the process on its first synthesis, twice of two, warm and cold: `KokoroSourceGenerator.mlmodelc/model.mil ... error: 'mps.expand_dims' op invalid axis: -1182483122` then MPSGraph `failed assertion 'shape for TensorData is not static'`, in the GPU fast generator. FluidAudio's docs record no macOS 27 report and Metal route MPSGraph aborts on iOS 27 (its issues 843 and 889). The legacy route keeps Noise and Tail off the GPU on OS 27 and runs.

## Pins

| What | Pin |
| --- | --- |
| Runtime | `FluidInference/FluidAudio@e396cf780c4abf8e395087467c0585d3b0d17f68` (SwiftPM revision pin in `worker/Package.swift`, resolved in `worker/Package.resolved`); its binary target `NemoTextProcessing.xcframework` v0.3.1 by checksum `5fa8c10d...`. Swift 6.4 (swiftlang-6.4.0.34.1) from Command Line Tools, macOS 27.0.1 |
| Weights, legacy | `FluidInference/kokoro-82m-coreml@2568276a60ff6e7f312afce3dc9e3f07521cb48d` (the commit `main` named on 2026-11-03; the worker pins it with `ModelRegistry.revisionOverrides`, else FluidAudio follows `main`); `ANE/` seven compiled stages, `vocab.json` (sha256 `8d65b0188b77...`), `af_heart.bin` (`d583ccff3cdc...`), `am_michael.bin` (`1d1f21dd8da3...`); 80 MB; `ANE/.fluidaudio-revision` records the commit |
| Frontend assets | Same repository: `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `g2p_vocab.json` (`295ed64b86c2...`), `us_lexicon_cache.json` (`6b36ba313202...`); 11 MB. Fetched by the first (v3) run before the pin was set, so from `main`, which named the same commit that day |
| Weights, v3 | `FluidInference/kokoro-82m-coreml@6c0750dd02ef9d981fb82f147d784a3f171755ee` `ANE-v3/en`, pinned by FluidAudio; 376 MB, removed after the blocker (refetched if `kokoro-coreml-v3-heart` runs) |

All assets sit under `MODELS/kokoro-ane-fluidaudio/` (`directory:` passed to `KokoroAneManager`).

## Licences

- Code: FluidAudio Apache 2.0; NemoTextProcessing (FluidInference/text-processing-rs, prebuilt binary) Apache 2.0 per kokoro-coreml's manifest comment, not checked at its source. The legacy chain derives from laishere/kokoro-coreml, used with its author's permission per FluidAudio's docs.
- Weights: `FluidInference/kokoro-82m-coreml` card licence Apache 2.0; base model hexgrad/Kokoro-82M Apache 2.0. No voice cloning. Full transitive distribution review not done.

## Setup

From the speak-easy repo root. The first run downloads the assets (about 90 MB) and compiles them on device; later loads take about 200 ms.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
(cd experiments/fast-tts/candidates/kokoro-coreml/worker && swift build -c release --scratch-path $TMPDIR/kokoro-ane-worker-build)
export KOKORO_ANE_WORKER=$TMPDIR/kokoro-ane-worker-build/release/KokoroAneWorker
```

## Run

```bash
node experiments/fast-tts/candidates/kokoro-coreml/render.ts --root $MODELS --engine kokoro-coreml-heart \
  --temperature warm --repeats 1 --json $TMPDIR/kokoro-coreml-heart--warm--stream.json --clips $ASSETS/clips
```

Flags match `render-baselines.ts`. Registration in `run-timed.sh`: [NATIVE.md](../NATIVE.md).

## Stream policy

- Segments: speak-easy `planSegments`, imported from `packages/speech-io/src/tts/stream.ts`, all submitted in one worker request. The worker makes one `synthesizeDetailed` call per segment (FluidAudio's English frontend: NeMo normalization, Misaki lexicon, BART G2P for misses; its own split past 510 phonemes) and writes one AUDIO frame per segment. The adapter applies speak-easy `trimSilence` (edge trim, 150 ms gap), as the baselines do. Each yield is one completed segment waveform; nothing is incremental within a segment.
- Cancellation checkpoint: between segments in the worker, which does not yet take a stop request; an active Core ML chain runs to completion.
- Full mode: one `synthesizeDetailed` call on the whole text; one chunk.
- Load (`process.loadMs`): worker spawn plus `initialize` (loads of the compiled stages, G2P assets, voice pack). The first synthesis in a process specializes the models, so cold first audio (about 0.9 to 1.4 s in smoke) is far above warm.
- Compute: FluidAudio requests CPU plus ANE for Albert, PostAlbert, Alignment, Prosody and Vocoder, and CPU only for Noise and Tail on OS 27. Core ML decides actual placement; ANE activity was not observed here.
- Frontend: FluidAudio's, not speak-easy's Piper Plus Misaki mapping that sherpa Kokoro uses, so pronunciation can differ from `kokoro-heart`; the listening page shows it.
