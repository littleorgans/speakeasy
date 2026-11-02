# Candidate: Supertonic 3 ONNX

Supertonic 3 (Supertone), four ONNX graphs run by `onnxruntime-node` on the CPU, in process, through the upstream Node example's `helper.js`. Engines `supertonic3-f1` and `supertonic3-m1` (2 threads) and `supertonic3-f1-t4` (4 threads). `render.ts` follows [CONTRACT.md](../../CONTRACT.md) through the shared [driver](../NATIVE.md).

## Pins

| What | Pin |
| --- | --- |
| Weights | `Supertone/supertonic-3@3cadd1ee6394adea1bd021217a0e650ede09a323`: `onnx/duration_predictor.onnx` (3,700,147 bytes), `onnx/text_encoder.onnx` (36,416,150), `onnx/vector_estimator.onnx` (256,534,781), `onnx/vocoder.onnx` (101,424,195), `onnx/tts.json`, `onnx/unicode_indexer.json`, `voice_styles/F1.json`, `voice_styles/M1.json`, `LICENSE`, `README.md`, `config.json`. Graphs total 398,075,273 bytes; every LFS file checked against its HF sha256 after download |
| Helper | `supertone-oss-archive/supertonic@1e9799e964ea4c0dad7cde993b65c3c813a7b373`, `nodejs/helper.js` and `nodejs/package.json`, unmodified |
| Runtime | `onnxruntime-node@1.31.1` (upstream range `^1.19.2`, resolved 2026-11-03), `onnxruntime-common@1.31.1`; Node 25.9.0 |
| Settings | 8 flow steps (upstream example default), speed 1.05 (upstream default), language `en`, 44.1 kHz mono |

The upstream Node README names `supertone-oss-archive/supertonic-3@aafc6e32`; this probe uses the shortlist pin above.

## Licences

- Code: Supertonic sample code MIT (Supertone Inc.); onnxruntime-node MIT.
- Weights: BigScience Open RAIL-M, with use restrictions (its Attachment A) and notices to retain. Not an unrestricted licence; voice-out onx-F5 records the distinction. No voice cloning; F1 and M1 are built in styles.

## Setup

From the speak-easy repo root. Nothing here needs `pnpm install`: the adapter imports only `stream.ts`, which loads no native addon.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
REV=3cadd1ee6394adea1bd021217a0e650ede09a323
D=$MODELS/supertonic-3-3cadd1ee; mkdir -p $D/onnx $D/voice_styles
for f in LICENSE README.md config.json onnx/duration_predictor.onnx onnx/text_encoder.onnx onnx/tts.json \
  onnx/unicode_indexer.json onnx/vector_estimator.onnx onnx/vocoder.onnx voice_styles/F1.json voice_styles/M1.json; do
  curl -sfL -o $D/$f https://huggingface.co/Supertone/supertonic-3/resolve/$REV/$f
done
git -C ~/Dev/research-clones init -q supertonic-1e9799e9 && cd ~/Dev/research-clones/supertonic-1e9799e9 \
  && git fetch -q --depth 1 https://github.com/supertone-oss-archive/supertonic.git 1e9799e964ea4c0dad7cde993b65c3c813a7b373 \
  && git checkout -q FETCH_HEAD && cd -
export SUPERTONIC_NODE=$TMPDIR/supertonic-node; mkdir -p $SUPERTONIC_NODE
cp ~/Dev/research-clones/supertonic-1e9799e9/nodejs/{helper.js,package.json} $SUPERTONIC_NODE/
(cd $SUPERTONIC_NODE && npm install --no-audit --no-fund onnxruntime-node@1.31.1 fft.js@4.0.4 js-yaml@4.3.2)
```

## Run

```bash
node experiments/fast-tts/candidates/supertonic3/render.ts --root $MODELS --engine supertonic3-f1 \
  --temperature warm --repeats 1 --json $TMPDIR/supertonic3-f1--warm--stream.json --clips $ASSETS/clips
```

Flags match `render-baselines.ts`. Registration in `run-timed.sh`: [NATIVE.md](../NATIVE.md).

## Stream policy

- Segments: speak-easy `planSegments`, imported from `packages/speech-io/src/tts/stream.ts` (first chunk is the first clause or 4 words of sentence 1, whichever is shorter; then whole sentences).
- Each segment is one upstream `TextToSpeech._infer` call: duration predictor, text encoder, 8 vector estimator steps, vocoder. The waveform is cut to the predicted duration, then passed through speak-easy `trimSilence` (edge trim, 150 ms gap), as the baselines do. Each yield is one completed segment waveform; there is no incremental waveform decoder on this path.
- Cancellation checkpoint: between segments; an active `_infer` runs its eight steps to the end.
- Full mode: upstream `TextToSpeech.call` on the whole text (its own sentence grouping up to 300 characters, 0.3 s silence joins), cut to the summed duration as `example_onnx.js` does; one chunk.
- Load (`process.loadMs`): four `InferenceSession.create` calls plus the unicode indexer and the voice style. Threads: `render.ts` wraps `InferenceSession.create` on the same module instance `helper.js` imports, setting `intraOpNumThreads` and `interOpNumThreads: 1`; `helper.js` itself passes empty options.
- Noise: the initial latent comes from `Math.random`, so repeats differ slightly.

## Findings so far (untimed)

- Almost all time is the vector estimator: one untimed profile of `status` (3.26 s of audio) gave 1.98 s of its 2.13 s in the eight vector estimator runs at onnxruntime's default threads, under load average 10 to 13. The JavaScript array handling in `helper.js` cost 10 to 30 ms.
- Its cost per call barely depends on text length, so short first segments do not buy the drop in first audio that they give Piper.
- Threads, same profile, load average 13 to 19: 2 threads about 1.0 s, 4 threads 1.1 to 1.4 s, 8 threads 1.5 to 2.1 s, default 2.1 to 2.8 s. Hence 2 threads by default; the timed pass decides.
