# Candidate: Piper Kristin with libpiper split decoder

The accepted Kristin medium voice, split into an encoder and a decoder, run by libpiper (piper1-gpl C API) in a resident worker (`worker.c`). With the split in place, libpiper runs the encoder once per sentence and then decodes in overlapping chunks, so a sentence's first audio arrives before the sentence is decoded. Engines:

- `piper-split-t4`: whole sentences (speak-easy `splitSentences`); the decoder chunking replaces the baselines' short first chunk, and every sentence keeps its full prosody.
- `piper-split-carved-t4`: the baselines' `planSegments` parts, for a like for like comparison with `piper-kristin-t4`.

`render.ts` follows [CONTRACT.md](../../CONTRACT.md) through the shared [driver](../NATIVE.md).

## Pins

| What | Pin |
| --- | --- |
| Acoustic model | Local `vits-piper-en_US-kristin-medium/en_US-kristin-medium.onnx` (sha256 `f9a5c375c70d...`, 63,531,507 bytes), checked against `rhasspy/piper-voices@e7229f7d33a10e145e109b886d661984201ac3c4` `en/en_US/kristin/medium/en_US-kristin-medium.onnx` (sha256 `5849957f929c...`, 63,531,379 bytes) with the `onnx` package: identical graph (6,423 nodes), initializers (350) and I/O; the local file only adds sherpa's metadata props. The voice config is byte identical (sha256 `5681426d4aea...`) |
| Split (derived) | `piper-kristin-split-e7229f7d/`: `en_US-kristin-medium.encoder.onnx` 57,248,767 bytes, sha256 `380d38540eaf9cb33e6ce43bb6e2458f55da0d19405c5f65852bb7a0e74573e4`; `en_US-kristin-medium.decoder.onnx` 6,670,623 bytes, sha256 `ecf60b852d2e7edd4d51796ec8a00c1560b19ee85b3ff819daf6a82ce6a768cb`; `en_US-kristin-medium.onnx.json` (Kristin's config plus `"phoneme_type": "text"`), sha256 `986bba3f536ab08d92c92e4e2223229eed686eda933d30a57604e981dcea0c24` |
| Runtime | `OHF-Voice/piper1-gpl@10880972db9e0aca101fed350eadf8f49deefe9f` `libpiper/` (reports version 0.0.1), CMake Release; its build downloads `onnxruntime-osx-arm64-1.22.0` (microsoft/onnxruntime release) and builds `espeak-ng@212928b394a96e8fd2096616bfd54e17845c48f6` statically. Split by that commit's `src/piper/split_voice_for_streaming.py` with `onnx==1.19.1` |
| Frontend | speak-easy `createTtsFrontend` (Piper Plus G2P `4fd95c24`, CMU dict `piper-plus-g2p-4fd95c24/`), unchanged |
| Toolchain | Apple clang from Command Line Tools (Swift 6.4 toolchain), CMake 4.4.5 |

## Licences

- Code: libpiper and piper1-gpl GPL 3.0. The worker links libpiper, which links eSpeak NG statically (GPL 3.0 or later) even though this path never calls it. GPL permits commercial use with its distribution obligations; this changes speak-easy's current runtime dependency boundary, so it stays an experiment pending packaging review. onnxruntime MIT.
- Weights: Kristin dedicated to the public domain by its creator (voice-out pip-F7, g2p-S20); dataset LibriVox, public domain. The split graphs are derived from it. CMU dict BSD style, per speak-easy `frontend.ts`.

## Frontend parity

The phonemes are the baselines' own: `render.ts` calls `createTtsFrontend` exactly as `TtsSynth.create` does for Kristin (`tokens.txt`, `lang` "en" from the model registry, so the RP mapping applies, family `vits`), on the clock in the adapter process. Each frontend sentence goes to libpiper as one string of phoneme characters. With `phoneme_type: "text"` libpiper lowercases and NFD normalizes that string, then maps each character through the voice's `phoneme_id_map` with a pad after each id, BOS and pad first and EOS last; eSpeak never runs. `render.ts` refuses any part that lowercasing or NFD would change; none did on the cohort.

Unverified: that this id sequence equals what sherpa builds from the same codepoints (read from libpiper's source only), and therefore equal audio quality. VITS sampling noise makes clips differ run to run, so a byte comparison is not possible; the owner's listening check covers it.

## Setup

From the speak-easy repo root. No `pnpm install` needed: `frontend.ts`, `models.ts` and `stream.ts` load no native addon.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
git -C ~/Dev/research-clones init -q piper1-gpl-10880972 && cd ~/Dev/research-clones/piper1-gpl-10880972 \
  && git fetch -q --depth 1 https://github.com/OHF-Voice/piper1-gpl.git 10880972db9e0aca101fed350eadf8f49deefe9f \
  && git checkout -q FETCH_HEAD && cd -
# Split (once): the split script needs only onnx; it writes next to the path it is given.
uv venv --python 3.13 $TMPDIR/piper-split-env && uv pip install --python $TMPDIR/piper-split-env/bin/python onnx==1.19.1
D=$MODELS/piper-kristin-split-e7229f7d; mkdir -p $D
ln -s ../vits-piper-en_US-kristin-medium/en_US-kristin-medium.onnx $D/en_US-kristin-medium.onnx
$TMPDIR/piper-split-env/bin/python -I ~/Dev/research-clones/piper1-gpl-10880972/src/piper/split_voice_for_streaming.py $D/en_US-kristin-medium.onnx
rm $D/en_US-kristin-medium.onnx
python3 -I -c 'import json,sys; c=json.load(open(sys.argv[1])); c["phoneme_type"]="text"; open(sys.argv[2],"w").write(json.dumps(c, ensure_ascii=False, indent=2)+"\n")' \
  $MODELS/vits-piper-en_US-kristin-medium/en_US-kristin-medium.onnx.json $D/en_US-kristin-medium.onnx.json
# libpiper and the worker:
cmake -S ~/Dev/research-clones/piper1-gpl-10880972/libpiper -B $TMPDIR/libpiper-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$TMPDIR/libpiper-install
cmake --build $TMPDIR/libpiper-build --config Release -j 6 && cmake --install $TMPDIR/libpiper-build --config Release
I=$TMPDIR/libpiper-install
cc -O2 -Wall -Wextra -std=c11 experiments/fast-tts/candidates/piper-split/worker.c -I$I/include -L$I/lib -lpiper -Wl,-rpath,$I/lib -o $I/bin/piper-split-worker
export PIPER_SPLIT_WORKER=$I/bin/piper-split-worker
```

## Run

```bash
node experiments/fast-tts/candidates/piper-split/render.ts --root $MODELS --engine piper-split-t4 \
  --temperature warm --repeats 1 --json $TMPDIR/piper-split-t4--warm--stream.json --clips $ASSETS/clips
```

Flags match `render-baselines.ts`; `stream` is the mode that matters here. Registration in `run-timed.sh`: [NATIVE.md](../NATIVE.md).

## Stream policy

- Parts: `splitSentences` (`piper-split-t4`) or `planSegments` (`piper-split-carved-t4`), from `packages/speech-io/src/tts/stream.ts`. The frontend runs per part, as the baseline pipeline runs it per synth call, and each part is one worker request.
- Chunks: each yield is one libpiper decoder chunk, unchanged. Source constants (`piper.cpp`): first chunk 32 latent frames (8,192 samples, 372 ms at 22.05 kHz), 12 context frames on each side; later chunks are sized from the measured decoder speed so decoding takes at most half the audio still buffered (24 to 256 frames). Upstream states the chunked audio equals whole sentence decoding up to floating point rounding.
- No silence trim: chunks are mid sentence, so `trimSilence` would cut speech; clips keep Kristin's own sentence padding and no 150 ms gaps. Audio durations, and so RTF, are therefore slightly longer than the trimmed baselines for the same speech.
- Cancellation checkpoint: between `piper_synthesize_next` calls in the worker, one decoder chunk apart (the worker does not yet accept a stop request; a host stops by not sending the next part, so the current part finishes).
- Threads: libpiper sets 4 intra op threads here (performance cores, at most 4, `default_intra_op_threads`), so compare with `piper-kristin-t4`.
- Full mode: the same parts drained to the end, one chunk; it equals stream synthesis without early delivery, not a single sherpa call.
- Load (`process.loadMs`): the Piper Plus frontend (CMU dict parse) plus worker spawn and `piper_create` with the encoder and decoder sessions.
- Upstream note: libpiper returns a split voice's last chunk with `PIPER_DONE` and its samples, so the README example loop (`while (piper_synthesize_next(...) != PIPER_DONE)`) drops the last chunk of every sentence. `worker.c` writes it.
