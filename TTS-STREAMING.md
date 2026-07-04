# TTS streaming: native callback root cause + sentence pipeline

Follow-up to TTS-SWEEP.md. Product driver: the littleorgans conversational
loop must start speaking before the full reply is synthesized. The native
streaming callback in sherpa-onnx-node 1.13.3 is fatally broken (root cause
below, with evidence), so streaming ships as an app-level sentence pipeline
(`src/tts/stream.ts`) built on the stable no-callback path. It works: first
audio in ~0.7s instead of ~2.3s on the 3-sentence test paragraph, and
synthesis provably stays ahead of playback.

## Part 1: the native callback OOM

### Root cause (from the addon source)

`scripts/node-addon-api/src/non-streaming-tts.cc` (a symlink to
`harmony-os/SherpaOnnxHar/sherpa_onnx/src/main/cpp/non-streaming-tts.cc`;
the v1.13.3 tag and current master are byte-identical). The generateAsync
path:

1. `TtsGenerateWorker.Execute()` runs on a background thread. The C-level
   progress callback heap-allocates a `TtsCallbackData` (a copy of the chunk
   samples), appends it to the worker's `data_list_`, and queues it to the
   main thread via `tsfn_.NonBlockingCall(data)`.
2. When generation finishes, the AsyncWorker completes on the main loop:
   `OnOK()` resolves the promise, then the framework destroys the worker.
   `~TtsGenerateWorker()` deletes **every** `TtsCallbackData` in
   `data_list_`, including items still sitting in the TSFN queue,
   undelivered.
3. The TSFN drain and the worker-complete callback are two independent
   main-loop events with no ordering guarantee. When a delivery loses the
   race, `InvokeJsCallback` reads the freed struct, `data->samples.size()`
   returns garbage, and `Napi::ArrayBuffer::New(env, garbage)` aborts the
   process: `FATAL ERROR: v8::ArrayBuffer::New Allocation failed`. This is a
   native abort, not a catchable JS exception.

The final sentence's callback fires immediately before `Execute()` returns,
so the last chunk is almost always still queued when the worker dies. That
makes the crash frequent but nondeterministic, which the matrix confirms.

No JS-side code can close the window: by the time the resolved promise is
observable from JS, the destructor has already run.

### Isolation matrix (evidence)

One variable per axis, each cell in a fresh process, 5 runs per cell
(scratchpad `oom-cell.mjs` / `oom-matrix2.mjs`; a cell passes only if it
outlives a 1.5s post-generation settle window). piper-amy, 2-sentence text,
2 threads.

| Cell | Crashes |
|---|---|
| 3 calls, no callback, ext buffer | 0/5 |
| 3 calls, no callback, no ext buffer | 0/5 |
| 1 call, callback retains samples | 2/5 |
| 1 call, callback copies samples | 0/5 |
| 3 calls, callback retains | 2/5 |
| 3 calls, callback copies | 2/5 |
| 3 calls, retains, no ext buffer | 3/5 |
| 3 calls, copies, no ext buffer | 4/5 |
| 3 calls, copies, forced GC between calls | 5/5 |
| 3 calls, retains, forced GC between calls | 5/5 |
| 1 call per process, x3 (round 1) | crashed |

Conclusions:

- **No callback: 0 crashes in 10 runs.** The only stable configuration.
- **Any callback: 2/5 to 5/5.** Copy-vs-retain in the JS callback is
  irrelevant (the freed read happens in native code before JS is invoked).
  `enableExternalBuffer` is irrelevant (it only affects the final result
  buffer). Forced GC is irrelevant-to-adverse. Even one call per process is
  a coin flip, so process sacrifice is not a workaround either.
- Verdicts flip between identical runs (round 1 vs round 2 disagreed on 3
  cells), consistent with a scheduling race, not a deterministic leak.

### Upstream verdict

- 1.13.3 (published 2026-06-15) is the **latest** version on npm; there is
  nothing to upgrade to.
- The TTS addon source on master is **byte-identical** to v1.13.3, so the
  next release will not fix it either.
- No tracking issue found in k2-fsa/sherpa-onnx for this crash; the closest
  is #2574 (feature question: does node support streaming TTS). The bug
  appears unreported. Worth filing upstream with the root cause above; the
  fix is small (drain or invalidate the TSFN queue before destroying the
  worker, e.g. share ownership of `TtsCallbackData` between queue and list).

### Callback granularity

From `sherpa-onnx/csrc/offline-tts-vits-impl.h` (v1.13.3): the C callback
fires once per batch of `maxNumSentences` sentences (config default 1, so
per sentence), with `progress = batch / num_batches`. If the text has no
more sentences than `maxNumSentences`, the callback fires exactly once with
the entire audio. There is no frame- or chunk-level callback in the C API
for VITS-family models; per-sentence is the finest the engine offers, which
means the native callback (once fixed upstream) would deliver the same
granularity our sentence pipeline already achieves today.

## Part 2: sentence-pipelined streaming (shipped)

`src/tts/stream.ts`:

- `splitSentences(text)`: terminator-aware splitter (unit-tested).
- `streamSpeech(text, {model, speed})`: AsyncGenerator that synthesizes
  sentence-by-sentence with the stable no-callback `generateAsync` and
  yields each segment as it completes. Pipeline depth 1: sentence i+1
  synthesizes on the native worker thread while the consumer plays sentence
  i.
- `buildTimingReport(segments)`: pure gapless-playback prover. Segment i's
  play slot starts at `readyAt(segment 0) + sum(audio 0..i-1)`; the stream
  is ahead of playback iff every segment is ready before its slot.

`src/tts/stream-demo.ts` makes it audible: feeds each segment into one
persistent audio sink as it arrives (see Part 3 for the sink and the TTFA
work) and prints the timing report. Segments are also written to
`results/tts/stream-<i>.wav`.

### Measured results (piper-amy, 2 threads, Apple Silicon)

3-sentence build-status paragraph, live afplay run:

| Segment | Ready at | Synth | Audio | Play slot | Margin |
|---|---|---|---|---|---|
| 0 "The build finished on the staging channel." | 743.6ms | 743.4ms | 2384.0ms | 743.6ms | 0ms (defines TTFA) |
| 1 "Two checks are still running." | 1299.8ms | 556.2ms | 1808.0ms | 3127.6ms | +1827.8ms |
| 2 "Say the word and I will promote it." | 1789.9ms | 490.0ms | 2448.0ms | 4935.6ms | +3145.7ms |

- **Time-to-first-audio: 743.6ms**, vs 2277.0ms synthesizing the same
  paragraph monolithically (TTS-SWEEP.md): a 3.1x improvement, and the gap
  widens with reply length.
- **Ahead-of-playback: yes.** Every margin is positive and grows (piper-amy
  RTF ~0.35 means each played second buys ~2.9 seconds of synth headroom).
  No underruns; playback is gapless up to afplay process-spawn overhead
  (tens of ms, imperceptible in the demo; an in-process player would
  eliminate it).

### Hear it

```
node src/tts/stream-demo.ts                          # piper-amy, warm + pre-opened sink
node src/tts/stream-demo.ts --model kokoro-v0.19     # kokoro
node src/tts/stream-demo.ts --text "Your reply here. Second sentence."
node src/tts/stream-demo.ts --cold                   # fresh-process TTFA (no warm/pre-open)
node src/tts/stream-demo.ts --no-play                # timings + wavs only
```

## Part 3: gapless playback + TTFA

Two follow-up passes turned the raw pipeline into something that sounds like
speech and starts fast.

### Gapless playback + silence trim

The original demo chained one `afplay` per sentence. Measured head to head on
identical audio, that spawn adds **~1640ms of dead air per sentence boundary**
(afplay re-primes CoreAudio every call); the model's own end padding
(piper ~190ms, kokoro ~300ms per boundary) sat underneath it. Fix, both causes:

- **One persistent `ffplay`** reading raw f32le PCM from stdin
  (`-f f32le -ar <rate> -ch_layout mono -i pipe:0`; note `-ch_layout`, ffplay
  rejects the ffmpeg `-ac` flag). Segment i plays while i+1 synthesizes;
  RTF < 1 so the pipe never underruns. afplay remains a fallback (keeps its
  gap). Inter-sentence gap: ~1.8-2.0s → ~0.15-0.18s.
- **`trimSilence`** (in `stream.ts`) strips each segment's ragged leading/
  trailing near-silence and appends one controlled 150ms gap, so sentences
  carry a natural rhythm instead of variable dead air.

### TTFA budget and the three wins

TTFA budget, isolated per component (Apple Silicon, cpu):

| Component | kokoro-v0.19 (4thr) | piper-amy (2thr) |
|---|---|---|
| (a) addon + model load | ~530ms | ~640ms |
| (b) onnxruntime cold-start warmup tax | ~29ms | ~0ms |
| (c) ffplay audio-device open | ~500ms | ~500ms |
| (d) warm synth, full first sentence | ~835ms | ~98ms |
| (d) warm synth, split first chunk | ~583ms | ~58ms |

The onnxruntime warmup tax is negligible; kokoro's cost is its inherent fixed
per-synth floor (~460ms even for one word), not a cold-start artifact. So the
levers are (a)/(c) as one-time startup costs and (d) via a shorter first chunk:

1. **Warmup + resident reuse** (`TtsSynth.warmup()`): one throwaway synth at
   startup so the first real request runs warm; `TtsSynth` loads once and is
   reused across requests.
2. **Pre-opened sink**: the demo spawns ffplay and writes a 150ms silence
   primer during startup, so CoreAudio's ~500ms open finishes off the request
   path. Biggest single win for piper (whose synth is smaller than the open).
3. **Aggressive first-chunk split** (`planSegments`, `FIRST_CHUNK_MAX_WORDS`):
   the opening chunk is cut at the earliest of the first clause boundary or a
   word cap so first audio lands ASAP; later chunks stay whole sentences for
   prosody. Isolated saving: kokoro −252ms, piper −39ms.

Honest audible TTFA (from request; model load is a one-time startup, as in a
resident Electron main), end-to-end from the demo:

| Model | cold (fresh process) | warm + split (shipped) |
|---|---|---|
| kokoro-v0.19 | ~1600ms | ~1000ms |
| piper-amy | ~590ms | ~72ms |

Warm-without-split sits between (add back the split saving above). Kokoro's
end-to-end warm number runs above its isolated synth because heavy synth
contends with live playback. Ahead-of-playback stays **yes** in every case:
splitting the opener shortens segment 0's audio but its tail still lands before
the head finishes.

The metric itself was corrected: the demo now reports **audible** first sound
(first chunk reaching an already-open device), not the old synth-ready
`readyAtMs`, which silently excluded model load and device-open.

### CoreML probe (rejected)

sherpa-onnx-node 1.13.3 does **not** support the CoreML execution provider for
kokoro: `provider:"coreml"` throws an uncatchable
`Ort::Exception: model_builder.cc:851 ... Unable to get shape for output:
/Squeeze_output_0` and aborts the process (kokoro's dynamic output shapes are
not registrable by the CoreML EP). cpu stays the only viable provider; the
default is unchanged.

## Recommendation

Ship sentence pipelining as the streaming mechanism: it is stable (built
entirely on the 0/10-crash path), achieves sub-second first audio with
piper-amy, and per-sentence is already the engine's native granularity, so
the broken callback offers no finer streaming even when fixed. Report the
UAF upstream; revisit `onProgress` only after a release whose
`non-streaming-tts.cc` changes the worker/TSFN lifetime handling, and even
then it buys convenience, not latency.
