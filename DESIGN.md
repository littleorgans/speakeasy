# speak-easy — design

speak-easy is a cascaded speech-to-speech engine: microphone audio in, spoken
reply out. The cascade is STT → LLM → TTS, wired together by a half-duplex
conversation loop. This document is the durable record of the approved design so
it survives across sessions.

## Monorepo layout

A pnpm workspace with three packages, each owning one stage of the cascade plus
the loop that drives them.

```
speak-easy/
  packages/
    speech-io/      @speakeasy/speech-io   STT engines, TTS, rewrite, bench, demos
    llm/            @speakeasy/llm          ChatModel contract + Cerebras adapter   (later)
    convo-engine/   @speakeasy/convo-engine half-duplex VAD conversation loop        (later)
  models/           downloaded ONNX weights          (gitignored, repo root)
  corpus/           labeled WER recordings           (gitignored, repo root)
  results/          bench + sweep output             (gitignored, repo root)
```

- **speech-io** — everything audio↔text: STT engines (sherpa streaming zipformer,
  moonshine), the `withRewrite` post-decode decorator, the TTS streaming
  synthesizer, the WER bench, and the terminal demos. This is today's code.
- **llm** — a provider-agnostic `ChatModel` contract with a Cerebras adapter
  first. Text in, token stream out. No audio concerns.
- **convo-engine** — the half-duplex state machine that owns the microphone,
  runs VAD endpointing, calls STT → LLM → TTS, and plays the reply. Depends on
  both packages above; neither depends on it.

Assets (`models/`, `corpus/`, `results/`) stay at the repo root and are
gitignored. All runtime path resolution is `process.cwd()`-relative, so commands
run from the repo root regardless of which package owns the code.

## Stage contracts

The contracts are the seams between packages. Everything host-specific or
engine-specific stays behind them.

### VoiceToText / STTSession — exists, `packages/speech-io/src/contract.ts`, unchanged

```ts
interface VoiceToText { open(config?: STTConfig): Promise<STTSession>; }

interface STTSession extends EventEmitter {
  pushAudio(frame: Float32Array): void;  // 16kHz mono Float32 PCM
  flush(): void;
  reset(): void;
  end(): Promise<void>;
  // events: "partial", "final", "endpoint", "error"
}
```

The core consumes audio frames; it does not own the microphone. Headline metric:
`endpoint → final` gap, target < 200ms.

### TextToSpeech — new, `speech-io`, formalizes `src/tts/stream.ts`

The streaming synthesizer already exists as `streamSpeech(text)`; the contract
promotes it to a first-class, session-shaped stage that mirrors `VoiceToText`.

```ts
interface TextToSpeech { open(config?: TTSConfig): Promise<TTSSession>; }

interface TTSSession {
  speak(textStream: AsyncIterable<string>): AsyncIterable<AudioSegment>;
}
```

`speak` consumes a stream of text (LLM tokens, sentence-batched by
`planSegments`) and yields `AudioSegment`s as each is synthesized. This is what
`streamSpeech` does today — split into speakable sentences, carve a short first
chunk for low time-to-first-audio, run a 1-deep synth pipeline so segment *i+1*
synthesizes while the consumer plays segment *i*. `AudioSegment` carries the
existing `SpeechSegment` shape: `samples: Float32Array`, `sampleRate`, timing.

### ChatModel — new, `packages/llm`

```ts
interface ChatModel { stream(messages: Message[]): AsyncIterable<string>; }
```

Provider-agnostic token streaming. Cerebras first: OpenAI-compatible SSE. The
API key is read **only** from `process.env.CEREBRAS_API_KEY` (via a gitignored
`.env`), never logged, never committed.

## convo-engine v1

A half-duplex VAD state machine:

```
listening → thinking → speaking → listening
```

- **listening** — mic open, frames pushed to STT. sherpa eager endpointing plus
  VAD detect end-of-turn. On endpoint: capture the final transcript.
- **thinking** — final transcript → `ChatModel.stream(messages)`. Mic is gated.
- **speaking** — LLM token stream → sentence splitter (`planSegments`) →
  `TextToSpeech.speak` → one continuous ffplay PCM sink. Mic stays gated so the
  engine never hears itself.
- back to **listening** when playback drains.

Mic capture and sherpa eager endpointing are reused from speech-io. The LLM
token stream feeds the sentence splitter that already backs streaming TTS.

**Barge-in is explicitly v2**, slotted at the state boundaries (interrupt
`speaking`, flush the sink, return to `listening`). v1 does not listen while
speaking.

### Per-turn latency instrumentation

Every turn records:

- `endpoint → STT-final`
- `→ first LLM token`
- `→ first audio`

**Headline metric: end-of-user-speech to first spoken word.** The v1 deliverable
is a terminal app.

## Adapter roadmap (research-verified)

- **STT stays pure-TS, in-process.** sherpa-onnx's model zoo ships Parakeet TDT
  int8 ONNX (v2 English ~6.05% WER with punctuation/casing; v3 25-language),
  loadable via `OfflineRecognizer` with `modelType: 'nemo_transducer'` in
  `sherpa-onnx-node` 1.13.3 — **offline only**. So the pattern is: streaming
  zipformer (kroko) for live partials, plus a single Parakeet decode on the
  VAD-closed segment for the final. No sidecar.
- **TTS.** Qwen3-TTS open weights (released 2026-01-22, Apache-2.0, 12Hz,
  0.6B/1.7B) run on Apple M-series **only** via Python `mlx-audio`, so they enter
  behind `TextToSpeech` as a **Python-sidecar adapter**. sherpa kokoro/piper
  remain the in-process default.
- **LLM.** Cerebras hosted API only.

## Constraints

- Files < 700 lines, functions < 150 lines. Strict DRY. Match existing
  conventions.
- API keys only from `process.env`, sourced from a gitignored `.env`.
