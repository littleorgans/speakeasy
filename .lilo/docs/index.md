# speak-easy system map
<!-- lilo-page kind=index schema=1 -->
<!-- lilo-covers package.json pnpm-workspace.yaml packages/*/package.json packages/speech-io/src/contract.ts packages/speech-io/src/bench/config.ts packages/speech-io/src/bench/run.ts packages/speech-io/src/engines/sherpa-models.ts packages/speech-io/src/engines/hotwords.ts packages/speech-io/src/rewrite/decorator.ts packages/speech-io/src/rewrite/numbers.ts packages/speech-io/src/tts/synth.ts packages/speech-io/src/tts/stream.ts packages/speech-io/src/tts/player.ts packages/llm/src/contract.ts packages/convo-engine/src/state.ts packages/convo-engine/src/loop.ts packages/convo-engine/src/responder/contract.ts packages/browser-voice/src/server.ts -->
<!-- lilo-verified digest=b88790bbfd4dfdb1 commit=0251c0defed3 date=2026-10-23 -->

> A local first, cascaded speech to speech engine: microphone audio in, spoken reply out, with sub 200 ms speech recognition finalization as the headline goal.

## Purpose and scope

speak-easy turns a spoken user turn into a spoken assistant reply. The cascade is speech to text, then a streaming LLM, then text to speech, driven by a half duplex conversation loop (`code:package.json`). Speech recognition and the default voices run in process on CPU through sherpa-onnx; the LLM and optional voices are hosted providers. The product driver is a conversational loop for the littleorgans director domain: short, proper noun heavy voice commands (claimed by S1, unverified).

## Responsibilities

- `@speakeasy/speech-io`: streaming STT engines, the post decode rewrite, streaming TTS, audio playback, the WER bench and terminal demos (`code:packages/speech-io/src/contract.ts#VoiceToText`).
- `@speakeasy/llm`: provider agnostic streaming chat over OpenAI compatible APIs (`code:packages/llm/src/contract.ts#ChatModel`).
- `@speakeasy/convo-engine`: the half duplex loop, VAD, turn metrics and the cascade sweep; depends on speech-io and llm, and neither depends on it (`code:packages/convo-engine/package.json`).
- `@speakeasy/browser-voice`: a local web server plus browser microphone, playback and conversation UI over convo-engine (`code:packages/browser-voice/src/server.ts`).

## Reading path

Start at the STT contract (`code:packages/speech-io/src/contract.ts`), then the conversation state table (`code:packages/convo-engine/src/state.ts`), then the responder seam (`code:packages/convo-engine/src/responder/contract.ts`). Gates live in [verification gates](verification-gates.md).

## Vocabulary

- **Cascade**: STT, then LLM, then TTS as separate engines; `CascadeResponder` (`code:packages/convo-engine/src/responder/cascade.ts#CascadeResponder`).
- **Fused responder**: one speech model takes the transcript and emits reply audio and text directly, today OpenAI Realtime (`code:packages/convo-engine/src/responder/openai-realtime.ts`).
- **VoiceResponder**: the respond and speak seam the loop drives without knowing which shape sits behind it (`code:packages/convo-engine/src/responder/contract.ts#VoiceResponder`).
- **Partial / final**: an STT hypothesis still being revised versus the committed text of an utterance (`code:packages/speech-io/src/contract.ts#FinalEvent`).
- **Endpoint**: the STT signal that the user stopped speaking; modes `eager`, `turn-aware`, `manual` (`code:packages/speech-io/src/contract.ts#EndpointMode`).
- **Flush to final**: latency from end of speech (or push to talk release) to the final event; the bench's headline metric (`code:packages/speech-io/src/bench/config.ts#PASS_THRESHOLD_MS`).
- **TTFA**: time to first audio, from request to the first audible reply sound (claimed by S2, unverified).
- **RTF**: real time factor, synthesis time over audio duration; below 1 is faster than real time, and a voice must stay well below it to stream (claimed by S3, unverified).
- **Knee**: in an endpoint sweep, the config with the shortest trailing silence among those whose every run is transcript correct and finalizes after speech end, ties broken by median speech end to final. It ignores the 200 ms threshold, so a knee can still FAIL; `none` when no config qualifies (`code:packages/speech-io/src/bench/run.ts#selectBestSummary`, `code:packages/speech-io/src/bench/run.ts#compareEndpointSummaries`).
- **Barge in**: user speech during playback cuts the assistant off; opt in through the loop's `barge` option (`code:packages/convo-engine/src/loop.ts`).

## Rules

- Run every command from the repo root: model, results and hotword paths resolve from `process.cwd()` (`code:packages/speech-io/src/engines/sherpa-models.ts#SHERPA_ROOT`).
- Never pass an `onProgress` callback to sherpa TTS `generateAsync`: in sherpa-onnx-node 1.13.3 it aborts the process with a native use after free. Streaming is an app level sentence pipeline instead (`code:packages/speech-io/src/tts/synth.ts`, `code:packages/speech-io/src/tts/stream.ts#streamSpeech`).
- TTS runs on the `cpu` provider; CoreML aborts the process on kokoro (`code:packages/speech-io/src/tts/synth.ts`; CoreML failure claimed by S2, unverified).
- The STT default is `en-kroko-2025-08-06`, chosen for lowest WER and latency on the command corpus (`code:packages/speech-io/src/engines/sherpa-models.ts#DEFAULT_SHERPA_MODEL`).
- Bench PASS needs a median under 200 ms plus correctness, and the median differs by mode. Push to talk: warm runs' median flush to final, every run transcript correct (`code:packages/speech-io/src/bench/run.ts#runPttSummary`). Endpoint sweep: all runs' median speech end to final, every run transcript correct and finalized after speech end (`code:packages/speech-io/src/bench/run.ts#runEngineSummary`).
- Benchmarks judge end of speech with a VAD or an RMS window hangover, never a raw amplitude threshold, and never label a fast but incorrect config as the answer (claimed by S4, unverified).
- Post decode rewrite touches final events only; partials pass through to avoid flicker. The number stage defaults to `off` (`code:packages/speech-io/src/rewrite/decorator.ts#withRewrite`, `code:packages/speech-io/src/rewrite/numbers.ts`).
- Hotword biasing is inert unless `hotwords.txt` exists at the repo root, and enabling it switches decoding to `modified_beam_search` (`code:packages/speech-io/src/engines/hotwords.ts`). It is a no op on the kroko default, which ships no SentencePiece vocabulary (claimed by S1, unverified).
- Conversation states change only through the `LEGAL_TRANSITIONS` table (`code:packages/convo-engine/src/state.ts#LEGAL_TRANSITIONS`).
- API keys come only from `process.env`; the browser server loads a gitignored `.env` (`code:packages/browser-voice/src/server.ts#loadEnvFile`, `code:.gitignore`).
- The persistent ffplay sink takes `-ch_layout`, never `-ac` (`code:packages/speech-io/src/tts/player.ts`).
- Keep files under 700 lines and functions under 150 lines (claimed by S5, unverified).

## Components

none yet

## Interactions

none yet

## Operations

none yet

## Decisions

none yet

## Gaps

- Component pages for the four packages are unmapped in this quick look.
- `experiments/ruleFsts/` holds an archived FST rewrite experiment, not a production path (claimed by S1, unverified).
- Unread docs are listed in Sources as `ranked, unread`.

## Sources

| id | path | fingerprint | consulted | disposition | claims | watched |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | MODEL-SWEEP.md | ced475b7846f | 2026-10-23 | used | index.md#purpose-and-scope, index.md#rules, index.md#gaps | |
| S2 | TTS-STREAMING.md | 82a52949fecd | 2026-10-23 | used | index.md#vocabulary, index.md#rules | |
| S3 | TTS-SWEEP.md | a387d202a179 | 2026-10-23 | used | index.md#vocabulary | |
| S4 | LESSONS.md | 9ee7e4bc626d | 2026-10-23 | used | index.md#rules | |
| S5 | DESIGN.md | 13dc221670cd | 2026-10-23 | used | index.md#rules | |
| S6 | experiments/ruleFsts/README.md | 9e08466c1ea4 | 2026-10-23 | ranked, unread | | |
| S7 | CASCADE-SWEEP.md | 8ec850494326 | 2026-10-23 | ranked, unread | | |
| S8 | WARROOM.md | b39951a7917d | 2026-10-23 | ranked, unread | | |
