# speak-easy system map
<!-- lilo-page kind=index schema=1 -->
<!-- lilo-covers package.json pnpm-workspace.yaml packages/*/package.json packages/speech-io/src/bench/run.ts packages/speech-io/src/engines/sherpa-models.ts packages/speech-io/src/engines/sherpa-binding.ts packages/speech-io/src/engines/ort-local.ts packages/speech-io/src/engines/hotwords.ts packages/speech-io/src/engines/sherpa-offline.ts packages/speech-io/src/rewrite/decorator.ts packages/speech-io/src/rewrite/numbers.ts packages/speech-io/src/tts/synth.ts packages/speech-io/src/tts/stream.ts packages/speech-io/src/tts/player.ts packages/convo-engine/src/state.ts packages/browser-voice/src/server.ts packages/browser-voice/src/server-options.ts packages/browser-voice/src/names.ts -->
<!-- lilo-verified digest=26417e894d1db0cc commit=20554d4b6061 date=2026-11-08 -->

> A local first, cascaded speech to speech engine; the headline goal is sub 200 ms speech recognition finalization.

## Purpose and scope

speak-easy answers a spoken turn with a spoken reply: speech to text, a streaming LLM, then text to speech, driven by a half duplex loop (`code:package.json`). STT and the default voices run in process on CPU via sherpa-onnx; bench only onnxruntime-node STT sits behind `./ort` (`code:packages/speech-io/src/ort.ts`). The LLM is hosted; optional voices are hosted providers or local worker runtimes.

## Responsibilities

- `@speakeasy/speech-io`: STT engines, post decode rewrite, streaming TTS, playback, WER bench and demos (`code:packages/speech-io/src/contract.ts#VoiceToText`).
- `@speakeasy/llm`: provider agnostic streaming chat over OpenAI compatible APIs (`code:packages/llm/src/contract.ts#ChatModel`).
- `@speakeasy/convo-engine`: the half duplex loop, VAD, turn metrics and cascade sweep; depends on speech-io and llm, never the reverse (`code:packages/convo-engine/package.json`).
- `@speakeasy/browser-voice`: the local server for the conversation room and road test page, and worker runtimes for models outside Node (`code:packages/browser-voice/src/server.ts`, `code:packages/browser-voice/runtime/worker.py`).

## Reading path

Read the STT contract (`code:packages/speech-io/src/contract.ts`), then the state table (`code:packages/convo-engine/src/state.ts`), then the responder seam (`code:packages/convo-engine/src/responder/contract.ts`). Gates: [verification gates](verification-gates.md).

## Vocabulary

- **Cascade**: STT, LLM and TTS as separate engines (`code:packages/convo-engine/src/responder/cascade.ts#CascadeResponder`).
- **Fused responder**: one speech model answers the transcript with audio and text, today OpenAI Realtime (`code:packages/convo-engine/src/responder/openai-realtime.ts`).
- **VoiceResponder**: the respond and speak seam the loop drives, blind to its implementation (`code:packages/convo-engine/src/responder/contract.ts#VoiceResponder`).
- **Partial / final**: a revisable STT hypothesis vs an utterance's committed text (`code:packages/speech-io/src/contract.ts#FinalEvent`).
- **Endpoint**: STT's signal that the user stopped; modes `eager`, `turn-aware`, `manual` (`code:packages/speech-io/src/contract.ts#EndpointMode`).
- **Flush to final**: end of speech or push to talk release to the final; the bench headline metric (`code:packages/speech-io/src/bench/config.ts#PASS_THRESHOLD_MS`).
- **TTFA**: request to first audible reply sound (claimed by S2, unverified).
- **RTF**: real time factor, synthesis time over audio time; a voice streams only well below 1 (claimed by S3, unverified).
- **Knee**: the endpoint sweep config with the shortest trailing silence whose runs all are correct and final after speech end, ties to the lower median; it ignores the 200 ms threshold, so can FAIL; `none` if none qualifies (`code:packages/speech-io/src/bench/run.ts#selectBestSummary`).
- **Barge in**: user speech cuts playback; opt in with the loop's `barge` option (`code:packages/convo-engine/src/loop.ts`).
- **Voice settings**: a local voice's per reading controls; the page sends values off default, the host echoes those applied (`code:packages/speech-io/src/tts/catalogue.ts#localOutputSettings`, `code:packages/browser-voice/public/voice-settings.js#VoiceSettings`).
- **Reference**: the recording and transcript a voice copy model conditions on: a pinned synthetic pick, or the owner's recording, in server memory only (`code:packages/speech-io/src/tts/synth.ts#VoiceReference`, `code:packages/browser-voice/src/voice-references.ts#VoiceReferenceStore`).

## Rules

- Run commands from the repo root: model, results and hotword paths resolve from `process.cwd()` (`code:packages/speech-io/src/engines/sherpa-binding.ts#SHERPA_ROOT`, `code:packages/speech-io/src/engines/ort-local.ts#ORT_ROOT`).
- Never pass `onProgress` to sherpa TTS `generateAsync`: sherpa-onnx-node 1.13.3 aborted the process (use after free); 2.x is untested. Stream via the sentence pipeline (`code:packages/speech-io/src/tts/synth.ts`, `code:packages/speech-io/src/tts/stream.ts#streamSpeech`).
- Sherpa TTS runs on the `cpu` provider; CoreML aborts kokoro (`code:packages/speech-io/src/tts/synth.ts`; CoreML claimed by S2, unverified).
- The STT default is `en-kroko-2025-08-06`, for lowest WER and latency on the command corpus (`code:packages/speech-io/src/engines/sherpa-models.ts#DEFAULT_SHERPA_MODEL`).
- Bench PASS needs every transcript correct and a median under 200 ms: warm runs' flush to final for push to talk (`code:packages/speech-io/src/bench/run.ts#runPttSummary`), all runs' speech end to final, each after speech end, for an endpoint sweep (`code:packages/speech-io/src/bench/run.ts#runEngineSummary`).
- Benchmarks detect speech end by VAD or RMS window hangover, never raw amplitude, and never crown a fast, incorrect config (claimed by S4, unverified).
- Post decode rewrite edits finals only; its number stage defaults to `off` (`code:packages/speech-io/src/rewrite/decorator.ts#withRewrite`, `code:packages/speech-io/src/rewrite/numbers.ts`).
- A repo root `hotwords.txt` switches CPU decoding from greedy to `modified_beam_search` (`code:packages/speech-io/src/engines/hotwords.ts#loadHotwords`). Kroko lacks BPE data and cannot bias (claimed by S1, unverified). Offline, only Parakeet Ultra takes a list, checked against its export vocabulary; the others reject one (`code:packages/speech-io/src/engines/sherpa-offline.ts#SherpaOfflineEngine`). The browser server ignores it; each take gets one snapshot of its names list (`--names <file>`, default `models/road-test-names.json`; `code:packages/browser-voice/src/names.ts#NamesStore`).
- Conversation states change only through the `LEGAL_TRANSITIONS` table (`code:packages/convo-engine/src/state.ts#LEGAL_TRANSITIONS`).
- API keys come only from `process.env`; the browser server loads the gitignored `.env` (`code:packages/browser-voice/src/server.ts#loadLocalEnv`, `code:.gitignore`).
- The persistent ffplay sink takes `-ch_layout`, never `-ac` (`code:packages/speech-io/src/tts/player.ts`).
- Files stay under 700 lines, functions under 150 (claimed by S5, unverified).

## Components

none yet

## Interactions

none yet

## Operations

- [Road test server](operations/road-test-server.md): start detached, check, stop.

## Decisions

none yet

## Gaps

- Unmapped: a component page per package.
- `experiments/ruleFsts/`: an archived FST rewrite experiment, not in production (claimed by S1, unverified).

## Sources

| id | path | fingerprint | consulted | disposition | claims | watched |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | MODEL-SWEEP.md | ced475b7846f | 2026-10-23 | used | index.md#rules, index.md#gaps | |
| S2 | TTS-STREAMING.md | 82a52949fecd | 2026-10-23 | used | index.md#vocabulary, index.md#rules | |
| S3 | TTS-SWEEP.md | e87e06dd4d3e | 2026-11-03 | used | index.md#vocabulary | |
| S4 | LESSONS.md | 9ee7e4bc626d | 2026-10-23 | used | index.md#rules | |
| S5 | DESIGN.md | 13dc221670cd | 2026-10-23 | used | index.md#rules | |
| S6 | experiments/ruleFsts/README.md | 9e08466c1ea4 | 2026-10-23 | ranked, unread | | |
| S7 | CASCADE-SWEEP.md | 8ec850494326 | 2026-10-23 | ranked, unread | | |
| S8 | WARROOM.md | b39951a7917d | 2026-10-23 | ranked, unread | | |
