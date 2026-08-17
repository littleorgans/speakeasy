# Cascade versus Realtime sweep

Headless latency comparison of OpenAI Realtime and six cascade configurations
through the same conversation loop, speech recognizer, corpus, and timing code.

## Method

- The sweep drives `ConversationLoop` with `ScriptedWavSource` at the recorded
  20 ms frame cadence. Each WAV boundary commits one manual STT turn, so one
  labelled corpus file produces one measured conversation turn.
- One prepared Sherpa `VoiceToText` instance is reused across every row. The
  matrix changes only the responder, LLM, reasoning effort, and TTS engine.
- The pinned corpus subset is
  `utt-2026-07-03T20-26-00.828Z` ("Hello can you hear me"),
  `utt-2026-07-03T20-32-04.686Z` ("Open Chrome browser"), and
  `utt-2026-07-03T20-30-23.115Z` ("Spawn 10 Codex agents").
- This run uses three passes. Turn 1 is the cold measurement. The warm value is
  the median of turns 2 through 9.
- First token has different provider semantics. Realtime records the first
  output audio transcript delta. Cascade records the first LLM token piece.
- The null sink resolves as soon as the final audio segment arrives. First audio
  measures segment arrival because `ConversationLoop` records that timestamp
  before opening the sink. The result excludes device open time, buffering, and
  audible playback latency.

## Results

Run at 2026-08-17 15:35 UTC. Five rows completed nine turns. Both Cerebras rows
continued through the matrix but were marked skipped after HTTP 429 failures.

| Config | Status | Cold endpoint to final | Warm endpoint to final | Cold first token | Warm first token | Cold first audio | Warm first audio | Turns | Reason |
|---|---|---|---|---|---|---|---|---|---|
| `realtime` | completed | 1.0ms | 32.1ms | 771.7ms | 533.4ms | **1006.6ms** | **765.8ms** | 9 | |
| `cerebras-sherpa` | skipped | n/a | n/a | n/a | n/a | n/a | n/a | 7 | HTTP 429 queue limit on turn 4 |
| `cerebras-cartesia` | skipped | n/a | n/a | n/a | n/a | n/a | n/a | 5 | HTTP 429 request quota on turn 3 |
| `mercury-instant-sherpa` | completed | **0.7ms** | 26.1ms | 905.7ms | **486.7ms** | 1893.4ms | 1380.3ms | 9 | |
| `mercury-instant-cartesia` | completed | 0.8ms | 26.1ms | **501.5ms** | 521.7ms | **1239.1ms** | 1191.1ms | 9 | |
| `mercury-low-sherpa` | completed | **0.7ms** | 25.2ms | 1403.7ms | 723.0ms | 2148.7ms | 1653.0ms | 9 | |
| `mercury-low-cartesia` | completed | **0.7ms** | **25.1ms** | 617.8ms | 545.4ms | 1353.3ms | **1058.7ms** | 9 | |

## Findings

- **Realtime reached audio first.** Its 765.8ms warm first audio was 292.9ms
  faster than the best completed cascade row, `mercury-low-cartesia` at
  1058.7ms.
- **Cartesia improved both complete Mercury comparisons.** The warm first audio
  gain over Sherpa was 189.2ms for Mercury instant and 594.3ms for Mercury low.
- **Cerebras produced no comparable row.** HTTP 429 responses left seven Sherpa
  turns and five Cartesia turns. The report keeps those partial measurements in
  the JSON dump but excludes their medians from the table.

## Verdict

As measured on 2026-08-17, keep Realtime as the latency choice. Use
`mercury-low-cartesia` when the cascade architecture is required. Rerun the two
Cerebras rows after their provider quota recovers before making a Cerebras
comparison.

## Reproduce

```
node packages/convo-engine/src/sweep/run.ts --runs 3 --utterances utt-2026-07-03T20-26-00.828Z,utt-2026-07-03T20-32-04.686Z,utt-2026-07-03T20-30-23.115Z
```

The command writes the summary to `results/cascade-sweep.txt` and the 57
completed per-turn records to `results/cascade-sweep.json`. Both files remain
ignored local artifacts.
