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
- The Realtime and Mercury rows come from the 15:35 UTC matrix run. Cerebras
  Sherpa ran separately at 15:45 UTC with `--turn-gap-ms 10000`. A 10,000 ms
  Cartesia retry still hit the queue limit on turn 5, so the completed Cartesia
  row ran at 15:54 UTC with `--turn-gap-ms 20000`. The gap occurs before the next
  WAV starts and stays outside the measured endpoint intervals.
- First token has different provider semantics. Realtime records the first
  output audio transcript delta. Cascade records the first LLM token piece.
- The null sink resolves as soon as the final audio segment arrives. First audio
  measures segment arrival because `ConversationLoop` records that timestamp
  before opening the sink. The result excludes device open time, buffering, and
  audible playback latency.

## Results

Run on 2026-08-17. All seven rows completed nine turns across the main matrix
and the two isolated Cerebras reruns described above.

| Config | Status | Cold endpoint to final | Warm endpoint to final | Cold first token | Warm first token | Cold first audio | Warm first audio | Turns | Reason |
|---|---|---|---|---|---|---|---|---|---|
| `realtime` | completed | 1.0ms | 32.1ms | 771.7ms | 533.4ms | **1006.6ms** | **765.8ms** | 9 | |
| `cerebras-sherpa` | completed | 1.0ms | 26.6ms | 649.7ms | 602.8ms | 1312.9ms | 1400.0ms | 9 | isolated, 10,000 ms gap |
| `cerebras-cartesia` | completed | 1.0ms | **23.2ms** | 2506.0ms | 492.6ms | 3191.2ms | 1089.9ms | 9 | isolated, 20,000 ms gap |
| `mercury-instant-sherpa` | completed | **0.7ms** | 26.1ms | 905.7ms | **486.7ms** | 1893.4ms | 1380.3ms | 9 | |
| `mercury-instant-cartesia` | completed | 0.8ms | 26.1ms | **501.5ms** | 521.7ms | **1239.1ms** | 1191.1ms | 9 | |
| `mercury-low-sherpa` | completed | **0.7ms** | 25.2ms | 1403.7ms | 723.0ms | 2148.7ms | 1653.0ms | 9 | |
| `mercury-low-cartesia` | completed | **0.7ms** | **25.1ms** | 617.8ms | 545.4ms | 1353.3ms | **1058.7ms** | 9 | |

## Findings

- **Realtime reached audio first.** Its 765.8ms warm first audio was 292.9ms
  faster than the best completed cascade row, `mercury-low-cartesia` at
  1058.7ms.
- **Cartesia improved every complete cascade comparison.** Its warm first audio
  gain over Sherpa was 310.1ms for Cerebras, 189.2ms for Mercury instant, and
  594.3ms for Mercury low.
- **Cerebras needs request spacing for a full row.** A 10,000 ms gap completed
  Sherpa but left Cartesia partial after a turn 5 queue limit. A 20,000 ms gap
  completed all nine Cartesia turns.

## Verdict

As measured on 2026-08-17, keep Realtime as the latency choice. Use
`mercury-low-cartesia` when the cascade architecture is required. Rerun the two
Cerebras rows with the documented gaps when comparing them again.

## Reproduce

```
node packages/convo-engine/src/sweep/run.ts --runs 3 --utterances utt-2026-07-03T20-26-00.828Z,utt-2026-07-03T20-32-04.686Z,utt-2026-07-03T20-30-23.115Z
node packages/convo-engine/src/sweep/run.ts --only cerebras-sherpa --runs 3 --turn-gap-ms 10000 --utterances utt-2026-07-03T20-26-00.828Z,utt-2026-07-03T20-32-04.686Z,utt-2026-07-03T20-30-23.115Z
node packages/convo-engine/src/sweep/run.ts --only cerebras-cartesia --runs 3 --turn-gap-ms 20000 --utterances utt-2026-07-03T20-26-00.828Z,utt-2026-07-03T20-32-04.686Z,utt-2026-07-03T20-30-23.115Z
```

Each command overwrites `results/cascade-sweep.txt` and
`results/cascade-sweep.json`. Both files remain ignored local artifacts. This
document combines the completed rows from the three commands.
