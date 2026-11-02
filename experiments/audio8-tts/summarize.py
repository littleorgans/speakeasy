"""Merge timing JSON from render-baselines.ts, audio8_bench.py and audio8_torch.py.

Usage: python3 -I experiments/audio8-tts/summarize.py <results.json> <timing.json>...
Writes medians per engine, voice, mode and text, keeps raw runs, and prints markdown tables:
one row per text, then one row per stream chunk (medians across runs, by chunk sequence).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def rows_from(data: dict) -> tuple[dict, list[dict]]:
    if "engines" in data:  # render-baselines.ts
        for engine in data["engines"]:
            meta = {k: engine[k] for k in ("engine", "model", "voice", "threads", "coldOpenMs", "nativeLoadMs", "warmupMs")}
            yield meta, [{"voice": engine["voice"], "mode": "sentence", **text} for text in engine["texts"]]
    else:
        meta = {k: data[k] for k in data if k not in ("results", "schema")}
        yield meta, data["results"]


def rounded(value):
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def chunk_medians(runs: list[dict]) -> list[dict]:
    """Median of each chunk field across runs; runs share a seed, so chunk counts match."""
    records = [run["chunkRecords"] for run in runs if "chunkRecords" in run]
    if not records:
        return []
    return [
        {key: statistics.median(chunks[seq][key] for chunks in records if seq < len(chunks)) for key in records[0][seq]}
        for seq in range(len(records[0]))
    ]


def main() -> None:
    target, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    engines, table = [], []
    for source in sources:
        for meta, rows in rows_from(json.loads(source.read_text())):
            entries = []
            for row in rows:
                runs = row["runs"]
                entry = {
                    "voice": row["voice"], "mode": row["mode"], "id": row["id"], "n": len(runs),
                    "ttfaMs": round(statistics.median(r["ttfaMs"] for r in runs), 1),
                    "rtf": round(statistics.median(r["rtf"] for r in runs), 3),
                    "audioMs": round(statistics.median(r["audioMs"] for r in runs), 1),
                    **{key: round(statistics.median(r[key] for r in runs), 1)
                       for key in ("prefillMs", "decodeMs") if key in runs[0]},
                    "chunks": rounded(chunk_medians(runs)),
                    "raw": rounded(runs),
                }
                entries.append(entry)
                table.append((meta["engine"], entry))
            engines.append({**meta, "results": entries})
    target.write_text(json.dumps({"schema": 1, "engines": engines}, indent=1) + "\n")
    print("| engine | voice | mode | text | n | TTFA ms | RTF | audio s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for engine, e in table:
        print(f"| {engine} | {e['voice']} | {e['mode']} | {e['id']} | {e['n']} | {e['ttfaMs']:.0f} | {e['rtf']:.2f} | {e['audioMs'] / 1000:.1f} |")
    print("\n| engine | text | chunk | ready ms | gap ms | chunk audio ms | window frames | decode ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for engine, e in table:
        for c in e["chunks"]:
            print(f"| {engine} | {e['id']} | {c['seq']:.0f} | {c['readyMs']:.0f} | {c['gapMs']:.0f} | {c['audioMs']:.0f} | "
                  f"{c['windowFrames']:.0f} | {c['decodeMs']:.0f} |")


if __name__ == "__main__":
    main()
