"""Merge timing JSON from render-baselines.ts, audio8_bench.py and audio8_torch.py.

Usage: python3 -I experiments/audio8-tts/summarize.py <results.json> <timing.json>...
Writes medians per engine, voice, mode and text, keeps raw runs, and prints a markdown table.
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
                    "raw": [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in runs],
                }
                entries.append(entry)
                table.append((meta["engine"], entry))
            engines.append({**meta, "results": entries})
    target.write_text(json.dumps({"schema": 1, "engines": engines}, indent=1) + "\n")
    print("| engine | voice | mode | text | n | TTFA ms | RTF | audio s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for engine, e in table:
        print(f"| {engine} | {e['voice']} | {e['mode']} | {e['id']} | {e['n']} | {e['ttfaMs']:.0f} | {e['rtf']:.2f} | {e['audioMs'] / 1000:.1f} |")


if __name__ == "__main__":
    main()
