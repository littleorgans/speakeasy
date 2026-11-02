"""Merge fast-tts result files (CONTRACT.md) into one summary JSON and a markdown table.

Usage: python3 -I experiments/fast-tts/summarize.py <summary.json> <result.json>...
Validates every file, derives metrics with fasttts.derive, and reports median and
p90 per engine, mode, temperature and text, plus pooled per length class.
Adapted from experiments/audio8-tts/summarize.py (audio8-tts thread).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasttts as ft  # noqa: E402

METRICS = ("firstAudioMs", "playbackStartMs", "rtf", "underruns", "stallMs", "audioMs")


def stats(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    p90 = statistics.quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]
    return {"n": len(values), "median": round(statistics.median(values), 3), "p90": round(p90, 3),
            "max": round(max(values), 3)}


def aggregate(runs: list[dict]) -> dict:
    return {metric: stats([run[metric] for run in runs]) for metric in METRICS}


def main() -> None:
    target, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    if not sources:
        raise SystemExit(__doc__)
    lengths = {text["id"]: text["length"] for text in ft.load_cohort()}
    engines: dict[str, dict] = {}
    for source in sources:
        doc = json.loads(source.read_text())
        try:
            ft.validate(doc)
        except ValueError as error:
            raise SystemExit(f"{source}: {error}")
        entry = engines.setdefault(doc["engine"]["id"], {
            "engine": doc["engine"], "environment": doc["environment"], "files": [], "processes": [], "runs": []})
        if doc["environment"]["machine"] != entry["environment"]["machine"]:
            raise SystemExit(f"{source}: machine differs from earlier files for {doc['engine']['id']}")
        entry["files"].append(source.name)
        entry["processes"].append(doc["process"])
        for run in doc["runs"]:
            entry["runs"].append({"textId": run["textId"], "length": lengths.get(run["textId"], "other"),
                                  "mode": run["mode"], "temperature": run["temperature"], "repeat": run["repeat"],
                                  **ft.derive(run)})

    out, table = [], []
    for engine_id, entry in engines.items():
        processes = entry.pop("processes")
        runs = entry.pop("runs")
        groups: dict[tuple, list] = {}
        pooled: dict[tuple, list] = {}
        for run in runs:
            groups.setdefault((run["mode"], run["temperature"], run["length"], run["textId"]), []).append(run)
            pooled.setdefault((run["mode"], run["temperature"], run["length"]), []).append(run)
        load = {temp: stats([p["loadMs"] for p in processes if p["temperature"] == temp]) for temp in ft.TEMPERATURES}
        entry.update({
            "timed": all(p["timed"] for p in processes),
            "playback": sorted({p["playback"] for p in processes if p["playback"]}),
            "loadMs": load,
            "warmupMs": stats([p["warmupMs"] for p in processes if p["temperature"] == "warm"]),
            "byText": [{"mode": m, "temperature": t, "length": length, "textId": text, **aggregate(rs), "runs": rs}
                       for (m, t, length, text), rs in groups.items()],
            "byLength": [{"mode": m, "temperature": t, "length": length, **aggregate(rs)}
                         for (m, t, length), rs in pooled.items()],
        })
        out.append(entry)
        for row in entry["byLength"]:
            table.append((engine_id, row, load[row["temperature"]]))

    summary = {"schema": "fast-tts.summary.v1", "timed": all(e["timed"] for e in out), "engines": out}
    target.write_text(json.dumps(summary, indent=1) + "\n")

    def cell(s: dict | None, scale: float = 1, digits: int = 0) -> str:
        return "n/a" if s is None else f"{s['median'] * scale:.{digits}f} / {s['p90'] * scale:.{digits}f}"

    print(f"{'Timed' if summary['timed'] else 'UNTIMED smoke'}; cells are median / p90; first audio = first decoded samples to caller.")
    print("| engine | mode | temp | length | runs | first audio ms | playback start ms | RTF | underruns max | load ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for engine_id, row, load in table:
        print(f"| {engine_id} | {row['mode']} | {row['temperature']} | {row['length']} | {row['firstAudioMs']['n']} "
              f"| {cell(row['firstAudioMs'])} | {cell(row['playbackStartMs'])} | {cell(row['rtf'], digits=3)} "
              f"| {row['underruns']['max']:.0f} | {cell(load)} |")


if __name__ == "__main__":
    main()
