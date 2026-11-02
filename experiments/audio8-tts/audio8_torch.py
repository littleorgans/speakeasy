"""Render or time the cohort with Audio8 Realistic Air 0.6B through PyTorch (no ONNX export exists).

Mirrors upstream audio8_tts_infer.py's synthesize(): processor with reference audio and
transcript, model.generate, model.decode_audio. Loads once, one warmup, then `runs` passes.
Full synthesis only: generate returns all codes at the end, so TTFA equals total time.
Run with -I, HF_HOME under $TMPDIR and HF_HUB_OFFLINE=1; the model dir supplies remote code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voice", nargs=3, action="append", metavar=("NAME", "AUDIO", "TRANSCRIPT"),
                        required=True, help="reference voice: name, audio path, exact transcript")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--text", nargs="*", help="cohort ids (default all)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--label", default="a8-air")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoProcessor

    cohort = json.loads((Path(__file__).parent / "cohort.json").read_text())
    texts = [item for item in cohort["texts"] if not args.text or item["id"] in args.text]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, dtype=torch.float32)
    model = model.eval().to(device)
    cold_load_ms = (time.perf_counter() - start) * 1000
    rate = int(model.config.codec_sample_rate)

    def synthesize(text: str, audio: Path, transcript: str):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        inputs = processor(text=[text], reference_audio=[audio], reference_text=[transcript], return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                                    do_sample=True, generator=generator, return_dict_in_generate=True)
            waveforms, lengths = model.decode_audio(output.codes)
        if device.type == "mps":
            torch.mps.synchronize()
        return waveforms[0, : int(lengths[0])].float().cpu().numpy(), bool(output.finished[0])

    name, audio, transcript = args.voice[0]
    start = time.perf_counter()
    synthesize("Warm up.", Path(audio), transcript)
    warmup_ms = (time.perf_counter() - start) * 1000

    results = []
    for name, audio, transcript in args.voice:
        for item in texts:
            passes = []
            for run in range(args.runs):
                start = time.perf_counter()
                waveform, finished = synthesize(item["text"], Path(audio), transcript)
                total_ms = (time.perf_counter() - start) * 1000
                audio_ms = waveform.size / rate * 1000
                passes.append({"ttfaMs": total_ms, "lastReadyMs": total_ms, "audioMs": audio_ms,
                               "rtf": total_ms / audio_ms, "finished": finished})
                if run == 0:
                    sf.write(str(args.out / f"{item['id']}--{args.label}-{name}.wav"), waveform, rate)
            results.append({
                "voice": name, "mode": "full", "id": item["id"], "runs": passes,
                "medianTtfaMs": statistics.median(p["ttfaMs"] for p in passes),
                "medianRtf": statistics.median(p["rtf"] for p in passes),
            })
            print(f"{args.label} {name} {args.device} {item['id']}: ttfa {results[-1]['medianTtfaMs']:.0f} ms, "
                  f"rtf {results[-1]['medianRtf']:.2f}, finished {passes[0]['finished']}")
    args.json.write_text(json.dumps({
        "schema": 1, "engine": args.label, "modelDir": args.model_dir.name, "device": args.device,
        "torch": torch.__version__, "sampleRate": rate, "seed": args.seed, "runsPerText": args.runs,
        "coldLoadMs": cold_load_ms, "warmupMs": warmup_ms, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
