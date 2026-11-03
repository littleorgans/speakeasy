"""Keeps the GPU busy with continuous MLX LM generation until killed or the duration ends.
usage: llm_load.py <model dir> <seconds> <log.jsonl>; one line per generation with tokens/s."""
import json, sys, time
from mlx_lm import load, stream_generate

model, tok = load(sys.argv[1])
end = time.time() + float(sys.argv[2])
prompt = tok.apply_chat_template([{"role": "user", "content": "Write a long, detailed history of railway signalling."}],
                                 add_generation_prompt=True, tokenize=False)
with open(sys.argv[3], "a") as log:
    while time.time() < end:
        last = None
        for last in stream_generate(model, tok, prompt, max_tokens=512):
            if time.time() >= end:
                break
        log.write(json.dumps({"t": time.time(), "tokens": last.generation_tokens, "tps": last.generation_tps,
                              "peakGB": last.peak_memory}) + "\n")
        log.flush()
