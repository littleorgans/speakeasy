"""fast-tts candidate: Kokoro v1.0 on MLX (mlx-audio), Apple GPU.

Emits the shared result JSON (CONTRACT.md) for one engine per process through
fasttts.run_engine, with the flags of render-baselines.ts.

Stream mode follows the baselines' chunk policy so first audio compares: a port
of speak-easy planSegments (short first chunk of at most 4 words or the first
clause, then whole sentences) and trimSilence (edge trim plus 150 ms gap) from
packages/speech-io/src/tts/stream.ts. The segments go to one Model.generate call,
newline separated; each yield is one completed segment waveform. Cancellation
checkpoint: between segments only, an active segment runs to completion.

Run from the speak-easy repo root with the env from README.md:
    $TMPDIR/venv-kokoro-mlx/bin/python -I experiments/fast-tts/candidates/kokoro-mlx/render.py \
        --engine kokoro-mlx-heart --temperature warm --json <file> [--clips <dir>]
"""

from __future__ import annotations

import os
import re
import sys
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # local snapshot only; online, the first generate call took 39 s once (README)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/fast-tts
import fasttts as ft  # noqa: E402

import numpy as np  # noqa: E402

REPO = "mlx-community/Kokoro-82M-bf16"
REVISION = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
SNAPSHOT = "kokoro-82m-mlx-bf16-a71e4d38"
WEIGHTS = ("kokoro-v1_0.safetensors", "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8")
VOICES = {"af_heart": "2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094",
          "am_michael": "3940147ded35deba0bb52e8132f89b719298e0520258c34584358aa5a24da2ea"}
ENGINES = {"kokoro-mlx-heart": ("Kokoro v1.0 MLX bf16 · af_heart", "af_heart"),
           "kokoro-mlx-michael": ("Kokoro v1.0 MLX bf16 · am_michael", "am_michael")}
LANG_CODE = "a"  # misaki en-US G2P with espeak-ng fallback

# speak-easy packages/speech-io/src/tts/stream.ts constants, ported unchanged.
FIRST_CHUNK_MAX_WORDS = 4
SILENCE_AMPLITUDE = 0.015
EDGE_KEEP_MS = 8
INTER_SENTENCE_GAP_MS = 150


def split_sentences(text: str) -> list[str]:
    """splitSentences: terminators plus closing quotes stay attached."""
    pieces = re.findall(r"[^.!?]*[.!?]+[)\]\"'”’]*\s*|[^.!?]+$", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def plan_segments(text: str) -> list[str]:
    """planSegments: carve the first sentence at its first clause or FIRST_CHUNK_MAX_WORDS."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    words = sentences[0].split()
    clause = next((i + 1 for i, word in enumerate(words) if re.search(r"[,;:—–-]$", word)), len(words))
    cut = min(clause, FIRST_CHUNK_MAX_WORDS)
    head = [sentences[0]] if cut >= len(words) else [" ".join(words[:cut]), " ".join(words[cut:])]
    return head + sentences[1:]


def trim_silence(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """trimSilence: drop edge near-silence (keeping a guard), then append the fixed gap."""
    gap = np.zeros(round(sample_rate * INTER_SENTENCE_GAP_MS / 1000), dtype=np.float32)
    loud = np.flatnonzero(np.abs(samples) >= SILENCE_AMPLITUDE)
    if loud.size == 0:
        return gap
    keep = round(sample_rate * EDGE_KEEP_MS / 1000)
    start, end = max(0, loud[0] - keep), min(samples.size, loud[-1] + 1 + keep)
    return np.concatenate([samples[start:end], gap])


class KokoroMlx:
    def __init__(self, engine_id: str, root: Path) -> None:
        self.label, self.voice = ENGINES[engine_id]
        self.snapshot = root / SNAPSHOT
        ft.verify_digests([(self.snapshot / WEIGHTS[0], WEIGHTS[1]),
                           (self.snapshot / f"voices/{self.voice}.safetensors", VOICES[self.voice])])

    def load(self) -> dict:
        start = ft.now_ms()
        from mlx_audio.tts.utils import load_model

        self.model = load_model(str(self.snapshot))
        loaded = ft.now_ms()
        # G2P pipeline (spaCy, misaki, espeak-ng): built lazily by the first generate, so build it here.
        self.model._get_pipeline(LANG_CODE)
        self.sample_rate = self.model.config.sample_rate
        return {"modelLoadMs": round(loaded - start, 3), "g2pPipelineMs": round(ft.now_ms() - loaded, 3),
                "note": "loadMs is mlx_audio load_model on the local snapshot plus the misaki G2P pipeline; "
                        "mlx-audio reloads the voice pack inside every generate call"}

    def describe(self) -> tuple[dict, dict[str, str]]:
        versions = {name: version(name) for name in ("mlx-audio", "mlx", "misaki", "spacy", "en-core-web-sm",
                                                     "espeakng-loader", "phonemizer-fork", "numpy")}
        return {
            "label": self.label, "runtime": f"mlx-audio {versions['mlx-audio']} (mlx {versions['mlx']})",
            "provider": "mlx-metal-gpu", "model": REPO, "modelRevision": REVISION, "voice": self.voice,
            "sampleRate": self.sample_rate,
            "streaming": "segment waveforms: speak-easy planSegments + trimSilence port, one generate call, "
                         "newline separated segments; not incremental within a segment",
            "chunkPolicy": f"first chunk: first clause or {FIRST_CHUNK_MAX_WORDS} words of sentence 1, "
                           "then whole sentences; each chunk trimmed, 150 ms gap appended",
            "cancellation": "between segments (host stops pulling the generator); an active segment's "
                            "MLX inference is not preempted",
        }, versions

    def generate(self, text: str, split_pattern: str | None):
        """Each completed segment waveform as float32 numpy (np.asarray forces MLX evaluation)."""
        for result in self.model.generate(text, voice=self.voice, speed=1.0, lang_code=LANG_CODE,
                                          split_pattern=split_pattern):
            yield np.asarray(result.audio, dtype=np.float32).reshape(-1)

    def stream(self, text: str):
        """(trimmed samples, planned line). Labels assume one yield per line; misaki splits a line only past 510 phonemes."""
        lines = plan_segments(text)
        for index, segment in enumerate(self.generate("\n".join(lines), r"\n+")):
            yield trim_silence(segment, self.sample_rate), lines[index] if index < len(lines) else None

    def full(self, text: str) -> np.ndarray:
        return np.concatenate(list(self.generate(text, None)))


if __name__ == "__main__":
    ft.run_engine(__doc__, ENGINES, KokoroMlx)
