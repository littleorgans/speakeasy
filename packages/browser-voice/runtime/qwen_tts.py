"""Qwen3 TTS 0.6B CustomVoice on speech.cpp, Metal, as a speech out voice.

Backend options are exactly setupId, an absolute local modelPath (the pinned
payload directory), files, its pins from the host registry, and runnerPath,
the speech.cpp 0.10.0 `speech` binary (the release Qwen3 ASR uses; the
voice-out research built the same commit). Construction starts nothing; load
attests the payload, starts `speech worker` on the GPU without warmup, as the
research's confirmation bench did, and waits for its ready line. speak sends
one synthesize request and gathers its 16 bit PCM chunks until the end.
"""

import base64
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from pins import attest_snapshot, pinned_files


VOICE_ID = "qwen3-tts-0.6b"
MODEL_FILE = "Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf"
VOICES = ("ryan", "aiden")
SAMPLE_RATE = 24_000
SEED = 1
THREADS = 2


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files", "runnerPath"}:
            raise ValueError("Qwen3 TTS options require exactly setupId, modelPath, files and runnerPath")
        if options["setupId"] != VOICE_ID:
            raise ValueError("Qwen3 TTS backend requires qwen3-tts-0.6b")
        for name in ("modelPath", "runnerPath"):
            value = options[name]
            if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
                raise ValueError(f"Qwen3 TTS {name} must be absolute")
        self._model_path = Path(options["modelPath"])
        self._runner = Path(options["runnerPath"])
        self._files = pinned_files(options["files"], "Qwen3 TTS")
        self._model = None
        self._metadata = None
        self._serial = 0

    def _line(self):
        line = self._model.stdout.readline()
        if not line:
            raise RuntimeError(f"Qwen3 TTS worker exited with {self._model.poll()}")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise RuntimeError("Qwen3 TTS worker message must be an object")
        if message.get("type") == "fatal":
            raise RuntimeError("Qwen3 TTS worker failed")
        return message

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, self._files, "Qwen3 TTS")
        start = time.perf_counter()
        # Native diagnostics go to the transport's stderr; the protocol owns stdout.
        self._model = subprocess.Popen(
            [str(self._runner), "worker", str(self._model_path / MODEL_FILE), "--device", "gpu",
             "--threads", str(THREADS), "--no-warmup"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
            env={**os.environ, "SPEECH_NO_UPDATE_CHECK": "1"})
        try:
            ready = self._line()
            if ready.get("type") != "ready":
                raise RuntimeError("Qwen3 TTS worker sent no ready line")
        except BaseException:
            self.close()
            raise
        self._metadata = {"setupId": VOICE_ID, "runtime": "speech.cpp 0.10.0", "device": "gpu", "threads": THREADS,
                          "sampleRate": SAMPLE_RATE, "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def speak(self, text, voice, speed):
        if self._model is None:
            raise ValueError("Qwen3 TTS backend is not loaded")
        if voice not in VOICES:
            raise ValueError("Unsupported Qwen3 TTS voice")
        if speed != 1:
            raise ValueError("Qwen3 TTS has no speed control; speak at 1")
        self._serial += 1
        request_id = str(self._serial)
        start = time.perf_counter()
        self._model.stdin.write(json.dumps({"type": "synthesize", "id": request_id, "text": text, "voice": voice,
                                            "language": "en", "seed": SEED}) + "\n")
        self._model.stdin.flush()
        pcm = bytearray()
        while True:
            message = self._line()
            if message.get("id") != request_id:
                continue
            if message.get("type") == "chunk":
                pcm += base64.b64decode(message["pcm"])
            elif message.get("type") == "end":
                break
            elif message.get("type") in ("error", "cancelled"):
                raise RuntimeError("Qwen3 TTS synthesis failed")
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed) or len(pcm) % 2:
            raise ValueError("Qwen3 TTS returned malformed audio")
        count = len(pcm) // 2
        samples = memoryview(bytes(pcm)).cast("h")
        floats = bytearray(count * 4)
        view = memoryview(floats).cast("f")
        for index in range(count):
            view[index] = samples[index] / 32768
        return {"samples": bytes(floats), "sampleRate": SAMPLE_RATE, "processingMs": elapsed}

    def close(self):
        process, self._model, self._metadata = self._model, None, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait()
