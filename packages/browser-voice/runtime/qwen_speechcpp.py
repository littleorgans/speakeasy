"""Qwen3 TTS CustomVoice 0.6B and 1.7B on speech.cpp, Metal, as speech out voices.

Backend options are exactly setupId, an absolute local modelPath (the pinned
payload directory), files, its pins from the host registry, and runnerPath,
the speech.cpp 0.10.0 `speech` binary (the release Qwen3 ASR uses; the
voice-out research built the same commit). Construction starts nothing; load
attests the payload, starts `speech worker` on the GPU without warmup, as the
research's confirmation bench did, and waits for its ready line. speak sends
one synthesize request with the reading's settings and gathers its 16 bit PCM
chunks until the end. A field at its stock value is omitted, so a reading with
nothing changed sends exactly today's request.
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


# Per setup: its GGUF and whether it takes instructions (speech.cpp gives them to every size but 0.6B).
MODELS = {
    "qwen3-tts-0.6b": {"file": "Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf", "instructions": False},
    "qwen3-tts-1.7b": {"file": "Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf", "instructions": True},
}
VOICES = ("ryan", "aiden", "dylan", "eric", "ono_anna", "serena", "sohee", "uncle_fu", "vivian")
SAMPLE_RATE = 24_000
THREADS = 2
# Today's request: English, seed 1; everything else at the model file's defaults.
DEFAULTS = {"instructions": "", "language": "en", "seed": 1, "maxSeconds": 0, "temperature": 0.9, "topP": 1,
            "topK": 50, "repetitionPenalty": 1.05, "greedy": False, "codeTemperature": 0.9, "codeTopP": 1,
            "codeTopK": 50, "codeGreedy": False}
# Host setting, native field, and the greedy switch that makes it step aside.
SAMPLING = (("temperature", "temperature", "greedy"), ("topP", "top_p", "greedy"), ("topK", "top_k", "greedy"),
            ("repetitionPenalty", "repetition_penalty", None),
            ("codeTemperature", "code_predictor_temperature", "codeGreedy"),
            ("codeTopP", "code_predictor_top_p", "codeGreedy"), ("codeTopK", "code_predictor_top_k", "codeGreedy"))


def native_fields(settings, instructions_allowed):
    """The reading's settings as speech.cpp worker fields after voice and text; stock values are omitted."""
    chosen = {**DEFAULTS, **settings}
    if set(chosen) != set(DEFAULTS):
        raise ValueError("Unsupported Qwen3 TTS setting")
    fields = {"language": chosen["language"], "seed": chosen["seed"]}
    # Auto is the native default and not among its language choices: leave it unset.
    if chosen["language"] == "auto":
        del fields["language"]
    if type(chosen["seed"]) is not int or not 0 <= chosen["seed"] <= 2**53 - 1:
        raise ValueError("Qwen3 TTS seed must be an integer")
    if chosen["maxSeconds"]:
        fields["max_seconds"] = float(chosen["maxSeconds"])
    for switch, native in (("greedy", "do_sample"), ("codeGreedy", "code_predictor_do_sample")):
        if chosen[switch]:
            fields[native] = False
    for name, native, switch in SAMPLING:
        # Greedy refuses its stack's sampling fields, so they are never sent with it.
        if chosen[name] != DEFAULTS[name] and not (switch and chosen[switch]):
            fields[native] = chosen[name]
    if chosen["instructions"]:
        if not instructions_allowed:
            raise ValueError("Qwen3 TTS 0.6B takes no instructions")
        fields["instructions"] = chosen["instructions"]
    return fields


class Backend:
    # The worker passes per reading settings only to backends that declare them.
    takes_settings = True

    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files", "runnerPath"}:
            raise ValueError("Qwen3 TTS options require exactly setupId, modelPath, files and runnerPath")
        if options["setupId"] not in MODELS:
            raise ValueError("Qwen3 TTS backend requires a Qwen3 TTS CustomVoice setup")
        self._setup_id = options["setupId"]
        self._spec = MODELS[self._setup_id]
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
            [str(self._runner), "worker", str(self._model_path / self._spec["file"]), "--device", "gpu",
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
        self._metadata = {"setupId": self._setup_id, "runtime": "speech.cpp 0.10.0", "device": "gpu", "threads": THREADS,
                          "sampleRate": SAMPLE_RATE, "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def speak(self, text, voice, speed, settings=None):
        if self._model is None:
            raise ValueError("Qwen3 TTS backend is not loaded")
        if voice not in VOICES:
            raise ValueError("Unsupported Qwen3 TTS voice")
        if speed != 1:
            raise ValueError("Qwen3 TTS has no speed control; speak at 1")
        fields = native_fields(settings or {}, self._spec["instructions"])
        self._serial += 1
        request_id = str(self._serial)
        start = time.perf_counter()
        self._model.stdin.write(json.dumps({"type": "synthesize", "id": request_id, "text": text, "voice": voice,
                                            **fields}) + "\n")
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
