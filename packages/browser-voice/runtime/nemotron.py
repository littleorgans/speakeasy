"""NVIDIA Nemotron Speech Streaming 0.6B on MLX, from the stt-ux mlx-stream track.

Backend options are exactly setupId and an absolute local modelPath (the
nvidia/nemotron-speech-streaming-en-0.6b snapshot at ebe59e5). Construction
is lazy; load attests the snapshot and runtime, then loads the model with no
warmup. The engine only streams: transcribe runs one greedy streaming session
over the whole take at the research's 560 ms delay, and feed runs the same
session live at the delay the page chose. Its words are final once emitted.
"""

import copy
from pathlib import Path
import sys
import time

from pins import attest_snapshot, runtime_identity


SETUP_ID = "nemotron/streaming"
PYTHON_VERSION = "3.12.5"
# The Kokoro MLX environment (requirements-kokoro-mlx.txt) carries this mlx-audio.
RUNTIME_VERSIONS = {"mlx-audio": "0.6.3", "mlx": "0.32.4", "numpy": "2.5.4"}
# Bytes and SHA256 from the research mlx-stream pins.json.
FILES = {
    "config.json": (1284, "dffe850bc79ad2b0f8117804502b24d2c4a445aafbed4c1e40f8d78e0cb44065"),
    "tokenizer.json": (400216, "60dc0361763fa3cd62df60f34fca3e61134676a939967853931db5f3869b2db2"),
    "processor_config.json": (525, "cf35efc9abdd0963db7e96967f8a91d9c8243803b9f6bbd900aa936ee3ce42f3"),
    "model.safetensors": (2472413604, "bddd8a7300826efd19cf7e01f1c7db8402bed6786fc4c7739632894f69c71473"),
}
# The live setting the page offers (src/live-settings.ts); the final keeps 560.
DELAY_MS = (560, 1_120, 160)
DEFAULT_DELAY_MS = 560
DECODE_BUDGET = 256


def _pending(session):
    # As the research replay drains: mlx-audio 0.6.3 has no public accessor for
    # queued or encoded audio, so these private fields are pinned with it.
    with session._lock:
        return session._queued > 0 or bool(session._encoded)


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath"}:
            raise ValueError("Nemotron options require exactly setupId and modelPath")
        if options["setupId"] != SETUP_ID:
            raise ValueError("Nemotron backend requires nemotron/streaming")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or "\0" in model_path or not Path(model_path).is_absolute():
            raise ValueError("Nemotron modelPath must be absolute")
        self._model_path = Path(model_path)
        self._model = None
        self._metadata = None
        self._live = None
        self._text = ""

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, FILES, "Nemotron")
        runtime = runtime_identity(PYTHON_VERSION, RUNTIME_VERSIONS, "requirements-kokoro-mlx.txt", "Nemotron")
        start = time.perf_counter()
        import numpy as np
        import mlx.core as mx
        from mlx_audio.stt import load

        model = load(str(self._model_path), lazy=False, strict=True)
        if model.model_type != "nemotron_asr_streaming" or model.preprocessor_config.sample_rate != 16000:
            raise ValueError("Nemotron model identity differs from the pinned streaming model")
        self._np, self._mx, self._model = np, mx, model
        self._metadata = {"setupId": SETUP_ID, "runtime": runtime, "loadMs": (time.perf_counter() - start) * 1000,
                          "delayMs": DEFAULT_DELAY_MS, "decodeBudget": DECODE_BUDGET, "temperature": 0}
        return copy.deepcopy(self._metadata)

    def _samples(self, samples):
        if self._model is None:
            raise RuntimeError("Nemotron load() must complete before use")
        view = memoryview(samples)
        if view.ndim != 1 or view.itemsize != 4 or not view.c_contiguous or sys.byteorder != "little":
            raise ValueError("Nemotron samples must be a contiguous little endian Float32 view")
        data = self._np.frombuffer(view, dtype="<f4")
        if not self._np.isfinite(data).all():
            raise ValueError("Nemotron samples must be finite")
        return data

    def _session(self, delay_ms):
        return self._model.create_streaming_session(temperature=0.0, language="en", transcription_delay_ms=delay_ms)

    def _drain(self, session):
        text = ""
        while _pending(session):
            text += "".join(session.step(max_decode_tokens=DECODE_BUDGET))
        self._mx.synchronize()
        return text

    def feed(self, samples, start, settings=None):
        """Live words: every whole encoder chunk heard so far decodes; words never change once shown."""
        data = self._samples(samples)
        if start:
            settings = {} if settings is None else settings
            if not isinstance(settings, dict) or set(settings) - {"delayMs"}:
                raise ValueError("Nemotron live settings are delayMs")
            delay = settings.get("delayMs", DEFAULT_DELAY_MS)
            if isinstance(delay, bool) or delay not in DELAY_MS:
                raise ValueError("Nemotron delay must be an offered value")
            self._live, self._text = self._session(delay), ""
        elif settings is not None or self._live is None:
            raise ValueError("Nemotron live settings go with start, which opens the stream")
        started = time.perf_counter()
        if data.size:
            self._live.feed(data)
        self._text += self._drain(self._live)
        return {"stable": self._text, "tentative": "", "processingMs": (time.perf_counter() - started) * 1000}

    def transcribe(self, samples):
        data = self._samples(samples)
        if not data.size:
            raise ValueError("Nemotron needs samples")
        self._live = None
        started = time.perf_counter()
        session = self._session(DEFAULT_DELAY_MS)
        text = ""
        # The session queues at most 30 s; feed in blocks and drain as it goes.
        for offset in range(0, data.size, 2_560):
            session.feed(data[offset:offset + 2_560])
            text += self._drain(session)
        session.close()
        while not session.done:
            text += "".join(session.step(max_decode_tokens=DECODE_BUDGET))
        self._mx.synchronize()
        return {"text": text, "processingMs": (time.perf_counter() - started) * 1000,
                "diagnostics": {"sampleCount": int(data.size), "delayMs": DEFAULT_DELAY_MS}}

    def close(self):
        self._live = None
