"""Hex Parakeet Unified bridge over the native worker built by hex/build.py.

Backend options contain absolute runnerPath and modelPath, plus an optional
setupId fixed to hex/unified. Construction starts nothing; load verifies the
model hash, starts the worker and waits for its ready line. Each take goes
through Hex's own stream pipeline inside the worker; feed runs the same
pipeline live, at the lookahead the page chose, and answers with its
committed and tentative words.
"""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time


SETUP_ID = "hex/unified"
MODEL_FILE = "parakeet-unified-en-0.6b-Q8_0.gguf"
MODEL_SHA256 = "4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795"
RUNTIME_COMMIT = "7045d1c0fe7cee37747b2792977421d435e75fca"
HEX_COMMIT = "826bdbc44c8806630a45c08ed8cef35a36817a17"
PROCESSING_TIMING = "worker Instant around session, stream feeds, pause cap and finalize"
# Live lookahead the page offers (src/live-settings.ts), as transcribe.cpp's
# Unified (chunk, right) menu in ms; Hex ships 2080, the model default.
LOOKAHEAD_MS = {2080: (1040, 1040), 1120: (560, 560), 480: (160, 320)}
DEFAULT_LOOKAHEAD_MS = 2080


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Backend:
    def __init__(self, options):
        required = {"runnerPath", "modelPath"}
        if (not isinstance(options, dict) or not required <= set(options)
                or set(options) - required - {"setupId"}):
            raise ValueError("Hex options require runnerPath, modelPath and optional setupId")
        if options.get("setupId", SETUP_ID) != SETUP_ID:
            raise ValueError("Hex backend requires hex/unified")
        for name in ("runnerPath", "modelPath"):
            value = options[name]
            if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
                raise ValueError(f"Hex {name} must be an absolute path")
        self.runner_path = Path(options["runnerPath"])
        self.model_path = Path(options["modelPath"])
        self.process = None
        self.metadata = None

    def _reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Hex worker exited with {self.process.poll()}")
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise RuntimeError("Hex worker reply must be an object")
        if reply.get("kind") == "error":
            raise RuntimeError(f"Hex worker: {reply.get('error')}")
        return reply

    def load(self):
        if self.metadata is not None:
            return copy.deepcopy(self.metadata)
        model_sha256 = _digest(self.model_path)
        if model_sha256 != MODEL_SHA256:
            raise ValueError("Hex model SHA256 mismatch")
        try:
            started = time.perf_counter()
            # Native diagnostics go to the transport's stderr, never the protocol.
            self.process = subprocess.Popen([str(self.runner_path), str(self.model_path)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
            ready = self._reply()
            if ready.get("kind") != "ready" or ready.get("backend") != "metal" or ready.get("variant") != "unified-en-0.6b":
                raise RuntimeError("Hex worker identity mismatch")
            self.metadata = {
                "setupId": SETUP_ID, "loadMs": (time.perf_counter() - started) * 1000,
                "modelFile": MODEL_FILE, "modelSha256": model_sha256, "runtime": "transcribe.cpp",
                "runtimeCommit": RUNTIME_COMMIT, "hexCommit": HEX_COMMIT, "runnerSha256": _digest(self.runner_path),
                "device": ready.get("device"), "workerLoadMs": ready.get("loadMs"),
                "pipeline": "Hex stream: 2560 sample feeds, PauseCap, 6400 trailing zeros",
                "loadTiming": "worker start to ready line", "processingTiming": PROCESSING_TIMING,
            }
            return copy.deepcopy(self.metadata)
        except BaseException:
            self.close()
            raise

    def _send(self, op, samples, extra=b""):
        if self.process is None or self.metadata is None:
            raise RuntimeError("Hex backend is not loaded")
        if sys.byteorder != "little":
            raise RuntimeError("Hex worker requires a little endian host")
        body = samples.tobytes()
        self.process.stdin.write(op + struct.pack("<I", len(body) // 4) + extra + body)
        self.process.stdin.flush()
        return body

    def feed(self, samples, start, settings=None):
        """Live words: start opens a fresh stream, as Hex does per dictation, at the chosen lookahead."""
        if not start:
            if settings is not None:
                raise ValueError("Hex live settings go with start")
            self._send(b"F", samples)
        else:
            settings = {} if settings is None else settings
            if not isinstance(settings, dict) or set(settings) - {"lookaheadMs"}:
                raise ValueError("Hex live settings are lookaheadMs")
            lookahead = settings.get("lookaheadMs", DEFAULT_LOOKAHEAD_MS)
            if isinstance(lookahead, bool) or lookahead not in LOOKAHEAD_MS:
                raise ValueError("Hex lookahead must be an offered value")
            self._send(b"S", samples, struct.pack("<ii", *LOOKAHEAD_MS[lookahead]))
        reply = self._reply()
        if reply.get("kind") != "partial":
            raise RuntimeError("Hex worker returned no partial")
        return {"stable": reply.get("stable"), "tentative": reply.get("tentative"),
                "processingMs": reply.get("processingMs")}

    def transcribe(self, samples):
        body = self._send(b"T", samples)
        reply = self._reply()
        if reply.get("kind") != "result" or not isinstance(reply.get("text"), str):
            raise RuntimeError("Hex worker returned no result")
        return {
            "text": reply["text"], "rawText": reply.get("rawText", reply["text"]),
            "processingMs": reply["processingMs"],
            "diagnostics": {"sampleCount": len(body) // 4, "pauseDroppedSamples": reply.get("pauseDroppedSamples"),
                            "processingTiming": PROCESSING_TIMING},
        }

    def close(self):
        process, self.process = self.process, None
        self.metadata = None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait()
