"""Bridge over the native speech in workers: transcribe.cpp (hex/build.py) and Apple (apple_speech/build.py).

Both speak one stdin protocol (hex/worker.rs). Backend options contain an
absolute runnerPath, an absolute modelPath unless the system manages the
weights, and an optional setupId from SETUPS (default hex/unified).
Construction starts nothing; load verifies the model hash (pins.file_sha256), starts the worker
in that setup's mode and waits for its ready line. For hex/unified each take
goes through Hex's own stream pipeline inside the worker, and feed runs the
same pipeline live, at the lookahead the page chose, answering with its
committed and tentative words; Apple streams the same way. The whole take
setups (Hex with its word list, Cohere) give final text only; Cohere's take
may carry its one final setting, the names bias.
"""

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

from pins import file_sha256


SETUP_ID = "hex/unified"
HEX_MODEL = {"file": "parakeet-unified-en-0.6b-Q8_0.gguf", "backend": "metal",
             "sha256": "4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795", "variant": "unified-en-0.6b"}
# Each streaming setup's one live setting (src/live-settings.ts): the page's
# values, each as the two i32 its worker's start request carries.
# Hex: transcribe.cpp's Unified (chunk, right) menu in ms; Hex ships 2080.
HEX_LIVE = {"setting": "lookaheadMs", "default": 2080, "values": {2080: (1040, 1040), 1120: (560, 560), 480: (160, 320)}}
APPLE_LIVE = {"setting": "fastResults", "default": 0, "values": {0: (0, 0), 1: (1, 0)}}
# Cohere's one final setting (src/final-settings.ts): the logit bias toward its
# five names, sent as the worker's `B` take; Off sends today's `T` take.
COHERE_FINAL = {"setting": "namesBias", "default": 0, "values": (0, 1.25, 2.5)}
# Every setup a native worker runs: its model pin, worker mode and pipeline.
SETUPS = {
    "hex/unified": {**HEX_MODEL, "mode": [], "live": HEX_LIVE,
                    "pipeline": "Hex stream: 2560 sample feeds, PauseCap, 6400 trailing zeros"},
    "hex/vocabulary": {**HEX_MODEL, "mode": ["--whole=vocabulary"], "live": None,
                       "pipeline": "whole take session.run, word list Claude GPT Grok Codex TM whoami at boost 2.5, "
                                   "padded to 24000 then 6400 zeros"},
    # handy-computer/cohere-transcribe-03-2026-gguf at dfa4ade, Apache 2.0.
    "cohere/transcribe": {"file": "cohere-transcribe-03-2026-Q8_0.gguf",
                          "sha256": "931916663432fd895423a4291a8400221802b288967ca2d435fc5e3141c9e71e",
                          "variant": "cohere-transcribe-03-2026", "mode": ["--whole=cohere"], "live": None, "backend": "metal",
                          "final": COHERE_FINAL,
                          "pipeline": "whole take session.run per window of at most 560000 samples, padded to 24000"},
    # SpeechAnalyzer and SpeechTranscriber, en-US; the system manages its weights.
    "apple/speech": {"file": None, "sha256": None, "variant": "speech-transcriber", "backend": "apple", "mode": [],
                     "live": APPLE_LIVE, "pipeline": "SpeechAnalyzer, SpeechTranscriber volatile results, "
                                                     "2560 sample buffers, finalize through end of input"},
}
RUNTIME_COMMIT = "7045d1c0fe7cee37747b2792977421d435e75fca"
HEX_COMMIT = "826bdbc44c8806630a45c08ed8cef35a36817a17"
PROCESSING_TIMING = "worker Instant around the take's sessions and runs, or stream feeds, pause cap and finalize"


class Backend:
    def __init__(self, options):
        setup_id = options.get("setupId", SETUP_ID) if isinstance(options, dict) else None
        if setup_id not in SETUPS:
            raise ValueError("Native backend requires a native speech in setup")
        self.setup_id, self.spec = setup_id, SETUPS[setup_id]
        required = {"runnerPath"} | ({"modelPath"} if self.spec["file"] else set())
        if set(options) - {"setupId"} != required:
            raise ValueError(f"{setup_id} options require {', '.join(sorted(required))} and optional setupId")
        for name in sorted(required):
            value = options[name]
            if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
                raise ValueError(f"Hex {name} must be an absolute path")
        self.runner_path = Path(options["runnerPath"])
        self.model_path = Path(options["modelPath"]) if self.spec["file"] else None
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
        model_sha256 = file_sha256(self.model_path) if self.model_path else None
        if model_sha256 != self.spec["sha256"]:
            raise ValueError("Hex model SHA256 mismatch")
        try:
            started = time.perf_counter()
            # Native diagnostics go to the transport's stderr, never the protocol.
            model = [str(self.model_path)] if self.model_path else []
            self.process = subprocess.Popen([str(self.runner_path), *model, *self.spec["mode"]],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
            ready = self._reply()
            if (ready.get("kind") != "ready" or ready.get("backend") != self.spec["backend"]
                    or ready.get("variant") != self.spec["variant"]):
                raise RuntimeError("Hex worker identity mismatch")
            apple = self.spec["backend"] == "apple"
            self.metadata = {
                "setupId": self.setup_id, "loadMs": (time.perf_counter() - started) * 1000,
                "modelFile": self.spec["file"], "modelSha256": model_sha256,
                "runtime": "Apple Speech, system managed weights" if apple else "transcribe.cpp",
                **({"locale": ready.get("locale"), "reservedHere": ready.get("reservedHere")} if apple
                   else {"runtimeCommit": RUNTIME_COMMIT, "hexCommit": HEX_COMMIT}),
                "runnerSha256": file_sha256(self.runner_path),
                "device": ready.get("device"), "workerLoadMs": ready.get("loadMs"),
                "pipeline": self.spec["pipeline"],
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
        """Live words: start opens a fresh stream, as Hex does per dictation, with the chosen live setting."""
        live = self.spec["live"]
        if live is None:
            raise ValueError(f"{self.setup_id} gives final text only")
        if not start:
            if settings is not None:
                raise ValueError("Native live settings go with start")
            self._send(b"F", samples)
        else:
            settings = {} if settings is None else settings
            if not isinstance(settings, dict) or set(settings) - {live["setting"]}:
                raise ValueError(f"{self.setup_id} live settings are {live['setting']}")
            value = settings.get(live["setting"], live["default"])
            if isinstance(value, bool) or value not in live["values"]:
                raise ValueError(f"{self.setup_id} {live['setting']} must be an offered value")
            self._send(b"S", samples, struct.pack("<ii", *live["values"][value]))
        reply = self._reply()
        if reply.get("kind") != "partial":
            raise RuntimeError("Hex worker returned no partial")
        return {"stable": reply.get("stable"), "tentative": reply.get("tentative"),
                "processingMs": reply.get("processingMs")}

    def transcribe(self, samples, settings=None):
        final = self.spec.get("final")
        if settings is not None and final is None:
            raise ValueError(f"{self.setup_id} has no final settings")
        bias = 0
        if final is not None:
            settings = {} if settings is None else settings
            if not isinstance(settings, dict) or set(settings) - {final["setting"]}:
                raise ValueError(f"{self.setup_id} final settings are {final['setting']}")
            bias = settings.get(final["setting"], final["default"])
            if isinstance(bias, bool) or bias not in final["values"]:
                raise ValueError(f"{self.setup_id} {final['setting']} must be an offered value")
        body = self._send(b"B", samples, struct.pack("<f", bias)) if bias else self._send(b"T", samples)
        reply = self._reply()
        if reply.get("kind") != "result" or not isinstance(reply.get("text"), str):
            raise RuntimeError("Hex worker returned no result")
        if bias and reply.get("namesBias") != bias:
            raise RuntimeError("Hex worker did not apply the names bias")
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
