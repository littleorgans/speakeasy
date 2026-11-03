"""Bridge to resident native voice runners from the fast-tts candidates.

Backend options are exactly setupId, an absolute modelPath (the voice's
snapshot), files (its pins from the host registry) and an absolute runnerPath
(the binary its runtime/<voice>/build.py built). Construction starts nothing; load attests the snapshot and starts the
default voice's runner. speak sends one request line and gathers its frames.

Runner protocol (experiments/fast-tts/candidates/NATIVE.md): a request is one
line on stdin, `S<TAB>part<US>part...`; `Q` quits. Replies on stdout are frames
of little endian int32 kind, int32 count, then the payload: READY 1, END 3 and
ERROR 4 carry `count` bytes of UTF-8 JSON, AUDIO 2 carries `count` Float32
samples. A runner whose voice is a launch argument runs once per voice.
"""

import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import time

from pins import attest_snapshot, pinned_files


READY, AUDIO, END, ERROR = 1, 2, 3, 4
MAX_INFO_BYTES = 65_536
MAX_SAMPLES = 1_920_000
CLOSE_WAIT_S = 2
SEPARATORS = ("\t", "\n", "\r")
SPECS = {}


def spec(voice_id, *, label, sample_rate, voices, args, per_voice):
    SPECS[voice_id] = {"label": label, "sampleRate": sample_rate, "voices": voices, "args": args, "perVoice": per_voice}


# Kristin with libpiper's split decoder; the host sends Piper Plus phoneme characters, one part per sentence.
spec("piper-kristin-split", label="Piper split", sample_rate=22_050, voices=("kristin",), per_voice=False,
     args=lambda model, voice: [str(model / "en_US-kristin-medium.onnx"), str(model / "en_US-kristin-medium.onnx.json")])
# Kokoro v1.0 through FluidAudio's legacy Core ML route; the voice is a launch argument.
spec("kokoro-coreml", label="Kokoro Core ML", sample_rate=24_000, voices=("af_heart", "am_michael"), per_voice=True,
     args=lambda model, voice: [str(model), voice, "legacy"])


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files", "runnerPath"}:
            raise ValueError("Native voice options require exactly setupId, modelPath, files and runnerPath")
        if options["setupId"] not in SPECS:
            raise ValueError("Unsupported native voice")
        for name in ("modelPath", "runnerPath"):
            value = options[name]
            if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
                raise ValueError(f"Native voice {name} must be absolute")
        self._spec = SPECS[options["setupId"]]
        self._voice_id = options["setupId"]
        self._model_path = Path(options["modelPath"])
        self._runner = Path(options["runnerPath"])
        self._files = pinned_files(options["files"], self._spec["label"])
        self._processes = {}
        self._metadata = None

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, self._files, self._spec["label"])
        if not self._runner.is_file():
            raise ValueError(f"{self._spec['label']} runner is missing")
        start = time.perf_counter()
        ready = self._process(self._spec["voices"][0])[1]
        self._metadata = {"setupId": self._voice_id, "runtime": ready, "sampleRate": self._spec["sampleRate"],
                          "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def _process(self, voice):
        """The runner for a voice, started on first use: one per voice where the voice is a launch argument."""
        key = voice if self._spec["perVoice"] else None
        if key not in self._processes:
            process = subprocess.Popen([str(self._runner), *self._spec["args"](self._model_path, voice)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self._processes[key] = (process, None)
            kind, payload = self._frame(process)
            if kind != READY:
                raise RuntimeError(f"{self._spec['label']} runner did not start: {payload}")
            self._processes[key] = (process, payload)
        return self._processes[key]

    def _frame(self, process):
        header = process.stdout.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{self._spec['label']} runner exited with {process.poll()}")
        kind, count = struct.unpack("<ii", header)
        size = count * 4 if kind == AUDIO else count
        if count < 0 or kind not in (READY, AUDIO, END, ERROR) or size > max(MAX_INFO_BYTES, MAX_SAMPLES * 4):
            raise RuntimeError(f"{self._spec['label']} runner sent an invalid frame")
        payload = process.stdout.read(size)
        if len(payload) != size:
            raise RuntimeError(f"{self._spec['label']} runner exited mid frame")
        return kind, payload if kind == AUDIO else (json.loads(payload) if payload else {})

    def speak(self, text, voice, speed):
        if not self._metadata:
            raise ValueError(f"{self._spec['label']} backend is not loaded")
        if voice not in self._spec["voices"]:
            raise ValueError(f"Unsupported {self._spec['label']} voice")
        if speed != 1:
            raise ValueError(f"{self._spec['label']} has no speed control; speak at 1")
        if any(separator in text for separator in SEPARATORS):
            raise ValueError("speak text contains a runner protocol separator")
        process = self._process(voice)[0]
        start = time.perf_counter()
        process.stdin.write(f"S\t{text}\n".encode("utf-8"))
        process.stdin.flush()
        audio = bytearray()
        while True:
            kind, payload = self._frame(process)
            if kind == AUDIO:
                audio += payload
                if len(audio) > MAX_SAMPLES * 4:
                    raise RuntimeError(f"{self._spec['label']} audio exceeds limit")
            elif kind == END:
                break
            else:
                raise RuntimeError(f"{self._spec['label']} runner: {payload}")
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("Native voice timing must be finite")
        return {"samples": bytes(audio), "sampleRate": self._spec["sampleRate"], "processingMs": elapsed}

    def close(self):
        processes, self._processes = self._processes, {}
        for process, _ in processes.values():
            try:
                process.stdin.write(b"Q\n")
                process.stdin.close()
                process.wait(timeout=CLOSE_WAIT_S)
            except Exception:
                process.kill()
                process.wait()
            finally:
                process.stdout.close()
