"""Pinned Qwen3 speech.cpp bridge, derived from known-words/qwen/native.py.

Backend options contain absolute libraryPath and modelPath, plus an optional
setupId fixed to qwen/context. Construction does not load native code.
transcribe accepts a contiguous Float32 buffer;
transport validates its declared PCM hash, and this adapter reports the hash
of the exact bytes handed to the C API.
"""

import array
from contextlib import contextmanager
import copy
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time


SETUP_ID = "qwen/context"
MODEL_ID = "sakasegawa/Qwen3-ASR-0.6B-GGUF"
MODEL_REVISION = "ee78f8709547f39bea3b924a16c72403826cac92"
MODEL_FILE = "Qwen3-ASR-0.6B-Q8_0.gguf"
MODEL_SHA256 = "416e10c15b4a3d9002bd337d18fc450233fdf68502b6e10d1379d2789838afd0"
LIBRARY_SHA256 = "e3fac058886bc215cb237a6745fced4d1555263808569998a42478fc889d85b6"
RUNTIME_VERSION = "0.10.0"
RUNTIME_COMMIT = "06590811a90041f935a470ba0e2b85346d2fc041"
PYTHON_VERSION = "3.13.2"
DEVICE = b"gpu"
THREADS = 2
WARMUP = 0
LANGUAGE = b"auto"
PROMPT = "launch, claude, grok"
SAMPLE_RATE = 16000
PROCESSING_TIMING = (
    "perf_counter around request creation, options, audio copy and "
    "speech_transcribe; excludes PCM validation/copy/hash, result reading, "
    "request release, load and warmup"
)

POINTER = C.c_void_p
SIGNATURES = {
    "speech_version": (C.c_char_p, []),
    "speech_last_error": (C.c_char_p, []),
    "speech_option_from_name": (C.c_int, [C.c_char_p, C.POINTER(C.c_int)]),
    "speech_load_params_new": (C.c_int, [C.POINTER(POINTER)]),
    "speech_load_params_free": (None, [POINTER]),
    "speech_load_params_set_device": (C.c_int, [POINTER, C.c_char_p]),
    "speech_load_params_set_threads": (C.c_int, [POINTER, C.c_int]),
    "speech_load_params_set_warmup": (C.c_int, [POINTER, C.c_int]),
    "speech_model_load": (C.c_int, [C.c_char_p, POINTER, C.POINTER(POINTER)]),
    "speech_model_free": (None, [POINTER]),
    "speech_model_get_info": (C.c_int, [POINTER, C.POINTER(POINTER)]),
    "speech_model_info_free": (None, [POINTER]),
    "speech_model_info_json": (C.c_char_p, [POINTER]),
    "speech_model_info_takes": (C.c_int, [POINTER, C.c_int]),
    "speech_model_info_option_steers": (C.c_int, [POINTER, C.c_int]),
    "speech_request_new": (C.c_int, [POINTER, C.POINTER(POINTER)]),
    "speech_request_free": (None, [POINTER]),
    "speech_request_set_audio": (
        C.c_int, [POINTER, C.POINTER(C.c_float), C.c_size_t, C.c_int]
    ),
    "speech_request_set_string": (C.c_int, [POINTER, C.c_int, C.c_char_p]),
    "speech_transcribe": (C.c_int, [POINTER]),
    "speech_request_result": (POINTER, [POINTER]),
    "speech_result_text": (C.c_char_p, [POINTER]),
    "speech_result_stop": (C.c_int, [POINTER]),
}


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _audio(samples):
    """Own a byte identical native buffer without numeric conversion."""
    view = memoryview(samples)
    if (sys.byteorder != "little" or C.sizeof(C.c_float) != 4
            or view.itemsize != 4 or view.format not in ("f", "<f", "=f")
            or view.ndim != 1 or not view.c_contiguous):
        raise ValueError("Qwen requires contiguous little endian Float32 samples")
    payload = view.cast("B")
    pcm_sha256 = hashlib.sha256(payload).hexdigest()
    values = array.array("f")
    values.frombytes(payload)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Qwen samples must be finite")
    native = (C.c_float * len(values)).from_buffer(values)
    if hashlib.sha256(memoryview(native).cast("B")).hexdigest() != pcm_sha256:
        raise ValueError("Qwen native PCM identity mismatch")
    return values, native, pcm_sha256


class Backend:
    def __init__(self, options):
        required = {"libraryPath", "modelPath"}
        if (not isinstance(options, dict) or not required <= set(options)
                or set(options) - required - {"setupId"}):
            raise ValueError("Qwen options require libraryPath, modelPath and optional setupId")
        if options.get("setupId", SETUP_ID) != SETUP_ID:
            raise ValueError("Qwen backend requires qwen/context")
        for name in ("libraryPath", "modelPath"):
            value = options[name]
            if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
                raise ValueError(f"Qwen {name} must be an absolute path")
        self.library_path = Path(options["libraryPath"])
        self.model_path = Path(options["modelPath"])
        self.lib = None
        self.model = POINTER()
        self.options = {}
        self.metadata = None

    def _check(self, status):
        if status != 0:
            error = self.lib.speech_last_error()
            detail = error.decode("utf-8", errors="replace") if error else "no error detail"
            raise RuntimeError(f"speech status {status}: {detail}")

    @contextmanager
    def _handle(self, create, free, *args):
        handle = POINTER()
        try:
            self._check(create(*args, C.byref(handle)))
            if not handle:
                raise RuntimeError("speech returned a null handle")
            yield handle
        finally:
            if handle:
                free(handle)

    def _bind(self):
        self.lib = C.CDLL(str(self.library_path))
        for name, (result, args) in SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = result, args
        if self.lib.speech_version() != RUNTIME_VERSION.encode():
            raise RuntimeError(f"Qwen requires speech.cpp {RUNTIME_VERSION}")
        self.options = {}
        for name in ("language", "prompt"):
            option = C.c_int()
            self._check(self.lib.speech_option_from_name(name.encode(), C.byref(option)))
            self.options[name] = option.value

    def load(self):
        if self.metadata is not None:
            return copy.deepcopy(self.metadata)
        if platform.python_version() != PYTHON_VERSION:
            raise RuntimeError(f"Qwen requires Python {PYTHON_VERSION}")
        model_sha256 = _digest(self.model_path)
        library_sha256 = _digest(self.library_path)
        if model_sha256 != MODEL_SHA256 or library_sha256 != LIBRARY_SHA256:
            raise ValueError("Qwen model or speech.cpp library SHA256 mismatch")
        try:
            self._bind()
            with self._handle(self.lib.speech_load_params_new,
                              self.lib.speech_load_params_free) as params:
                self._check(self.lib.speech_load_params_set_device(params, DEVICE))
                self._check(self.lib.speech_load_params_set_threads(params, THREADS))
                self._check(self.lib.speech_load_params_set_warmup(params, WARMUP))
                started = time.perf_counter()
                self._check(self.lib.speech_model_load(
                    str(self.model_path).encode(), params, C.byref(self.model)))
                load_ms = (time.perf_counter() - started) * 1000
                if not self.model:
                    raise RuntimeError("speech returned a null model")
            with self._handle(self.lib.speech_model_get_info,
                              self.lib.speech_model_info_free, self.model) as info:
                prompt = self.options["prompt"]
                if (self.lib.speech_model_info_takes(info, prompt) != 1
                        or self.lib.speech_model_info_option_steers(info, prompt) != 1):
                    raise RuntimeError("Qwen model must accept and steer the prompt option")
                model_info = json.loads(self.lib.speech_model_info_json(info))
                if not isinstance(model_info, dict):
                    raise ValueError("speech model info must be a JSON object")
            self.metadata = {
                "setupId": SETUP_ID, "loadMs": load_ms,
                "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
                "modelFile": MODEL_FILE, "modelSha256": model_sha256,
                "runtime": "speech.cpp", "runtimeVersion": RUNTIME_VERSION,
                "runtimeCommit": RUNTIME_COMMIT, "librarySha256": library_sha256,
                "pythonVersion": PYTHON_VERSION, "deviceRequested": DEVICE.decode(),
                "threads": THREADS, "warmup": False, "language": LANGUAGE.decode(),
                "prompt": PROMPT, "modelInfo": model_info,
                "loadTiming": "speech_model_load only",
                "processingTiming": PROCESSING_TIMING,
            }
            return copy.deepcopy(self.metadata)
        except BaseException:
            self.close()
            raise

    def transcribe(self, samples):
        if not self.model or self.metadata is None:
            raise RuntimeError("Qwen backend is not loaded")
        values, native, pcm_sha256 = _audio(samples)
        started = time.perf_counter()
        with self._handle(self.lib.speech_request_new,
                          self.lib.speech_request_free, self.model) as request:
            self._check(self.lib.speech_request_set_string(
                request, self.options["language"], LANGUAGE))
            self._check(self.lib.speech_request_set_string(
                request, self.options["prompt"], PROMPT.encode()))
            self._check(self.lib.speech_request_set_audio(
                request, native, len(values), SAMPLE_RATE))
            decode_started = time.perf_counter()
            self._check(self.lib.speech_transcribe(request))
            finished = time.perf_counter()
            result = self.lib.speech_request_result(request)
            if not result:
                raise RuntimeError("speech returned a null result")
            text = self.lib.speech_result_text(result)
            if text is None:
                raise RuntimeError("speech returned null text")
            return {
                "text": text.decode(), "processingMs": (finished - started) * 1000,
                "decodeMs": (finished - decode_started) * 1000,
                "stop": self.lib.speech_result_stop(result),
                "diagnostics": {"sampleCount": len(values), "pcmSha256": pcm_sha256,
                                "processingTiming": PROCESSING_TIMING},
            }

    def close(self):
        if self.model:
            self.lib.speech_model_free(self.model)
            self.model = POINTER()
        self.metadata = None
