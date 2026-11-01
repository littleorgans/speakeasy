"""Pinned local Parakeet computation from the retained mlx-parakeet measure runner.

Backend options are exactly setupId and an absolute local modelPath. Construction
is lazy; load is explicit and has no warmup. Transport owns finite mono 16 kHz
little endian Float32 ingress and process scheduling. This adapter also checks
the buffer contract before passing its exact view to MLX.
"""

import copy
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import time


PYTHON_VERSION = "3.12.5"
RUNTIME_VERSIONS = {
    "parakeet-mlx": "0.5.3",
    "mlx": "0.32.4",
    "mlx-metal": "0.32.4",
    "numpy": "2.5.4",
}
WEIGHT_TENSOR_COUNT = 697
LOADED_WEIGHT_DTYPES = ["mlx.core.bfloat16"]
SEED = 0

# Byte sizes and SHA256 from the frozen mlx-parakeet/artifacts.json. Shared
# preprocessing/tokenizer bytes are identical for both measured snapshots.
COMMON_FILES = {
    "config.json": (244093, "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c"),
    "tokenizer.model": (360916, "eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e"),
    "tokenizer.vocab": (101024, "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17"),
    "vocab.txt": (46772, "3cde1409fd78783a79b29ed4d32da57c746993856f7c8263bcb905d2e5839db7"),
}
PRESETS = {
    "mlx-parakeet/v3-greedy": {
        "repo": "mlx-community/parakeet-tdt-0.6b-v3",
        "revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
        "onDiskDtypes": ["F32"],
        "files": {
            **COMMON_FILES,
            "model.safetensors": (2508288736, "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592"),
            "README.md": (1081, "724e3ba46903986ff6caeb8fddb7495c06b5a7a8d908f11e43cc0a7e136b5e23"),
        },
    },
    "mlx-parakeet/ultra-greedy": {
        "repo": "selcukkubur/parakeet-ultra-mlx",
        "revision": "b554592c50b2a48471add2daa3d46fa9f00fef5e",
        "onDiskDtypes": ["BF16"],
        "files": {
            **COMMON_FILES,
            "model.safetensors": (1254184462, "ee62d5a5f62f931b73ab000f07d864ee00b1e773205b1e666ff7990680c328b4"),
            "README.md": (3971, "7aa7eac9fbb09aeb4f85b174aad76b8efa91b149115f636b9a9dcc9fe4a03670"),
        },
    },
}


def _validate_snapshot(model_path, spec):
    """Attest local bytes without importing a native runtime or fetching assets."""
    if not model_path.is_dir():
        raise ValueError("Parakeet modelPath must be an existing local snapshot directory")
    if {path.name for path in model_path.glob("*.safetensors")} != {"model.safetensors"}:
        raise ValueError("Parakeet snapshot must contain only the pinned model.safetensors weights")
    verified = {}
    for name, (size, expected_digest) in spec["files"].items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Parakeet snapshot size mismatch: {name}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Parakeet snapshot SHA256 mismatch: {name}")
        verified[name] = {"bytes": size, "sha256": digest}
    return verified


def _runtime_identity():
    python_version = platform.python_version()
    if python_version != PYTHON_VERSION:
        raise ValueError(f"Parakeet requires Python {PYTHON_VERSION}")
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_VERSIONS}
    if versions != RUNTIME_VERSIONS:
        raise ValueError("Parakeet runtime version mismatch; use requirements-parakeet.txt")
    return {"python": python_version, **versions}


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath"}:
            raise ValueError("Parakeet options require exactly setupId and modelPath")
        setup_id = options["setupId"]
        if not isinstance(setup_id, str) or setup_id not in PRESETS:
            raise ValueError("Unsupported Parakeet setupId")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or not Path(model_path).is_absolute():
            raise ValueError("Parakeet modelPath must be absolute")
        self._setup_id = setup_id
        self._spec = PRESETS[setup_id]
        self._model_path = Path(model_path)
        self._metadata = None

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        # Complete pin attestation before from_pretrained can see this path.
        validation_start = time.perf_counter()
        files = _validate_snapshot(self._model_path, self._spec)
        runtime = _runtime_identity()
        validation_ms = (time.perf_counter() - validation_start) * 1000
        imported = time.perf_counter()
        import numpy as np
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from parakeet_mlx import from_pretrained, DecodingConfig, Greedy
        from parakeet_mlx.audio import get_logmel
        import_ms = (time.perf_counter() - imported) * 1000
        if not mx.metal.is_available() or not mx.is_available(mx.gpu):
            raise ValueError("Parakeet requires Metal and the MLX GPU device")
        mx.set_default_device(mx.gpu)
        gpu_stream = mx.default_stream(mx.gpu)
        if gpu_stream.device != mx.default_device():
            raise ValueError("Parakeet default stream must be on the GPU device")
        with mx.stream(gpu_stream):
            start = time.perf_counter()
            # Stock 0.5.3 default loads BF16, including the v3 F32 disk weights.
            model = from_pretrained(str(self._model_path))
            mx.eval(model.parameters())
            mx.synchronize(gpu_stream)
            load_ms = (time.perf_counter() - start) * 1000
            if model.preprocessor_config.sample_rate != 16000:
                raise ValueError("Parakeet preprocessing must use 16000 Hz")
            config = DecodingConfig(decoding=Greedy())
            weights = tree_flatten(model.parameters())
            dtypes = sorted({str(value.dtype) for _, value in weights})
            if len(weights) != WEIGHT_TENSOR_COUNT or dtypes != LOADED_WEIGHT_DTYPES:
                raise ValueError("Parakeet loaded weight count or dtype differs from the measured snapshot")
        self._np = np
        self._mx = mx
        self._model = model
        self._gpu_stream = gpu_stream
        self._decoding_config = config
        self._get_logmel = get_logmel
        self._metadata = {
            "setupId": self._setup_id,
            "repo": self._spec["repo"],
            "revision": self._spec["revision"],
            "files": files,
            "runtime": runtime,
            "validationMs": validation_ms,
            "importMs": import_ms,
            "loadMs": load_ms,
            "warmupMs": None,
            "metalAvailable": True,
            "gpuAvailable": True,
            "defaultDevice": str(mx.default_device()),
            "streamDevice": str(gpu_stream.device),
            "onDiskDtypes": self._spec["onDiskDtypes"],
            "weightTensorCount": len(weights),
            "weightDtypes": dtypes,
            "sampleRate": 16000,
            "features": 128,
            "decoding": "greedy",
            "attention": "stock-full",
            "seed": SEED,
            "seedBoundary": "request",
            "context": None,
            "hotwords": None,
            "chunking": None,
        }
        return copy.deepcopy(self._metadata)

    def transcribe(self, samples):
        if self._metadata is None:
            raise RuntimeError("Parakeet load() must complete before transcribe()")
        view = memoryview(samples)
        native_little_endian = sys.byteorder == "little" and view.format in {"f", "=f", "@f"}
        if (view.ndim != 1 or view.itemsize != 4 or not view.c_contiguous or not view.nbytes
                or not (native_little_endian or view.format == "<f")):
            raise ValueError("Parakeet samples must be a contiguous little endian Float32 view")
        data = self._np.frombuffer(view, dtype="<f4")
        if not self._np.isfinite(data).all():
            raise ValueError("Parakeet samples must be finite")
        pcm_sha256 = hashlib.sha256(view.cast("B")).hexdigest()
        mx = self._mx
        with mx.stream(self._gpu_stream):
            # The historical runner seeds each 20 take batch. A persistent
            # independent request uses seed 0 once here, outside processingMs.
            mx.random.seed(SEED)
            mx.synchronize(self._gpu_stream)
            start = time.perf_counter()
            audio = mx.array(data, dtype=mx.float32)
            mel = self._get_logmel(audio, self._model.preprocessor_config)
            if len(mel.shape) != 3 or mel.shape[0] != 1 or mel.shape[2] != 128:
                raise ValueError("Parakeet preprocessing must produce a single 128 feature mel")
            if str(mel.dtype) != "mlx.core.float32":
                raise ValueError("Parakeet mel dtype must remain Float32")
            results = self._model.generate(mel, decoding_config=self._decoding_config)
            mx.synchronize(self._gpu_stream)
            processing_ms = (time.perf_counter() - start) * 1000
        if len(results) != 1 or not isinstance(results[0].text, str):
            raise ValueError("Parakeet must return exactly one text result")
        return {
            "text": results[0].text,
            "processingMs": processing_ms,
            "diagnostics": {
                "sampleCount": view.nbytes // 4,
                "pcmSha256": pcm_sha256,
                "audioDtype": str(audio.dtype),
                "audioShape": list(audio.shape),
                "melShape": list(mel.shape),
                "melDtype": str(mel.dtype),
                "streamDevice": str(self._gpu_stream.device),
                "seed": SEED,
                "seedBoundary": "request",
            },
        }
