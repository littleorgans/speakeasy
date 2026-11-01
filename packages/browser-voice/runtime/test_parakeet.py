"""Deterministic Parakeet contract checks; no installed model runtime is imported."""

import array
import contextlib
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import parakeet


ROOT = Path(__file__).resolve().parent
SETUP_IDS = ("mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy")


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.now = 0.0
        self.text = "  launch, Claude\n"
        self.sample_rate = 16000
        self.mel_shape = (1, 4, 128)
        self.mel_dtype = "mlx.core.float32"
        self.weight_dtype = "mlx.core.bfloat16"
        self.weight_count = 697
        self.result_count = 1
        self.metal_available = True
        self.gpu_available = True
        self.stream_device = "gpu"
        self.active_stream = False
        self.received_pcm = None
        self.loaded_paths = []
        self.configurations = []
        self.pending_gpu = False
        self.mx = types.ModuleType("mlx.core")
        self.mx.gpu = "gpu"
        self.mx.float32 = "mlx.core.float32"
        self.mx.metal = types.SimpleNamespace(is_available=lambda: self.metal_available)
        self.mx.is_available = lambda device: device == "gpu" and self.gpu_available
        self.mx.set_default_device = lambda device: self.record("device", device)
        self.mx.default_device = lambda: "gpu"
        self.mx.default_stream = lambda device: types.SimpleNamespace(device=self.stream_device)
        self.mx.stream = self.stream
        self.mx.array = self.audio
        self.mx.eval = lambda parameters: self.record("eval", parameters)
        self.mx.synchronize = self.synchronize
        self.mx.random = types.SimpleNamespace(seed=lambda seed: self.record("seed", seed))
        self.np = types.ModuleType("numpy")
        self.np.frombuffer = self.frombuffer
        self.np.isfinite = lambda data: types.SimpleNamespace(
            all=lambda: all(math.isfinite(value) for value in data.values)
        )
        self.api = types.ModuleType("parakeet_mlx")
        self.api.from_pretrained = self.from_pretrained
        self.api.Greedy = lambda: types.SimpleNamespace(kind="stock-greedy")
        self.api.DecodingConfig = self.decoding_config
        self.audio_api = types.ModuleType("parakeet_mlx.audio")
        self.audio_api.get_logmel = self.get_logmel
        self.utils = types.ModuleType("mlx.utils")
        self.utils.tree_flatten = lambda parameters: [
            (str(index), types.SimpleNamespace(dtype=self.weight_dtype))
            for index in range(self.weight_count)
        ]

    def record(self, *event):
        self.events.append(event)

    def clock(self):
        self.record("clock")
        return self.now

    @contextlib.contextmanager
    def stream(self, stream):
        self.active_stream = True
        self.record("enter-stream", stream.device)
        try:
            yield
        finally:
            self.active_stream = False
            self.record("exit-stream")

    def synchronize(self, stream):
        if not self.active_stream:
            raise AssertionError("synchronize must use the active GPU stream")
        self.record("synchronize", stream.device)
        self.now += 0.019 if self.pending_gpu else 0.007
        self.pending_gpu = False

    def frombuffer(self, samples, *, dtype):
        if dtype != "<f4":
            raise AssertionError("input must remain little endian Float32")
        self.record("frombuffer")
        self.now += 0.5  # Transport/view work must stay outside processingMs.
        self.received_pcm = samples.tobytes()
        values = [value for (value,) in struct.iter_unpack("<f", self.received_pcm)]
        return types.SimpleNamespace(values=values, shape=(len(values),))

    def audio(self, data, *, dtype):
        if dtype != self.mx.float32 or not self.active_stream:
            raise AssertionError("MLX audio must be Float32 on the GPU stream")
        self.record("array", tuple(data.values), dtype)
        self.now += 0.002
        return types.SimpleNamespace(shape=data.shape, dtype=dtype)

    def from_pretrained(self, path):
        self.record("from-pretrained", path)
        self.loaded_paths.append(path)
        self.now += 0.2
        return types.SimpleNamespace(
            parameters=lambda: "model-parameters",
            preprocessor_config=types.SimpleNamespace(sample_rate=self.sample_rate),
            generate=self.generate,
        )

    def decoding_config(self, *, decoding):
        self.record("decoding-config", decoding.kind)
        config = types.SimpleNamespace(decoding=decoding)
        self.configurations.append(config)
        return config

    def get_logmel(self, audio, preprocessor):
        self.record("get-logmel", audio.shape, preprocessor.sample_rate)
        self.now += 0.003
        return types.SimpleNamespace(shape=self.mel_shape, dtype=self.mel_dtype)

    def generate(self, mel, *, decoding_config):
        if decoding_config is not self.configurations[0]:
            raise AssertionError("stock greedy configuration must be reused")
        self.record("generate")
        self.now += 0.005
        self.pending_gpu = True
        return [types.SimpleNamespace(text=self.text) for _ in range(self.result_count)]

    @contextlib.contextmanager
    def installed(self):
        modules = {
            "numpy": self.np,
            "mlx": types.ModuleType("mlx"),
            "mlx.core": self.mx,
            "mlx.utils": self.utils,
            "parakeet_mlx": self.api,
            "parakeet_mlx.audio": self.audio_api,
        }
        versions = {"parakeet-mlx": "0.5.3", "mlx": "0.32.4",
                    "mlx-metal": "0.32.4", "numpy": "2.5.4"}
        with patch.dict(sys.modules, modules), \
                patch.object(parakeet.platform, "python_version", return_value="3.12.5"), \
                patch.object(parakeet.importlib.metadata, "version", side_effect=versions.__getitem__), \
                patch.object(parakeet.time, "perf_counter", side_effect=self.clock), \
                patch.object(parakeet, "_validate_snapshot", return_value={"verified": True}):
            yield


class BackendTests(unittest.TestCase):
    def backend(self, setup_id=SETUP_IDS[0]):
        return parakeet.Backend({"setupId": setup_id, "modelPath": "/local/pinned/snapshot"})

    def test_module_and_constructor_are_lazy(self):
        script = """
import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mlx', 'parakeet_mlx'}:
            raise AssertionError('eager native import: ' + fullname)
sys.meta_path.insert(0, BlockNative())
sys.path.insert(0, sys.argv[1])
import parakeet
parakeet.Backend({'setupId': 'mlx-parakeet/v3-greedy', 'modelPath': '/missing/snapshot'})
"""
        result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_options_require_exact_preset_and_absolute_path(self):
        for options in [None, {}, {"setupId": "v3", "modelPath": "/local"},
                        {"setupId": SETUP_IDS[0], "modelPath": "model/repo"},
                        {"setupId": SETUP_IDS[0], "modelPath": "/local", "dtype": "float16"}]:
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                parakeet.Backend(options)

    def test_both_presets_load_once_without_warmup(self):
        expected = [
            ("mlx-community/parakeet-tdt-0.6b-v3", "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15", "F32"),
            ("selcukkubur/parakeet-ultra-mlx", "b554592c50b2a48471add2daa3d46fa9f00fef5e", "BF16"),
        ]
        for setup_id, (repo, revision, disk_dtype) in zip(SETUP_IDS, expected):
            with self.subTest(setup_id=setup_id):
                runtime = FakeRuntime()
                with runtime.installed():
                    backend = self.backend(setup_id)
                    metadata = backend.load()
                    self.assertEqual(backend.load(), metadata)
                self.assertEqual(runtime.loaded_paths, ["/local/pinned/snapshot"])
                self.assertEqual((metadata["setupId"], metadata["repo"], metadata["revision"]),
                                 (setup_id, repo, revision))
                self.assertEqual(metadata["onDiskDtypes"], [disk_dtype])
                self.assertEqual(metadata["weightDtypes"], ["mlx.core.bfloat16"])
                self.assertEqual(metadata["weightTensorCount"], 697)
                self.assertEqual(metadata["seedBoundary"], "request")
                self.assertIsNone(metadata["warmupMs"])
                self.assertAlmostEqual(metadata["loadMs"], 207.0)
                self.assertNotIn("generate", [event[0] for event in runtime.events])
                json.dumps(metadata, allow_nan=False)

    def test_snapshot_validation_precedes_native_imports(self):
        with patch.object(parakeet, "_validate_snapshot", side_effect=ValueError("bad snapshot")), \
                patch.object(parakeet.importlib.metadata, "version") as version:
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                self.backend().load()
            version.assert_not_called()

    def test_load_rejects_runtime_and_model_drift(self):
        for field, value in [("metal_available", False), ("gpu_available", False),
                             ("stream_device", "cpu"), ("sample_rate", 8000),
                             ("weight_dtype", "mlx.core.float32"), ("weight_count", 696)]:
            runtime = FakeRuntime()
            setattr(runtime, field, value)
            with self.subTest(field=field), runtime.installed(), self.assertRaises(ValueError):
                self.backend().load()
        runtime = FakeRuntime()
        with runtime.installed(), patch.object(parakeet.platform, "python_version", return_value="3.13.2"):
            with self.assertRaisesRegex(ValueError, "Python"):
                self.backend().load()
        with runtime.installed(), patch.object(parakeet.importlib.metadata, "version", return_value="newer"):
            with self.assertRaisesRegex(ValueError, "version"):
                self.backend().load()

    def test_exact_audio_stock_calls_seed_and_synchronized_timer(self):
        raw = bytes.fromhex("0000803f0000008000000040000080bf01000000")
        storage = array.array("f", [99.0])
        storage.frombytes(raw)
        storage.append(99.0)
        samples = memoryview(storage)[1:-1]
        for setup_id in SETUP_IDS:
            runtime = FakeRuntime()
            with self.subTest(setup_id=setup_id), runtime.installed():
                backend = self.backend(setup_id)
                backend.load()
                runtime.events.clear()
                result = backend.transcribe(samples)
                second = backend.transcribe(samples)
            self.assertEqual(runtime.received_pcm, raw)
            self.assertEqual(result["text"], runtime.text)
            self.assertAlmostEqual(result["processingMs"], 29.0)
            self.assertAlmostEqual(second["processingMs"], 29.0)
            self.assertEqual(result["diagnostics"]["pcmSha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["diagnostics"]["sampleCount"], 5)
            json.dumps(result, allow_nan=False)
            names = [event[0] for event in runtime.events]
            first = names[:names.index("exit-stream") + 1]
            self.assertEqual(first, ["frombuffer", "enter-stream", "seed", "synchronize", "clock",
                                     "array", "get-logmel", "generate", "synchronize", "clock", "exit-stream"])
            self.assertEqual([event for event in runtime.events if event[0] == "seed"],
                             [("seed", 0), ("seed", 0)])

    def test_empty_output_is_success_and_no_text_normalization(self):
        for text in ["", " \n", "{\"text\":\"été\"}\n"]:
            runtime = FakeRuntime()
            runtime.text = text
            with self.subTest(text=text), runtime.installed():
                backend = self.backend()
                backend.load()
                self.assertEqual(backend.transcribe(array.array("f", [0.0]))["text"], text)

    def test_transcribe_requires_explicit_load_and_finite_float32_view(self):
        with self.assertRaisesRegex(RuntimeError, "load"):
            self.backend().transcribe(array.array("f", [0.0]))
        runtime = FakeRuntime()
        with runtime.installed():
            backend = self.backend()
            backend.load()
            for samples in [array.array("d", [0.0]), b"\0\0\0\0", [0.0],
                            array.array("f"), array.array("f", [float("nan")]),
                            array.array("f", [float("inf")]),
                            memoryview(array.array("f", [1.0, 2.0, 3.0]))[::2]]:
                with self.subTest(kind=type(samples).__name__), self.assertRaises((TypeError, ValueError)):
                    backend.transcribe(samples)
        self.assertNotIn("generate", [event[0] for event in runtime.events])

    def test_rejects_mel_and_result_contract_drift_without_retry(self):
        for field, value in [("mel_shape", (1, 4, 80)), ("mel_dtype", "mlx.core.float16"),
                             ("result_count", 0), ("result_count", 2), ("text", None)]:
            runtime = FakeRuntime()
            setattr(runtime, field, value)
            with self.subTest(field=field, value=value), runtime.installed():
                backend = self.backend()
                backend.load()
                with self.assertRaises(ValueError):
                    backend.transcribe(array.array("f", [0.0]))
            self.assertLessEqual([event[0] for event in runtime.events].count("generate"), 1)

    def test_generate_failure_propagates_once_without_fallback(self):
        runtime = FakeRuntime()
        with runtime.installed():
            backend = self.backend()
            backend.load()
            with patch.object(backend._model, "generate", side_effect=RuntimeError("decode failed")) as generate:
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    backend.transcribe(array.array("f", [0.0]))
            self.assertEqual(generate.call_count, 1)


class SnapshotTests(unittest.TestCase):
    def test_pinned_files_must_match_bytes_and_digest_and_no_extra_weights(self):
        with tempfile.TemporaryDirectory(prefix="speakeasy-parakeet-test-") as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"config")
            (root / "model.safetensors").write_bytes(b"weights")
            pinned = {name: (len(data), hashlib.sha256(data).hexdigest())
                      for name, data in [("config.json", b"config"), ("model.safetensors", b"weights")]}
            spec = {"files": pinned}
            metadata = parakeet._validate_snapshot(root, spec)
            self.assertEqual(metadata["config.json"]["sha256"], pinned["config.json"][1])
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model.safetensors"):
                parakeet._validate_snapshot(root, spec)
            (root / "model.safetensors").write_bytes(b"weight")
            with self.assertRaisesRegex(ValueError, "model.safetensors"):
                parakeet._validate_snapshot(root, spec)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                parakeet._validate_snapshot(root, spec)
            (root / "extra.safetensors").unlink()
            (root / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "config.json"):
                parakeet._validate_snapshot(root, spec)


if __name__ == "__main__":
    unittest.main()
