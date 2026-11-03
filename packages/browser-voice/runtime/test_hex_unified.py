"""Standard library bridge tests with a scripted worker. No model or Metal runtime is loaded."""

import array
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import hex_unified as hexu
import worker


FAKE_WORKER = """#!{python}
import json, struct, sys
print(json.dumps({{"kind": "ready", "backend": "metal", "device": "MTL0", "variant": "{variant}", "loadMs": 5}}), flush=True)
fed = 0
while True:
    head = sys.stdin.buffer.read(5)
    if len(head) < 5: break
    op, count = head[:1], struct.unpack("<I", head[1:])[0]
    samples = struct.unpack("<%df" % count, sys.stdin.buffer.read(count * 4))
    if op == b"T":
        print(json.dumps({{"kind": "result", "text": "heard %d" % len(samples), "rawText": " heard", "processingMs": 2.5,
                          "pauseDroppedSamples": 0}}), flush=True)
    else:
        fed = (0 if op == b"S" else fed) + len(samples)
        print(json.dumps({{"kind": "partial", "stable": "fed %d" % fed, "tentative": " so far", "processingMs": 1.5}}), flush=True)
"""


class HexUnifiedTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.model = self.root / hexu.MODEL_FILE
        self.model.write_bytes(b"fake gguf")
        self.addCleanup(shutil.rmtree, self.root)

    def runner(self, variant="unified-en-0.6b"):
        path = self.root / "worker"
        path.write_text(FAKE_WORKER.format(python=sys.executable, variant=variant))
        os.chmod(path, 0o755)
        return path

    def backend(self, runner):
        return hexu.Backend({"runnerPath": str(runner), "modelPath": str(self.model), "setupId": "hex/unified"})

    def test_worker_routes_the_setup_to_this_backend(self):
        self.assertEqual(worker.BACKENDS["hex/unified"], "hex_unified")

    def test_options_require_absolute_paths_and_this_setup(self):
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "relative", "modelPath": str(self.model)})
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "/w", "modelPath": "/m", "setupId": "qwen/context"})
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "/w", "modelPath": "/m", "libraryPath": "/l"})

    def test_load_checks_the_model_hash_before_starting_the_worker(self):
        backend = self.backend(self.runner())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            backend.load()
        self.assertIsNone(backend.process)

    def test_load_transcribe_and_close_through_the_worker(self):
        backend = self.backend(self.runner())
        with patch.object(hexu, "MODEL_SHA256", hashlib.sha256(b"fake gguf").hexdigest()):
            metadata = backend.load()
        self.assertEqual(metadata["setupId"], "hex/unified")
        self.assertEqual(metadata["device"], "MTL0")
        samples = memoryview(array.array("f", [0.1, -0.2, 0.3]).tobytes()).cast("f")
        result = backend.transcribe(samples)
        self.assertEqual(result["text"], "heard 3")
        self.assertEqual(result["rawText"], " heard")
        self.assertEqual(result["processingMs"], 2.5)
        worker.validate_result(result)
        process = backend.process
        backend.close()
        self.assertEqual(process.returncode, 0)
        self.assertIsNone(backend.process)

    def test_feed_streams_live_words_and_start_opens_a_fresh_stream(self):
        backend = self.backend(self.runner())
        with patch.object(hexu, "MODEL_SHA256", hashlib.sha256(b"fake gguf").hexdigest()):
            backend.load()
        samples = lambda count: memoryview(array.array("f", [0.1] * count).tobytes()).cast("f")
        first = backend.feed(samples(3), True)
        self.assertEqual(first, {"stable": "fed 3", "tentative": " so far", "processingMs": 1.5})
        worker.validate_partial(first)
        self.assertEqual(backend.feed(samples(2), False)["stable"], "fed 5")
        self.assertEqual(backend.transcribe(samples(4))["text"], "heard 4", "a take follows a live stream")
        self.assertEqual(backend.feed(samples(1), True)["stable"], "fed 1")
        process = backend.process
        backend.close()
        self.assertEqual(process.returncode, 0)

    def test_identity_mismatch_retires_the_worker(self):
        backend = self.backend(self.runner(variant="other"))
        with patch.object(hexu, "MODEL_SHA256", hashlib.sha256(b"fake gguf").hexdigest()):
            with self.assertRaisesRegex(RuntimeError, "identity"):
                backend.load()
        self.assertIsNone(backend.process)


if __name__ == "__main__":
    unittest.main()
