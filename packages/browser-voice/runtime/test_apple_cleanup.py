"""Standard library bridge tests with a scripted runner. No Foundation Models runtime is loaded."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import apple_cleanup
import pins
import worker


def setUpModule():
    """Runner hashes cache in a temporary directory, never the owner's cache."""
    directory = tempfile.TemporaryDirectory(prefix="speakeasy-digest-test-")
    unittest.addModuleCleanup(directory.cleanup)
    cache = patch.object(pins, "DIGEST_CACHE", Path(directory.name))
    cache.start()
    unittest.addModuleCleanup(cache.stop)


# ready: the ready line's fields, or an unavailable kind; error: the kind every take answers with.
FAKE_RUNNER = """#!{python}
import json, sys
ready, error = {ready!r}, {error!r}
if isinstance(ready, str):
    print(json.dumps({{"kind": "error", "error": "unavailable", "reason": ready}}), flush=True)
    sys.exit(1)
print(json.dumps({{"kind": "ready", "contextSize": 4096, "loadMs": 3, **ready}}), flush=True)
for line in sys.stdin:
    if error:
        print(json.dumps({{"kind": "error", "error": error, "processingMs": 1}}), flush=True)
    else:
        print(json.dumps({{"kind": "result", "text": " " + json.dumps(json.loads(line), sort_keys=True) + "\\n", "processingMs": 2.5}}), flush=True)
"""
IDENTITY = {"backend": "apple", "variant": "foundation-models"}


class AppleCleanupTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def backend(self, ready=IDENTITY, error=None):
        path = self.root / "apple-cleanup-worker"
        path.write_text(FAKE_RUNNER.format(python=sys.executable, ready=ready, error=error))
        os.chmod(path, 0o755)
        backend = apple_cleanup.Backend({"runnerPath": str(path), "setupId": apple_cleanup.SETUP_ID})
        self.addCleanup(backend.close)
        return backend

    def test_worker_routes_the_model_to_this_bridge_which_takes_context(self):
        self.assertEqual(worker.BACKENDS[apple_cleanup.SETUP_ID], "apple_cleanup")
        self.assertIn(apple_cleanup.SETUP_ID, worker.TEXT_CLEANUPS)
        self.assertTrue(apple_cleanup.Backend.takes_context)

    def test_options_require_an_absolute_runner_for_this_model(self):
        for options in ({"runnerPath": "relative"}, {"runnerPath": "/r", "setupId": "apple/speech"},
                        {"runnerPath": "/r", "modelPath": "/m"}, None):
            with self.subTest(options=options), self.assertRaises(ValueError):
                apple_cleanup.Backend(options)

    def test_ready_identity_is_checked(self):
        backend = self.backend()
        ready = backend.load()
        self.assertEqual(ready["setupId"], apple_cleanup.SETUP_ID)
        self.assertEqual(ready["contextSize"], 4096)
        self.assertEqual(len(ready["runnerSha256"]), 64)
        process = backend.process
        self.assertEqual(backend.load(), ready)
        self.assertIs(backend.process, process, "a second load reuses the runner")
        for wrong in ({"backend": "apple", "variant": "speech-transcriber"}, {"backend": "metal", "variant": "foundation-models"}):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(RuntimeError, "^load: Apple cleanup worker identity mismatch$"):
                self.backend(ready=wrong).load()

    def test_each_take_is_one_request_line_with_its_fields(self):
        backend = self.backend()
        backend.load()
        result = backend.cleanup("um tell shawn", instruction="Clean it.", names=["Sean"], earlier=["Before."])
        self.assertEqual(json.loads(result["text"]),
                         {"text": "um tell shawn", "instruction": "Clean it.", "names": ["Sean"], "earlier": ["Before."]})
        self.assertEqual(result["processingMs"], 2.5)
        self.assertEqual(json.loads(backend.cleanup("again")["text"]),
                         {"text": "again", "instruction": "", "names": [], "earlier": []})
        self.assertEqual(backend.cleanup("  ")["text"], "", "blank text never reaches the model")

    def test_each_unavailable_kind_maps_to_its_reason_text(self):
        for kind, reason in (("appleIntelligenceNotEnabled", "Apple Intelligence is off"),
                             ("modelNotReady", "model not ready, it may still be downloading"),
                             ("deviceNotEligible", "this Mac is not eligible"),
                             ("unknown", "Apple Foundation Models is unavailable")):
            backend = self.backend(ready=kind)
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, f"^unavailable: {reason}$"):
                backend.load()
            self.assertIsNone(backend.process, "a failed load leaves no runner")

    def test_errors_carry_the_kind_only_as_fixed_text(self):
        for kind, message in (("contextSizeExceeded", "limit: the take is too long for the model's context"),
                              ("guardrailViolation", "execution: the model's guardrails refused the take"),
                              ("system", "execution: the model failed"),
                              ("something new", "execution: the model failed")):
            backend = self.backend(error=kind)
            backend.load()
            with self.subTest(kind=kind), self.assertRaises(RuntimeError) as raised:
                backend.cleanup("um secret take text")
            self.assertEqual(str(raised.exception), message)
            self.assertNotIn("secret", str(raised.exception))

    def test_close_ends_the_runner(self):
        backend = self.backend()
        backend.load()
        process = backend.process
        backend.close()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(backend.metadata)


if __name__ == "__main__":
    unittest.main()
