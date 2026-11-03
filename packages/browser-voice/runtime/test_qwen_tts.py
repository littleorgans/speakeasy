"""Qwen3 TTS bridge over a scripted speech.cpp worker: launch flags, chunks, end and errors."""

import array
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest

import qwen_tts


# Answers like `speech worker`: ready, then per synthesize two 16 bit PCM chunks and an end, or an error.
FAKE_SPEECH = textwrap.dedent("""\
    #!{python}
    import base64, json, struct, sys
    print(json.dumps({{"type": "ready", "args": sys.argv[1:]}}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["text"] == "fail":
            print(json.dumps({{"type": "error", "id": request["id"]}}), flush=True); continue
        print(json.dumps({{"type": "chunk", "id": "other", "pcm": ""}}), flush=True)
        for value in (16384, -32768):
            pcm = base64.b64encode(struct.pack("<h", value)).decode()
            print(json.dumps({{"type": "chunk", "id": request["id"], "pcm": pcm}}), flush=True)
        print(json.dumps({{"type": "end", "id": request["id"], "stop": "complete"}}), flush=True)
""")


class QwenTtsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / qwen_tts.MODEL_FILE).write_bytes(b"gguf")
        self.runner = self.root / "speech"
        self.runner.write_text(FAKE_SPEECH.format(python=sys.executable))
        os.chmod(self.runner, 0o755)
        self.files = [{"name": qwen_tts.MODEL_FILE, "bytes": 4, "sha256": hashlib.sha256(b"gguf").hexdigest()}]

    def backend(self):
        return qwen_tts.Backend({"setupId": "qwen3-tts-0.6b", "modelPath": str(self.root), "files": self.files,
                                 "runnerPath": str(self.runner)})

    def test_speak_gathers_its_own_chunks_as_float_pcm_until_the_end(self):
        backend = self.backend()
        self.assertEqual(backend.load()["device"], "gpu")
        result = backend.speak("Hello.", "ryan", 1)
        self.assertEqual(array.array("f", result["samples"]).tolist(), [0.5, -1.0])
        self.assertEqual(result["sampleRate"], 24_000)
        with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
            backend.speak("fail", "aiden", 1)
        self.assertEqual(len(array.array("f", backend.speak("Again.", "aiden", 1)["samples"])), 2,
                         "the worker serves the next request after an error")
        process = backend._model
        backend.close()
        self.assertEqual(process.returncode, 0)

    def test_load_attests_the_payload_before_starting_the_worker(self):
        (self.root / qwen_tts.MODEL_FILE).write_bytes(b"nope")
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            backend.load()
        self.assertIsNone(backend._model)


if __name__ == "__main__":
    unittest.main()
