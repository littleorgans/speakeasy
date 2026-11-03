"""Qwen3 TTS bridge over a scripted speech.cpp worker: launch flags, requests, chunks, end and errors."""

import array
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest

import qwen_speechcpp


# Answers like `speech worker`: ready, then per synthesize two 16 bit PCM chunks and an end, or an error.
FAKE_SPEECH = textwrap.dedent("""\
    #!{python}
    import base64, json, os, struct, sys
    print(json.dumps({{"type": "ready", "args": sys.argv[1:]}}), flush=True)
    for line in sys.stdin:
        open(os.path.join(os.path.dirname(sys.argv[0]), "requests.jsonl"), "a").write(line)
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
        for spec in qwen_speechcpp.MODELS.values():
            (self.root / spec["file"]).write_bytes(b"gguf")
        self.runner = self.root / "speech"
        self.runner.write_text(FAKE_SPEECH.format(python=sys.executable))
        os.chmod(self.runner, 0o755)
        self.files = [{"name": qwen_speechcpp.MODELS["qwen3-tts-0.6b"]["file"], "bytes": 4, "sha256": hashlib.sha256(b"gguf").hexdigest()}]

    def backend(self, setup_id="qwen3-tts-0.6b"):
        model_file = qwen_speechcpp.MODELS[setup_id]["file"]
        files = [{**self.files[0], "name": model_file}]
        return qwen_speechcpp.Backend({"setupId": setup_id, "modelPath": str(self.root), "files": files,
                                       "runnerPath": str(self.runner)})

    def requests(self):
        return (self.root / "requests.jsonl").read_text().splitlines()

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

    def test_stock_settings_send_exactly_todays_request(self):
        backend = self.backend()
        backend.load()
        try:
            backend.speak("Hello.", "ryan", 1)
            backend.speak("Hello.", "ryan", 1, dict(qwen_speechcpp.DEFAULTS))
        finally:
            backend.close()
        today = '{"type": "synthesize", "id": "%s", "text": "Hello.", "voice": "ryan", "language": "en", "seed": 1}'
        self.assertEqual(self.requests(), [today % 1, today % 2], "byte identical to the request before settings")

    def test_settings_map_to_native_fields_and_greedy_omits_its_sampling(self):
        fields = qwen_speechcpp.native_fields
        self.assertEqual(fields({"language": "auto", "seed": 9, "maxSeconds": 10, "temperature": 0.7, "topP": 0.9,
                                 "topK": 10, "repetitionPenalty": 1.15, "codeTemperature": 0.5, "codeTopP": 0.8,
                                 "codeTopK": 10}, False),
                         {"seed": 9, "max_seconds": 10.0, "temperature": 0.7, "top_p": 0.9, "top_k": 10,
                          "repetition_penalty": 1.15, "code_predictor_temperature": 0.5,
                          "code_predictor_top_p": 0.8, "code_predictor_top_k": 10})
        self.assertEqual(fields({"greedy": True, "codeGreedy": True, "temperature": 0.5, "codeTopK": 10}, False),
                         {"language": "en", "seed": 1, "do_sample": False, "code_predictor_do_sample": False},
                         "speech.cpp refuses sampling fields with greedy")
        self.assertEqual(fields({"instructions": "Calm."}, True)["instructions"], "Calm.")
        self.assertNotIn("instructions", fields({"instructions": ""}, True), "empty instructions stay unset")
        for bad, allowed in (({"instructions": "Calm."}, False), ({"other": 1}, True), ({"seed": 1.5}, True)):
            with self.subTest(settings=bad), self.assertRaises(ValueError):
                fields(bad, allowed)

    def test_both_sizes_take_all_nine_speakers_and_the_17b_its_own_file_and_instructions(self):
        self.assertEqual(len(qwen_speechcpp.VOICES), 9)
        backend = self.backend("qwen3-tts-1.7b")
        try:
            self.assertEqual(backend.load()["setupId"], "qwen3-tts-1.7b")
            self.assertEqual(backend._model.args[2], str(self.root / "Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf"))
            for voice in qwen_speechcpp.VOICES:
                backend.speak("Hi.", voice, 1, {"instructions": "Speak slowly."})
        finally:
            backend.close()
        sent = [json.loads(line) for line in self.requests()]
        self.assertEqual([request["voice"] for request in sent], list(qwen_speechcpp.VOICES))
        self.assertTrue(all(request["instructions"] == "Speak slowly." for request in sent))

    def test_load_attests_the_payload_before_starting_the_worker(self):
        (self.root / qwen_speechcpp.MODELS["qwen3-tts-0.6b"]["file"]).write_bytes(b"nope")
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            backend.load()
        self.assertIsNone(backend._model)


if __name__ == "__main__":
    unittest.main()
