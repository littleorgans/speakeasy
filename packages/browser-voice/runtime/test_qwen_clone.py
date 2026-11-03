"""Qwen3 TTS voice copy over fake qwen_tts, torch and numpy modules: pins, prompt cache, seeds and generate keywords."""

import hashlib
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qwen_clone


class FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def reshape(self, _shape):
        return self

    def astype(self, _dtype):
        return self

    def tobytes(self):
        return struct.pack(f"<{len(self.values)}f", *self.values)


class FakeModel:
    def __init__(self, kind="base", rate=24_000):
        self.model = types.SimpleNamespace(tts_model_type=kind,
                                           speech_tokenizer=types.SimpleNamespace(get_output_sample_rate=lambda: rate))
        self.prompts = []
        self.generations = []

    def get_supported_languages(self):
        return sorted(name.lower() for name in qwen_clone.LANGUAGES.values())

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        audio, rate = ref_audio
        self.prompts.append((audio.values, rate, ref_text, x_vector_only_mode))
        return [f"prompt {len(self.prompts)}"]

    def generate_voice_clone(self, **kwargs):
        self.generations.append(kwargs)
        return [[0.5, -0.5]], 24_000


class QwenCloneTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "model.safetensors").write_bytes(b"base")
        self.files = [{"name": "model.safetensors", "bytes": 4, "sha256": hashlib.sha256(b"base").hexdigest()}]
        self.model = FakeModel()
        self.seeds = []
        torch = types.SimpleNamespace(float32="float32", manual_seed=lambda seed: self.seeds.append(("cpu", seed)),
                                      mps=types.SimpleNamespace(manual_seed=lambda seed: self.seeds.append(("mps", seed))))
        self.loads = []

        def from_pretrained(path, **kwargs):
            self.loads.append((path, kwargs))
            return self.model

        modules = {"torch": torch, "numpy": types.SimpleNamespace(float32="float32", array=lambda values, dtype: FakeArray(values),
                                                                   asarray=lambda values, dtype: FakeArray(values)),
                   "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=from_pretrained))}
        for patcher in (patch.dict(sys.modules, modules), patch.dict("os.environ", {}),
                        patch.object(qwen_clone, "runtime_identity", self.identity)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.identities = []

    def identity(self, *args):
        self.identities.append(args)
        return {"python": args[0], **args[1]}

    def backend(self, setup_id="qwen3-tts-0.6b-base"):
        backend = qwen_clone.Backend({"setupId": setup_id, "modelPath": str(self.root), "files": self.files})
        backend.load()
        return backend

    def reference(self, sha="a" * 64, transcript="Calm words.", rate=48_000):
        return {"samples": [0.25, -0.25], "sampleRate": rate, "sha256": sha, "transcript": transcript}

    def test_load_checks_the_pinned_environment_and_loads_offline_on_mps_float32(self):
        self.backend("qwen3-tts-1.7b-base")
        self.assertEqual(self.identities, [("3.12.5", qwen_clone.RUNTIME_VERSIONS, "requirements-qwen-tts.txt",
                                            "Qwen3 TTS voice copy")])
        path, kwargs = self.loads[0]
        self.assertEqual(path, str(self.root))
        self.assertEqual(kwargs, {"device_map": "mps", "dtype": "float32", "attn_implementation": "sdpa",
                                  "local_files_only": True})
        import os
        self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))
        for model in (FakeModel(kind="custom_voice"), FakeModel(rate=16_000)):
            self.model = model
            with self.subTest(model=model.model.tts_model_type), self.assertRaises(ValueError):
                self.backend()

    def test_prompts_are_cached_by_audio_transcript_and_mode(self):
        backend = self.backend()
        for reference, settings in ((self.reference(), {}), (self.reference(), {"seed": 2}),
                                    (self.reference(transcript="Other words."), {}),
                                    (self.reference(), {"copyMode": "voice-only"}),
                                    (self.reference(transcript="Ignored."), {"copyMode": "voice-only"}),
                                    (self.reference(sha="b" * 64), {}), (self.reference(rate=44_100), {})):
            result = backend.speak("Hello.", "recording", 1, settings, reference)
        self.assertEqual([prompt[2:] for prompt in self.model.prompts],
                         [("Calm words.", False), ("Other words.", False), (None, True), ("Calm words.", False),
                          ("Calm words.", False)], "a voice only prompt ignores the transcript")
        self.assertEqual(self.model.prompts[0][:2], ([0.25, -0.25], 48_000), "the reference goes in at its own rate")
        self.assertEqual(self.model.generations[1]["voice_clone_prompt"], ["prompt 1"], "the whole item list is reused")
        self.assertEqual(struct.unpack("<2f", result["samples"]), (0.5, -0.5))
        self.assertEqual(result["sampleRate"], 24_000)
        for index in range(qwen_clone.PROMPT_CACHE + 1):
            backend.speak("Hi.", "recording", 1, {}, self.reference(sha=f"{index:064x}"))
        self.assertEqual(len(backend._prompts), qwen_clone.PROMPT_CACHE)

    def test_each_segment_is_seeded_and_settings_become_generate_keywords(self):
        backend = self.backend()
        backend.speak("One.", "ref-3s", 1, {"seed": 7, "language": "auto", "maxSeconds": 5, "temperature": 0.7,
                                            "codeGreedy": True}, self.reference())
        backend.speak("Two.", "ref-3s", 1, {"seed": 7, "maxSeconds": 0}, self.reference())
        self.assertEqual(self.seeds, [("cpu", 7), ("mps", 7)] * 2)
        first, second = self.model.generations
        self.assertEqual((first["language"], second["language"]), ("Auto", "English"))
        self.assertEqual((first["max_new_tokens"], second["max_new_tokens"]), (62, 8_192))
        self.assertEqual({key: first[key] for key in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                                                      "subtalker_dosample", "subtalker_temperature")},
                         {"do_sample": True, "temperature": 0.7, "top_p": 1, "top_k": 50, "repetition_penalty": 1.05,
                          "subtalker_dosample": False, "subtalker_temperature": 0.9})
        self.assertFalse(first["non_streaming_mode"])
        backend.speak("Three.", "ref-3s", 1, None, self.reference())
        self.assertEqual(self.model.generations[-1]["max_new_tokens"], 250, "D8: 20 s by default, 12.5 frames per second")

    def test_words_and_voice_needs_a_transcript_and_every_reading_a_reference(self):
        backend = self.backend()
        for settings, reference in (({}, self.reference(transcript=None)), ({}, self.reference(transcript=" ")),
                                    ({}, None), ({"copyMode": "other"}, self.reference()),
                                    ({"language": "xx"}, self.reference()), ({"seed": -1}, self.reference()),
                                    ({"instructions": "Calm."}, self.reference())):
            with self.subTest(settings=settings, reference=reference), self.assertRaises(ValueError):
                backend.speak("Hi.", "recording", 1, settings, reference)
        self.assertEqual(self.model.generations, [])
        backend.speak("Hi.", "recording", 1, {"copyMode": "voice-only"}, self.reference(transcript=None))
        self.assertEqual(len(self.model.generations), 1)


if __name__ == "__main__":
    unittest.main()
