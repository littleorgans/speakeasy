"""Qwen3 TTS 0.6B and 1.7B Base voice copy on official qwen-tts, PyTorch on Metal (MPS), float32.

Backend options are exactly setupId, an absolute local modelPath (the pinned
payload: the Base snapshot files plus the synthetic references) and files,
its pins from the host registry. Construction imports nothing; load attests
the payload and the pinned environment, then loads the model offline. speak
takes the reading's settings and its reference (audio, sample rate, hash and
transcript) and conditions on a clone prompt built once per reference,
transcript and copy mode, as the voice copy recipe found reusable across
sentences. The official API resamples the reference itself.
"""

from collections import OrderedDict
import copy
import math
import os
from pathlib import Path
import time

from pins import attest_snapshot, pinned_files, runtime_identity


SETUPS = ("qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base")
PYTHON_VERSION = "3.12.5"
RUNTIME_VERSIONS = {"qwen-tts": "0.1.1", "torch": "2.14.1", "torchaudio": "2.11.0", "transformers": "4.57.3",
                    "numpy": "2.5.4", "soundfile": "0.14.0"}
VOICES = ("ref-3s", "ref-10s", "ref-20s", "recording")
SAMPLE_RATE = 24_000
FRAMES_PER_SECOND = 12.5
MAX_NEW_TOKENS = 8_192
PROMPT_CACHE = 8
LANGUAGES = {"en": "English", "auto": "Auto", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German",
             "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}
DEFAULTS = {"copyMode": "words-and-voice", "language": "en", "seed": 1, "maxSeconds": 20, "temperature": 0.9,
            "topP": 1, "topK": 50, "repetitionPenalty": 1.05, "greedy": False, "codeTemperature": 0.9,
            "codeTopP": 1, "codeTopK": 50, "codeGreedy": False}
COPY_MODES = ("words-and-voice", "voice-only")


def generate_kwargs(chosen):
    """The reading's sampling as official generate keywords; the code predictor is the subtalker."""
    seconds = chosen["maxSeconds"]
    return {"do_sample": not chosen["greedy"], "temperature": chosen["temperature"], "top_p": chosen["topP"],
            "top_k": chosen["topK"], "repetition_penalty": chosen["repetitionPenalty"],
            "subtalker_dosample": not chosen["codeGreedy"], "subtalker_temperature": chosen["codeTemperature"],
            "subtalker_top_p": chosen["codeTopP"], "subtalker_top_k": chosen["codeTopK"],
            "max_new_tokens": math.floor(seconds * FRAMES_PER_SECOND) if seconds else MAX_NEW_TOKENS}


class Backend:
    # The worker passes settings and a reference body only to backends that declare them.
    takes_settings = True
    takes_reference = True

    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files"}:
            raise ValueError("Qwen3 TTS voice copy options require exactly setupId, modelPath and files")
        if options["setupId"] not in SETUPS:
            raise ValueError("Qwen3 TTS voice copy backend requires a Qwen3 TTS Base setup")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or "\0" in model_path or not Path(model_path).is_absolute():
            raise ValueError("Qwen3 TTS voice copy modelPath must be absolute")
        self._setup_id = options["setupId"]
        self._model_path = Path(model_path)
        self._files = pinned_files(options["files"], "Qwen3 TTS voice copy")
        self._model = None
        self._metadata = None
        self._prompts = OrderedDict()

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, self._files, "Qwen3 TTS voice copy")
        runtime = runtime_identity(PYTHON_VERSION, RUNTIME_VERSIONS, "requirements-qwen-tts.txt", "Qwen3 TTS voice copy")
        # The pinned payload only: no hub requests, before transformers reads its environment.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        start = time.perf_counter()
        import numpy as np
        import torch
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(str(self._model_path), device_map="mps", dtype=torch.float32,
                                              attn_implementation="sdpa", local_files_only=True)
        if model.model.tts_model_type != "base":
            raise ValueError("Qwen3 TTS voice copy needs a Base model")
        if model.model.speech_tokenizer.get_output_sample_rate() != SAMPLE_RATE:
            raise ValueError("Qwen3 TTS voice copy sample rate differs from the pinned 24000 Hz")
        supported = model.get_supported_languages()
        if supported is not None and any(name.lower() not in supported for name in LANGUAGES.values()):
            raise ValueError("Qwen3 TTS voice copy model lacks an offered language")
        self._np, self._torch, self._model = np, torch, model
        self._metadata = {"setupId": self._setup_id, "runtime": runtime, "device": "mps", "dtype": "float32",
                          "sampleRate": SAMPLE_RATE, "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def _prompt(self, reference, mode):
        transcript = reference["transcript"] if mode == "words-and-voice" else None
        if mode == "words-and-voice" and not (isinstance(transcript, str) and transcript.strip()):
            raise ValueError("Words and voice needs the reference's exact transcript")
        key = (reference["sha256"], reference["sampleRate"], transcript, mode)
        if key in self._prompts:
            self._prompts.move_to_end(key)
            return self._prompts[key]
        audio = self._np.array(reference["samples"], dtype=self._np.float32)
        # The whole item list: full prompt mode keeps the transcript inside it.
        prompt = self._model.create_voice_clone_prompt(ref_audio=(audio, reference["sampleRate"]), ref_text=transcript,
                                                       x_vector_only_mode=mode == "voice-only")
        self._prompts[key] = prompt
        if len(self._prompts) > PROMPT_CACHE:
            self._prompts.popitem(last=False)
        return prompt

    def speak(self, text, voice, speed, settings=None, reference=None):
        if self._model is None:
            raise ValueError("Qwen3 TTS voice copy backend is not loaded")
        if voice not in VOICES:
            raise ValueError("Unsupported Qwen3 TTS voice copy voice")
        if speed != 1:
            raise ValueError("Qwen3 TTS has no speed control; speak at 1")
        if reference is None:
            raise ValueError("Qwen3 TTS voice copy needs a reference")
        chosen = {**DEFAULTS, **(settings or {})}
        if set(chosen) != set(DEFAULTS) or chosen["copyMode"] not in COPY_MODES or chosen["language"] not in LANGUAGES:
            raise ValueError("Unsupported Qwen3 TTS voice copy setting")
        if type(chosen["seed"]) is not int or not 0 <= chosen["seed"] <= 2**53 - 1:
            raise ValueError("Qwen3 TTS seed must be an integer")
        start = time.perf_counter()
        prompt = self._prompt(reference, chosen["copyMode"])
        # Every segment starts from the reading's seed, as speech.cpp does per synthesis.
        self._torch.manual_seed(chosen["seed"])
        self._torch.mps.manual_seed(chosen["seed"])
        wavs, rate = self._model.generate_voice_clone(text=text, language=LANGUAGES[chosen["language"]],
                                                      voice_clone_prompt=prompt, non_streaming_mode=False,
                                                      **generate_kwargs(chosen))
        if rate != SAMPLE_RATE:
            raise ValueError("Qwen3 TTS voice copy returned another sample rate")
        samples = self._np.asarray(wavs[0], dtype=self._np.float32).reshape(-1)
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("Qwen3 TTS voice copy timing must be finite")
        return {"samples": samples.astype("<f4").tobytes(), "sampleRate": SAMPLE_RATE, "processingMs": elapsed}

    def close(self):
        self._model = None
        self._prompts.clear()
