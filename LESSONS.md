# LESSONS

- For STT latency benchmarks, do not use a raw sample-amplitude threshold as end-of-speech ground truth. Use a VAD or documented RMS-window hangover reference, then verify endpoint-to-final cannot be less than the configured trailing-silence budget for a valid config.
