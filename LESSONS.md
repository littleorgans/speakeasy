# LESSONS

- For STT latency benchmarks, do not use a raw sample-amplitude threshold as end-of-speech ground truth. Use a VAD or documented RMS-window hangover reference, then verify endpoint-to-final cannot be less than the configured trailing-silence budget for a valid config.
- For endpoint sweeps, distinguish lowest measured latency from lowest correct result. If no config passes the transcript correctness check, report knee=none and do not label a low-latency clipped or inaccurate output as the answer.
