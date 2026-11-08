"""JSON lines native runners: a ready line, then one reply line per request.

Shared by native_asr.py (hex/worker.rs, apple_speech/worker.swift) and
apple_cleanup.py (apple_cleanup/worker.swift). Native diagnostics go to the
transport's stderr, never the protocol.
"""

import json
import subprocess
import sys


def start(argv):
    return subprocess.Popen([str(part) for part in argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr)


def read_reply(process, label):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"{label} worker exited with {process.poll()}")
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise RuntimeError(f"{label} worker reply must be an object")
    return reply


def stop(process):
    """Closes the runner's input and waits for it, killing it if it does not exit."""
    try:
        process.stdin.close()
        process.wait(timeout=5)
    except Exception:
        process.kill()
        process.wait()
    finally:
        process.stdout.close()
