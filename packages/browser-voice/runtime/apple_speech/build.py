#!/usr/bin/env python3
"""Build the Apple on device speech worker with xcrun swiftc, as the stt-ux apple harness does.

Command Line Tools suffice; the Speech framework ships with macOS 27. Builds
in a fresh temporary scratch directory, copies the worker to --out and
removes the scratch directory after a successful build.

    python3 runtime/apple_speech/build.py --out ~/Library/Caches/speakeasy/apple-speech/<hash>
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
TARGET = "arm64-apple-macos27.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    root = Path(tempfile.mkdtemp(prefix="apple-speech-build-"))
    sdk = subprocess.check_output(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    with (root / "build.log").open("w") as log:
        result = subprocess.run(["/usr/bin/xcrun", "--sdk", "macosx", "swiftc", "-swift-version", "6",
                                 "-parse-as-library", "-O", "-sdk", sdk, "-target", TARGET,
                                 str(HERE / "worker.swift"), "-o", str(root / "apple-speech-worker")],
                                stdout=log, stderr=log)
    if result.returncode:
        raise SystemExit(f"swiftc failed with exit {result.returncode}; see {root / 'build.log'}")
    out.mkdir(parents=True)
    shutil.copy2(root / "apple-speech-worker", out / "apple-speech-worker")
    shutil.rmtree(root)
    with (out / "apple-speech-worker").open("rb") as stream:
        print(f"{out / 'apple-speech-worker'} sha256 {hashlib.file_digest(stream, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
