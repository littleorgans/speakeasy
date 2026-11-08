#!/usr/bin/env python3
"""Build an Apple on device worker with xcrun swiftc, as the stt-ux apple harness does.

--worker speech (the default) builds the speech in worker; --worker cleanup
builds the Foundation Models text cleanup worker (runtime/apple_cleanup).
Command Line Tools suffice; both frameworks ship with macOS 27. Builds in a
fresh temporary scratch directory, copies the worker to --out and removes the
scratch directory after a successful build.

    python3 runtime/apple_speech/build.py --out ~/Library/Caches/speakeasy/apple-speech/<hash>
    python3 runtime/apple_speech/build.py --worker cleanup --out ~/Library/Caches/speakeasy/apple-cleanup/<hash>
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
TARGET = "arm64-apple-macos27.0"
# Each worker: its one Swift source and the executable name it is built to.
WORKERS = {"speech": (HERE / "worker.swift", "apple-speech-worker"),
           "cleanup": (HERE.parent / "apple_cleanup" / "worker.swift", "apple-cleanup-worker")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", choices=WORKERS, default="speech")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    source, name = WORKERS[args.worker]
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    root = Path(tempfile.mkdtemp(prefix=f"apple-{args.worker}-build-"))
    sdk = subprocess.check_output(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    with (root / "build.log").open("w") as log:
        result = subprocess.run(["/usr/bin/xcrun", "--sdk", "macosx", "swiftc", "-swift-version", "6",
                                 "-parse-as-library", "-O", "-sdk", sdk, "-target", TARGET,
                                 str(source), "-o", str(root / name)],
                                stdout=log, stderr=log)
    if result.returncode:
        raise SystemExit(f"swiftc failed with exit {result.returncode}; see {root / 'build.log'}")
    out.mkdir(parents=True)
    shutil.copy2(root / name, out / name)
    shutil.rmtree(root)
    with (out / name).open("rb") as stream:
        print(f"{out / name} sha256 {hashlib.file_digest(stream, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
