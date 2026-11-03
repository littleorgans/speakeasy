#!/usr/bin/env python3
"""Build the Kokoro Core ML worker (FluidAudio, pinned in Package.swift and Package.resolved).

Command Line Tools suffice: FluidAudio has no package dependencies and no Metal
sources. Builds in a fresh temporary scratch directory and copies the worker
and its resource bundles to --out. Diagnostics go to a log in the scratch
directory, which is removed after a successful build.

    python3 runtime/kokoro_coreml/build.py --out ~/Library/Caches/speakeasy/kokoro-coreml/e396cf78
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    root = Path(tempfile.mkdtemp(prefix="kokoro-coreml-build-"))
    with (root / "build.log").open("w") as log:
        result = subprocess.run(["swift", "build", "-c", "release", "--scratch-path", str(root / "scratch")],
                                cwd=HERE, stdout=log, stderr=log)
    if result.returncode:
        raise SystemExit(f"swift build failed with exit {result.returncode}; see {root / 'build.log'}")
    release = root / "scratch" / "release"
    out.mkdir(parents=True)
    shutil.copy2(release / "KokoroAneWorker", out / "KokoroAneWorker")
    # SwiftPM resolves resource bundles beside the executable.
    for bundle in release.glob("*.bundle"):
        shutil.copytree(bundle, out / bundle.name)
    shutil.rmtree(root)
    print(out / "KokoroAneWorker")


if __name__ == "__main__":
    main()
