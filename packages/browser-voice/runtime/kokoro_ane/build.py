#!/usr/bin/env python3
"""Build the Kokoro ANE worker (experiments/kokoro-ane/worker, mattmireles/kokoro-coreml pinned by commit).

Needs Xcode 27.2 or later and its Metal toolchain (`xcodebuild -downloadComponent
MetalToolchain`, once): MisakiSwift pulls in mlx-swift. Builds in a fresh
temporary scratch directory and copies the worker and its resource bundles to
--out. Diagnostics go to a log in the scratch directory, which is removed
after a successful build.

    python3 runtime/kokoro_ane/build.py --out ~/Library/Caches/speakeasy/kokoro-ane/<hash>
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).resolve().parents[4] / "experiments" / "kokoro-ane" / "worker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    root = Path(tempfile.mkdtemp(prefix="kokoro-ane-build-"))
    with (root / "build.log").open("w") as log:
        result = subprocess.run(["xcrun", "swift", "build", "-c", "release", "--product", "KokoroCoreMLWorker",
                                 "--scratch-path", str(root / "scratch")], cwd=SOURCE, stdout=log, stderr=log)
    if result.returncode:
        raise SystemExit(f"swift build failed with exit {result.returncode}; see {root / 'build.log'}")
    release = root / "scratch" / "release"
    out.mkdir(parents=True)
    shutil.copy2(release / "KokoroCoreMLWorker", out / "KokoroCoreMLWorker")
    # SwiftPM resolves resource bundles, mlx-swift's Metal library among them, beside the executable,
    # and the worker loads MisakiSwift as a dynamic library from there too (@rpath).
    for bundle in release.glob("*.bundle"):
        shutil.copytree(bundle, out / bundle.name)
    for library in release.glob("*.dylib"):
        shutil.copy2(library, out / library.name)
    shutil.rmtree(root)
    print(out / "KokoroCoreMLWorker")


if __name__ == "__main__":
    main()
