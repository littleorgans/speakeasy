// swift-tools-version: 6.0
// Resident Kokoro ANE v3 worker for the fast-tts kokoro-coreml candidate.
// FluidAudio is pinned by commit; it has no package dependencies and no Metal
// sources, so Command Line Tools build it without Xcode (README.md).
import PackageDescription

let package = Package(
    name: "KokoroAneWorker",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", revision: "e396cf780c4abf8e395087467c0585d3b0d17f68"),
    ],
    targets: [
        .executableTarget(
            name: "KokoroAneWorker",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        ),
    ]
)
