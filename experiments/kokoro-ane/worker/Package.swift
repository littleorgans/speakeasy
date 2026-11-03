// swift-tools-version: 5.9
// Resident worker and placement probe for the kokoro-ane experiment, on
// mattmireles/kokoro-coreml's KokoroTTS SDK pinned by commit. Its MisakiSwift
// dependency pulls in mlx-swift, whose Metal kernels need Xcode's Metal
// toolchain (README.md).
import PackageDescription

let package = Package(
    name: "KokoroCoreMLWorker",
    platforms: [.macOS("15.0")],
    dependencies: [
        .package(url: "https://github.com/mattmireles/kokoro-coreml.git", revision: "b23caa862f0585c2c65187d0fc97e0f8311041dc"),
    ],
    targets: [
        .target(
            name: "KokoroAneShared",
            dependencies: [.product(name: "KokoroTTS", package: "kokoro-coreml")]
        ),
        .executableTarget(name: "KokoroCoreMLWorker", dependencies: ["KokoroAneShared"]),
        .executableTarget(name: "KokoroAneProbe", dependencies: ["KokoroAneShared"]),
    ]
)
