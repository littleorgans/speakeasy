// Generator variants of the kokoro-coreml starter bundle (one 15 s bucket):
//   hybrid  shipped default: 53 generator convolutions on the ANE, the rest on Metal (GPU);
//           KOKORO_HYBRID_GENERATOR=1 makes a hybrid failure an error instead of a silent folded fallback
//   folded  KOKORO_HYBRID_GENERATOR=0: the row-folded generator, all on the ANE
//   cpu     KokoroComputePolicy.cpuOnly: every stage on the CPU (placement control)
import Darwin
import Foundation
import KokoroTTS

public enum Variant: String, CaseIterable, Sendable {
    case hybrid, folded, cpu

    /// The generator path the SDK reports when this variant runs as intended.
    public var expectedGenerator: String { self == .hybrid ? "hybrid" : "folded" }

    /// Throws unless every chunk of the last synthesis ran this variant's generator.
    public func check(_ report: KokoroSynthesisReport?) throws -> [String] {
        let paths = (report?.chunks ?? []).map(\.generatorPath)
        guard !paths.isEmpty, paths.allSatisfy({ $0 == expectedGenerator }) else {
            throw VariantError.generator(expected: expectedGenerator, ran: paths)
        }
        return paths
    }

    /// Sets the SDK's environment switches, then loads the bundle at `root`
    /// (hydrated by kokoro-say --cache-dir, README.md). Call before any other SDK use.
    public func load(root: URL) async throws -> KokoroTTS {
        setenv("KOKORO_HYBRID_GENERATOR", self == .hybrid ? "1" : "0", 1)
        let resources = KokoroResourceProvider.downloadedDirectory(
            root: root, compiledModelsDirectory: root.appendingPathComponent("compiled-v2", isDirectory: true))
        return try await KokoroTTS.load(resources: resources, computePolicy: self == .cpu ? .cpuOnly : .gistDefault)
    }
}

/// Process CPU time (user + system) in milliseconds.
public func processCpuMs() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3
}

/// Monotonic milliseconds.
public func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

public enum VariantError: Error {
    case generator(expected: String, ran: [String])
}
