// Resident FluidAudio Kokoro ANE worker: the native worker protocol of
// ../../../driver.ts (NativeWorker). Each request part is raw text; one
// synthesizeDetailed call per part (FluidAudio's English frontend, its own
// chunking past 510 phonemes) and one AUDIO frame per part. Cancellation
// checkpoint: between parts; an active Core ML chain runs to completion.
//
// usage: KokoroAneWorker <models root> <voice> <legacy|v3>

import Darwin
import FluidAudio
import Foundation

enum Kind: Int32 { case ready = 1, audio = 2, end = 3, error = 4 }

// Frames own the original stdout; anything else printed goes to stderr.
let framesFD = dup(STDOUT_FILENO)
dup2(STDERR_FILENO, STDOUT_FILENO)

func writeAll(_ data: Data) {
    data.withUnsafeBytes { raw in
        var offset = 0
        while offset < raw.count {
            let written = write(framesFD, raw.baseAddress! + offset, raw.count - offset)
            if written <= 0 { exit(3) }
            offset += written
        }
    }
}

func frame(_ kind: Kind, count: Int, payload: Data) {
    var header = Data()
    withUnsafeBytes(of: kind.rawValue.littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: Int32(count).littleEndian) { header.append(contentsOf: $0) }
    writeAll(header + payload)
}

func info(_ kind: Kind, _ object: [String: Any]) {
    let json = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
    frame(kind, count: json.count, payload: json)
}

let arguments = CommandLine.arguments
guard arguments.count == 4, let version = KokoroAneVersion(rawValue: arguments[3]) else {
    FileHandle.standardError.write(Data("usage: KokoroAneWorker <models root> <voice> <legacy|v3>\n".utf8))
    exit(2)
}
let voice = arguments[2]
// The legacy route and the English G2P assets otherwise follow the hub's mutable
// `main`; pin them to the commit `main` named on 2026-11-03. v3 pins its own files.
ModelRegistry.revisionOverrides = ["FluidInference/kokoro-82m-coreml": "2568276a60ff6e7f312afce3dc9e3f07521cb48d"]

do {
    let manager = KokoroAneManager(
        defaultVoice: voice, directory: URL(fileURLWithPath: arguments[1], isDirectory: true), version: version)
    try await manager.initialize(preloadVoices: [voice])
    info(.ready, ["runtime": "FluidAudio KokoroAneManager \(version.rawValue)", "voice": voice])

    while let line = readLine() {
        if line == "Q" { break }
        guard line.hasPrefix("S\t") else {
            info(.error, ["error": "bad request"])
            continue
        }
        var fastVocoder: [Bool] = []
        do {
            for part in line.dropFirst(2).split(separator: "\u{1f}", omittingEmptySubsequences: false) {
                let result = try await manager.synthesizeDetailed(text: String(part), voice: voice)
                let samples = result.samples.withUnsafeBufferPointer { Data(buffer: $0) }
                frame(.audio, count: result.samples.count, payload: samples)
                fastVocoder.append(result.usedFastVocoder)
            }
            info(.end, ["fastVocoder": fastVocoder])
        } catch {
            info(.error, ["error": "\(error)"])
        }
    }
} catch {
    info(.error, ["error": "\(error)"])
    exit(1)
}
