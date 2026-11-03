// Resident kokoro-coreml worker: the native worker protocol of
// experiments/fast-tts/driver.ts (NativeWorker). Each request part is raw
// text; one KokoroTTS.synthesize call per part (Misaki G2P, the SDK's own
// chunking past 15 s) and one AUDIO frame per part. A part whose chunks ran
// another generator than the variant's is an ERROR, never a mislabelled run.
// Cancellation checkpoint: between parts; an active Core ML chain runs to completion.
//
// usage: KokoroCoreMLWorker <bundle root> <voice> <hybrid|folded|cpu>

import Darwin
import Foundation
import KokoroAneShared
import KokoroTTS

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
guard arguments.count == 4, let variant = Variant(rawValue: arguments[3]) else {
    FileHandle.standardError.write(Data("usage: KokoroCoreMLWorker <bundle root> <voice> <hybrid|folded|cpu>\n".utf8))
    exit(2)
}
let voice = KokoroVoiceID(arguments[2])

do {
    let tts = try await variant.load(root: URL(fileURLWithPath: arguments[1], isDirectory: true))
    info(.ready, ["runtime": "kokoro-coreml KokoroTTS \(variant.rawValue)", "voice": voice.rawValue])

    while let line = readLine() {
        if line == "Q" { break }
        guard line.hasPrefix("S\t") else {
            info(.error, ["error": "bad request"])
            continue
        }
        var generators: [String] = []
        do {
            for part in line.dropFirst(2).split(separator: "\u{1f}", omittingEmptySubsequences: false) {
                let audio = try await tts.synthesize(String(part), voice: voice)
                generators += try variant.check(await tts.lastReport)
                let samples = audio.samples.withUnsafeBufferPointer { Data(buffer: $0) }
                frame(.audio, count: audio.samples.count, payload: samples)
            }
            info(.end, ["generators": generators])
        } catch {
            info(.error, ["error": "\(error)"])
        }
    }
} catch {
    info(.error, ["error": "\(error)"])
    exit(1)
}
