// Apple on device speech for the browser voice host: SpeechAnalyzer with
// SpeechTranscriber for en-US, on system managed weights. Ported from the
// stt-ux apple track (Main.swift makeModule and runTake, Audio.swift
// feedAudio): volatile results, audio time ranges, 2,560 sample buffers
// through AnalyzerInputConverter, finalizeAndFinishThroughEndOfInput.
//
// stdin, as the transcribe.cpp worker: per request one op byte, a u32 LE
// sample count, then f32 LE samples at 16 kHz mono. `T` transcribes a whole
// take; `S` starts a live stream and `F` feeds it, each answered with the
// stream's committed and volatile text. `S` carries two i32 LE before its
// samples: 1 for fast results, then 0. Any other request ends the live
// stream. stdout: one JSON line at ready, then one per request.
import AVFoundation
import Foundation
import Speech

enum Failure: String, Error { case unavailable, assetsMissing, unsupportedFormat, badRequest, conversion }

struct Request: Sendable { let op: UInt8; let fast: Bool; let samples: [Float] }

let maxSamples = 1_920_000
let natural = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!

func emit(_ record: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func readExact(_ count: Int) -> Data? {
    var data = Data()
    while data.count < count {
        let chunk = FileHandle.standardInput.readData(ofLength: count - data.count)
        if chunk.isEmpty { return nil }
        data.append(chunk)
    }
    return data
}

func little<T: FixedWidthInteger>(_ data: Data, _ offset: Int, _ type: T.Type) -> T {
    data[offset..<offset + MemoryLayout<T>.size].enumerated().reduce(T(0)) { $0 | T($1.element) << (8 * $1.offset) }
}

/// Requests read on their own thread, so stdin never blocks the analyzer's results.
func requests() -> AsyncThrowingStream<Request, Error> {
    AsyncThrowingStream { continuation in
        Thread {
            while let head = readExact(5) {
                let op = head[0], count = Int(little(head, 1, UInt32.self))
                guard count <= maxSamples, [UInt8(ascii: "T"), UInt8(ascii: "S"), UInt8(ascii: "F")].contains(op),
                      op != UInt8(ascii: "T") || count > 0 else { continuation.finish(throwing: Failure.badRequest); return }
                var fast = false
                if op == UInt8(ascii: "S") {
                    guard let extra = readExact(8) else { break }
                    fast = little(extra, 0, Int32.self) == 1
                }
                guard let body = readExact(count * 4) else { break }
                let samples = body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                guard samples.allSatisfy(\.isFinite) else { continuation.finish(throwing: Failure.badRequest); return }
                continuation.yield(Request(op: op, fast: fast, samples: samples))
            }
            continuation.finish()
        }.start()
    }
}

func makeModule(_ locale: Locale, fast: Bool) -> SpeechTranscriber {
    SpeechTranscriber(locale: locale, transcriptionOptions: [],
                      reportingOptions: fast ? [.volatileResults, .fastResults] : [.volatileResults],
                      attributeOptions: [.audioTimeRange])
}

/// One analyzer over one take: finalized results are committed, the latest volatile one may still change.
@MainActor final class Take {
    let analyzer: SpeechAnalyzer
    let converter: AnalyzerInputConverter
    let input: AsyncThrowingStream<AnalyzerInput, Error>.Continuation
    var collector: Task<Void, Error>?
    var committed = ""
    var volatile = ""
    var frames: AVAudioFramePosition = 0

    init(locale: Locale, fast: Bool) async throws {
        let module = makeModule(locale, fast: fast)
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module], considering: natural)
        else { throw Failure.unsupportedFormat }
        // The worker owns the model for its lifetime; the host's idle release ends the worker.
        analyzer = SpeechAnalyzer(modules: [module], options: .init(priority: .userInitiated,
            modelRetention: .processLifetime, ignoresResourceLimits: false))
        converter = AnalyzerInputConverter(analyzerFormat: format)
        let pair = AsyncThrowingStream<AnalyzerInput, Error>.makeStream()
        input = pair.continuation
        try await analyzer.prepareToAnalyze(in: format)
        collector = Task { @MainActor [unowned self] in
            for try await result in module.results {
                let text = String(result.text.characters)
                if result.isFinal { committed += text; volatile = "" } else { volatile = text }
            }
        }
        try await analyzer.start(inputSequence: pair.stream)
    }

    func feed(_ samples: [Float]) throws {
        for start in stride(from: 0, to: samples.count, by: 2560) {
            let part = samples[start..<min(start + 2560, samples.count)]
            guard let buffer = AVAudioPCMBuffer(pcmFormat: natural, frameCapacity: AVAudioFrameCount(part.count))
            else { throw Failure.conversion }
            buffer.frameLength = AVAudioFrameCount(part.count)
            part.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: part.count) }
            for converted in try converter.convert(buffer, at: AVAudioTime(sampleTime: frames, atRate: 16000)) {
                input.yield(converted)
            }
            frames += AVAudioFramePosition(part.count)
        }
    }

    func finish() async throws -> String {
        for converted in try converter.flush() { input.yield(converted) }
        input.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        try await collector?.value
        return committed + volatile
    }

    func cancel() async {
        input.finish()
        await analyzer.cancelAndFinishNow()
        collector?.cancel()
    }
}

func elapsed(_ start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

@main struct Main {
    @MainActor static func main() async {
        do {
            let started = ContinuousClock.now
            guard SpeechTranscriber.isAvailable,
                  let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US"))
            else { throw Failure.unavailable }
            let module = makeModule(locale, fast: false)
            // System managed weights: reserve the locale, then install its assets if the system has not.
            var reservedHere = false
            if !(await AssetInventory.reservedLocales).contains(locale) {
                guard try await AssetInventory.reserve(locale: locale) else { throw Failure.assetsMissing }
                reservedHere = true
            }
            if await AssetInventory.status(forModules: [module]) != .installed,
               let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
                try await request.downloadAndInstall()
            }
            guard await AssetInventory.status(forModules: [module]) == .installed else { throw Failure.assetsMissing }
            emit(["kind": "ready", "backend": "apple", "variant": "speech-transcriber", "locale": locale.identifier,
                  "reservedHere": reservedHere, "loadMs": elapsed(started)])
            var live: Take?
            for try await request in requests() {
                let clock = ContinuousClock.now
                do {
                    if request.op == UInt8(ascii: "T") {
                        await live?.cancel(); live = nil
                        let take = try await Take(locale: locale, fast: false)
                        try take.feed(request.samples)
                        let text = try await take.finish()
                        emit(["kind": "result", "text": text, "rawText": text, "processingMs": elapsed(clock)])
                        continue
                    }
                    if request.op == UInt8(ascii: "S") {
                        await live?.cancel()
                        live = try await Take(locale: locale, fast: request.fast)
                    }
                    guard let take = live else { throw Failure.badRequest }
                    try take.feed(request.samples)
                    // Results arrive on their own task; let those already decoded land.
                    for _ in 0..<4 { await Task.yield() }
                    emit(["kind": "partial", "stable": take.committed, "tentative": take.volatile, "processingMs": elapsed(clock)])
                } catch {
                    await live?.cancel(); live = nil
                    // System error text may carry owner data; send only its kind.
                    emit(["kind": "error", "error": (error as? Failure)?.rawValue ?? "system"])
                }
            }
            await live?.cancel()
            if reservedHere { _ = await AssetInventory.release(reservedLocale: locale) }
        } catch {
            emit(["kind": "error", "error": (error as? Failure)?.rawValue ?? "system"])
            exit(1)
        }
    }
}
