// On device text cleanup of one final for the browser voice host: Apple
// Foundation Models, a fresh LanguageModelSession per take, greedy, with
// permissive content transformations. Built by apple_speech/build.py
// --worker cleanup; bridged by runtime/apple_cleanup.py.
//
// stdin: one JSON object per line, {"text", "instruction", "names"?,
// "earlier"?}. stdout: one JSON line at ready, then one per request. When the
// system model is unavailable it prints {"kind":"error","error":"unavailable",
// "reason":<kind>} and exits 1. Errors send only their kind.
import Foundation
import FoundationModels

struct Take: Decodable { let text: String; let instruction: String; let names: [String]?; let earlier: [String]? }

func emit(_ record: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func elapsed(_ start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

func prompt(_ take: Take) -> String {
    var parts: [String] = []
    if let names = take.names, !names.isEmpty { parts.append("Known names: \(names.joined(separator: ", "))") }
    if let earlier = take.earlier, !earlier.isEmpty {
        parts.append("Earlier takes, context only, do not repeat them:\n\(earlier.joined(separator: "\n"))")
    }
    parts.append("Transcript to clean:\n\(take.text)")
    return parts.joined(separator: "\n\n")
}

func unavailableKind(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
    switch reason {
    case .deviceNotEligible: "deviceNotEligible"
    case .appleIntelligenceNotEnabled: "appleIntelligenceNotEnabled"
    case .modelNotReady: "modelNotReady"
    @unknown default: "unknown"
    }
}

func errorKind(_ error: Error) -> String {
    switch error {
    case let e as LanguageModelError:
        switch e {
        case .guardrailViolation: "guardrailViolation"
        case .refusal: "refusal"
        case .contextSizeExceeded: "contextSizeExceeded"
        case .unsupportedLanguageOrLocale: "unsupportedLanguageOrLocale"
        case .rateLimited: "rateLimited"
        default: "languageModel"
        }
    case is DecodingError: "badRequest"
    default: "system"
    }
}

@main struct Main {
    static func main() async {
        let started = ContinuousClock.now
        let model = SystemLanguageModel(useCase: .general, guardrails: .permissiveContentTransformations)
        if case .unavailable(let reason) = model.availability {
            emit(["kind": "error", "error": "unavailable", "reason": unavailableKind(reason)])
            exit(1)
        }
        let options = GenerationOptions(samplingMode: .greedy)
        LanguageModelSession(model: model).prewarm()
        emit(["kind": "ready", "backend": "apple", "variant": "foundation-models", "contextSize": model.contextSize,
              "loadMs": elapsed(started)])
        while let line = readLine() {
            let clock = ContinuousClock.now
            do {
                let take = try JSONDecoder().decode(Take.self, from: Data(line.utf8))
                let session = LanguageModelSession(model: model, instructions: take.instruction)
                let response = try await session.respond(to: prompt(take), options: options)
                emit(["kind": "result", "text": response.content, "processingMs": elapsed(clock)])
            } catch {
                // System error text may carry owner data; send only its kind.
                emit(["kind": "error", "error": errorKind(error), "processingMs": elapsed(clock)])
            }
        }
    }
}
