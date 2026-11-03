// Times one kokoro-coreml variant on cohort texts, whole text per call, with
// process CPU per call: the compute unit comparison of ane-check's CuProbe.
// One process per variant, so the first synthesis is a true cold one.
// usage: KokoroAneProbe <bundle root> <hybrid|folded|cpu> <reps> <cohort.json> <out.json> <textId>...
import Foundation
import KokoroAneShared
import KokoroTTS

struct Cohort: Decodable {
    struct Text: Decodable { let id: String; let length: String; let text: String }
    let texts: [Text]
}

let a = CommandLine.arguments
guard a.count >= 7, let variant = Variant(rawValue: a[2]), let reps = Int(a[3]) else {
    FileHandle.standardError.write(Data("usage: KokoroAneProbe <root> <hybrid|folded|cpu> <reps> <cohort.json> <out.json> <textId>...\n".utf8))
    exit(2)
}
let cohort = try JSONDecoder().decode(Cohort.self, from: Data(contentsOf: URL(fileURLWithPath: a[4])))
let texts = a[6...].map { id in
    guard let t = cohort.texts.first(where: { $0.id == id }) else { fatalError("no cohort text \(id)") }
    return t
}

var out: [String: Any] = ["variant": variant.rawValue]
let t0 = nowMs(), c0 = processCpuMs()
let tts = try await variant.load(root: URL(fileURLWithPath: a[1], isDirectory: true))
out["loadMs"] = nowMs() - t0
out["loadCpuMs"] = processCpuMs() - c0
var runs: [[String: Any]] = []
for (i, t) in ([texts[0]] + Array(repeating: texts, count: reps).flatMap { $0 }).enumerated() {
    let s = nowMs(), c = processCpuMs()
    let audio = try await tts.synthesize(t.text, voice: .afHeart)
    let ms = nowMs() - s, cpuMs = processCpuMs() - c
    let report = await tts.lastReport
    let generators = try variant.check(report)
    let stages = (report?.chunks ?? []).map { chunk in
        Dictionary(chunk.stageMilliseconds.map { ($0.name, $0.ms) }, uniquingKeysWith: +)
    }
    runs.append(["i": i, "text": t.id, "length": t.length, "cold": i == 0, "wallMs": ms, "cpuMs": cpuMs,
                 "audioMs": audio.durationSeconds * 1000, "rtf": ms / (audio.durationSeconds * 1000),
                 "chunks": stages.count, "generators": generators, "buckets": (report?.chunks ?? []).map(\.bucketSeconds),
                 "stagesMs": stages])
}
out["runs"] = runs
try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: a[5]))
