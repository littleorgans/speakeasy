// Times FluidAudio v3 KokoroSourceGenerator at a given acoustic frame count F, shaped as
// KokoroAneV3Pipeline feeds it: x_pre/mask 2F, mask_x10 20F, har/mask_x60 120F+1.
// usage: v3sg <KokoroSourceGenerator.mlmodelc> <units> <frames> <n>
import CoreML
import Foundation
let a = CommandLine.arguments
let units: [String: MLComputeUnits] = ["cpuOnly": .cpuOnly, "cpuAndNeuralEngine": .cpuAndNeuralEngine, "cpuAndGPU": .cpuAndGPU, "all": .all]
let cfg = MLModelConfiguration(); cfg.computeUnits = units[a[2]]!
let f = Int(a[3])!, n = Int(a[4])!
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
func cpuNow() -> Double { var u = rusage(); getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3 }
let model = try MLModel(contentsOf: URL(fileURLWithPath: a[1]), configuration: cfg)
let shapes: [String: [Int]] = ["x_pre": [1, 512, 2 * f], "har": [1, 22, 120 * f + 1], "ref_s": [1, 256],
    "mask": [1, 1, 2 * f], "mask_x10": [1, 1, 20 * f], "mask_x60": [1, 1, 120 * f + 1]]
var feats: [String: MLFeatureValue] = [:]
for (name, d) in model.modelDescription.inputDescriptionsByName {
    let arr = try MLMultiArray(shape: shapes[name]!.map { NSNumber(value: $0) }, dataType: d.multiArrayConstraint!.dataType)
    for i in 0..<arr.count { arr[i] = name.hasPrefix("mask") ? 1 : 0 }
    feats[name] = MLFeatureValue(multiArray: arr)
}
let input = try MLDictionaryFeatureProvider(dictionary: feats)
let f0 = now(); _ = try model.prediction(from: input); let first = now() - f0
var wall: [Double] = [], cpu: [Double] = []
for _ in 0..<n { let s = now(), c = cpuNow(); _ = try model.prediction(from: input); wall.append(now() - s); cpu.append(cpuNow() - c) }
wall.sort(); cpu.sort()
print(String(format: "SourceGenerator\t%@\tframes %d\taudio %.2f s\tfirst %.1f\twall med %.1f\tcpu med %.1f", a[2], f, Double(f) / 40, first, wall[n / 2], cpu[n / 2]))
