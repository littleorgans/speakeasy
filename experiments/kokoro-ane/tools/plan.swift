// Per op Core ML placement for compiled models: MLComputePlan preferred device
// and estimated cost share. usage: plan <units: all|cpuAndNeuralEngine|cpuAndGPU|cpuOnly> <model.mlmodelc>...
// kokoro-ane: env PLAN_FUNCTIONS=a,b,... plans each function of a multifunction model.
import CoreML
import Foundation

func unitsFrom(_ s: String) -> MLComputeUnits {
    switch s {
    case "cpuOnly": return .cpuOnly
    case "cpuAndGPU": return .cpuAndGPU
    case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
    default: return .all
    }
}
func name(_ d: MLComputeDevice) -> String {
    switch d {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "?"
    }
}
func walk(_ block: MLModelStructure.Program.Block, _ visit: (MLModelStructure.Program.Operation) -> Void) {
    for op in block.operations {
        visit(op)
        for b in op.blocks { walk(b, visit) }
    }
}

let args = CommandLine.arguments
let units = unitsFrom(args[1])
let functions: [String?] = ProcessInfo.processInfo.environment["PLAN_FUNCTIONS"].map { $0.split(separator: ",").map(String.init) } ?? [nil]
for path in args.dropFirst(2) { for function in functions {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = units
    cfg.functionName = function
    let plan = try await MLComputePlan.load(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
    guard case let .program(program) = plan.modelStructure else { print("\(path): not an ML program"); continue }
    var ops: [String: Int] = [:], cost: [String: Double] = [:], supportsAne = 0, total = 0
    var cpuTypes: [String: Int] = [:]
    for (_, fn) in program.functions {
        walk(fn.block) { op in
            guard let usage = plan.deviceUsage(for: op) else { return }
            let d = name(usage.preferred)
            total += 1
            ops[d, default: 0] += 1
            cost[d, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
            if usage.supported.contains(where: { if case .neuralEngine = $0 { return true }; return false }) { supportsAne += 1 }
            if d != "ANE" { cpuTypes["\(d):\(op.operatorName)", default: 0] += 1 }
        }
    }
    let stage = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent + (function.map { ":" + $0 } ?? "")
    let opsStr = ops.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
    let costStr = cost.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1f", $0.value * 100))%" }.joined(separator: " ")
    let top = cpuTypes.sorted { $0.value > $1.value }.prefix(6).map { "\($0.key)x\($0.value)" }.joined(separator: ",")
    print("\(args[1])\t\(stage)\tops \(total)\t\(opsStr)\tcost \(costStr)\taneSupported \(supportsAne)\tnonANE \(top)")
}
}
