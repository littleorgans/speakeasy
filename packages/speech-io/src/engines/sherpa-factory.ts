import { SherpaEngine, type SherpaEngineOptions } from "./sherpa.ts";
import { SherpaOfflineEngine } from "./sherpa-offline.ts";
import { resolveSherpaModel } from "./sherpa-models.ts";

export type AnySherpaEngine = SherpaEngine | SherpaOfflineEngine;

/** Build the engine that matches the registry entry's kind. */
export function createSherpaEngine(options: SherpaEngineOptions): AnySherpaEngine {
  if (resolveSherpaModel(options.model).kind === "streaming") {
    return new SherpaEngine(options);
  }
  return new SherpaOfflineEngine(options);
}
