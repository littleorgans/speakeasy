import { join } from "node:path";
import { LOCAL_OUTPUT_MODELS, type LocalOutputModelId } from "../../speech-io/src/tts/catalogue.ts";
import type { WorkerTtsSynth } from "../../speech-io/src/tts/factory.ts";
import { LOCAL_TTS_MODELS } from "../../speech-io/src/tts/models.ts";
import { PythonSetupWorker } from "./setup-workers.ts";

/** Machine paths for one worker voice, from the setup runtime file. */
export type VoiceRuntimeLocation = {
  pythonPath: string; runnerPath?: string; loadTimeoutMs?: number; operationTimeoutMs?: number;
};

/**
 * A worker voice behind the shared sentence pipeline: each planned segment is
 * one speak request to the Python worker, which loads the voice on first use.
 * The pipeline's speaker index is the voice's position in the registry.
 */
export function createWorkerVoiceSynth(modelId: LocalOutputModelId, location: VoiceRuntimeLocation, root: string,
  workerPath?: string): WorkerTtsSynth {
  const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === modelId)!;
  const payload = LOCAL_TTS_MODELS[modelId];
  const worker = new PythonSetupWorker({ setupId: modelId, pythonPath: location.pythonPath, workerPath,
    backendOptions: { modelPath: join(root, payload.name), ...(location.runnerPath ? { runnerPath: location.runnerPath } : {}) },
    expectedIdentity: { setupId: modelId },
    loadTimeoutMs: location.loadTimeoutMs, operationTimeoutMs: location.operationTimeoutMs });
  return {
    numSpeakers: payload.numSpeakers,
    sampleRate: payload.sampleRate,
    async synth(request) {
      const voice = model.voices.find((entry) => entry.speakerId === (request.sid ?? 0));
      if (!voice) throw new Error("Invalid TTS speaker");
      const start = performance.now();
      const result = await worker.speak(request.text, voice.id, request.speed ?? 1);
      if (result.sampleRate !== payload.sampleRate) throw new Error("Voice worker sample rate differs from its pin");
      const totalSynthMs = performance.now() - start;
      const audioDurationMs = (result.samples.length / result.sampleRate) * 1_000;
      return { samples: result.samples, sampleRate: result.sampleRate, firstAudioMs: undefined, totalSynthMs,
        audioDurationMs, rtf: audioDurationMs > 0 ? totalSynthMs / audioDurationMs : 0 };
    },
    close: () => worker.close(),
  };
}
