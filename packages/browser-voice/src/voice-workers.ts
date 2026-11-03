import { join } from "node:path";
import { LOCAL_OUTPUT_MODELS, type LocalOutputModelId } from "../../speech-io/src/tts/catalogue.ts";
import type { WorkerTtsSynth } from "../../speech-io/src/tts/factory.ts";
import { createTtsFrontend, type TtsFrontend } from "../../speech-io/src/tts/frontend.ts";
import { LOCAL_TTS_MODELS, type WorkerTtsModel } from "../../speech-io/src/tts/models.ts";
import { PythonSetupWorker } from "./setup-workers.ts";

/** Machine paths for one worker voice, from the setup runtime file. */
export type VoiceRuntimeLocation = {
  pythonPath: string; runnerPath?: string; loadTimeoutMs?: number; operationTimeoutMs?: number;
};

/** Separates phoneme parts in one request, as the native runner protocol does. */
const PART_SEPARATOR = "\u001f";

/**
 * A worker voice behind the shared sentence pipeline: each planned segment is
 * one speak request to the Python worker, which attests the registry's pins
 * and loads the voice here, so its load counts as the voice's start. The pipeline's speaker index is the
 * voice's position in the registry. A phoneme runtime receives the segment as
 * Piper Plus phoneme characters, one part per frontend sentence.
 */
export async function createWorkerVoiceSynth(modelId: LocalOutputModelId, location: VoiceRuntimeLocation, root: string,
  workerPath?: string): Promise<WorkerTtsSynth> {
  const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === modelId)!;
  const payload = LOCAL_TTS_MODELS[modelId] as WorkerTtsModel;
  const frontend = payload.phonemesFrom ? await phonemeFrontend(payload.phonemesFrom, root) : undefined;
  const worker = new PythonSetupWorker({ setupId: modelId, pythonPath: location.pythonPath, workerPath,
    backendOptions: { modelPath: join(root, payload.name), files: payload.files,
      ...(location.runnerPath ? { runnerPath: location.runnerPath } : {}) },
    expectedIdentity: { setupId: modelId },
    loadTimeoutMs: location.loadTimeoutMs, operationTimeoutMs: location.operationTimeoutMs });
  await worker.load();
  return {
    numSpeakers: payload.numSpeakers,
    sampleRate: payload.sampleRate,
    async synth(request) {
      const voice = model.voices.find((entry) => entry.speakerId === (request.sid ?? 0));
      if (!voice) throw new Error("Invalid TTS speaker");
      const start = performance.now();
      const text = frontend ? phonemeParts(frontend, request.text) : request.text;
      // Text with nothing to say renders as silence, as the sherpa frontend does.
      const result = text ? await worker.speak(text, voice.id, request.speed ?? 1)
        : { samples: new Float32Array(0), sampleRate: payload.sampleRate };
      if (result.sampleRate !== payload.sampleRate) throw new Error("Voice worker sample rate differs from its pin");
      const totalSynthMs = performance.now() - start;
      const audioDurationMs = (result.samples.length / result.sampleRate) * 1_000;
      return { samples: result.samples, sampleRate: result.sampleRate, firstAudioMs: undefined, totalSynthMs,
        audioDurationMs, rtf: audioDurationMs > 0 ? totalSynthMs / audioDurationMs : 0 };
    },
    close: () => worker.close(),
  };
}

function phonemeFrontend(voiceId: NonNullable<WorkerTtsModel["phonemesFrom"]>, root: string): Promise<TtsFrontend> {
  const voice = LOCAL_TTS_MODELS[voiceId];
  return createTtsFrontend({ tokens: join(root, voice.name, voice.tokens), lang: voice.lang, family: voice.family }, root, false);
}

/** The runner maps characters through the voice's phoneme map after lowercasing and NFD; both must be no-ops. */
function phonemeParts(frontend: TtsFrontend, text: string): string {
  const parts = (frontend.request(text, 0, 1)?.generationConfig?.phonemeCodepoints ?? []).map((codepoints) => {
    const part = String.fromCodePoint(...codepoints);
    if (part.toLowerCase() !== part || part.normalize("NFD") !== part || /[\t\n\r\u001f]/.test(part)) {
      throw new Error("Phonemes would change under the runner's text normalization");
    }
    return part;
  });
  return parts.join(PART_SEPARATOR);
}
