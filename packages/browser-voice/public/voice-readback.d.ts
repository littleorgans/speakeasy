export type ReadbackResult = {
  outcome: "finished" | "stopped" | "error";
  message: string;
  firstAudioMs: number | null;
  totalMs: number | null;
  stopMs: number | null;
  hostFirstAudioMs: number | null;
  /** The voice's load on its first use, reported by the host; null when it was already loaded. */
  loadMs: number | null;
  samples: Float32Array;
  sampleRate: number | null;
  /** Every voice setting as the host applied it; null when the host accepted no reading. */
  settings: Record<string, number | string | boolean> | null;
};

export class VoiceReadback {
  constructor(options: { context: unknown; onChange?: () => void });
  readonly active: boolean;
  readonly playing: boolean;
  prepare(): Promise<void>;
  speak(request: { selection: unknown; catalogueRevision: string; text: string;
    settings?: Record<string, number | string | boolean>; reference?: { id: string; transcript: string };
    restore?: () => Promise<{ id: string; transcript: string } | null> }): Promise<ReadbackResult>;
  stop(from?: number): void;
}
