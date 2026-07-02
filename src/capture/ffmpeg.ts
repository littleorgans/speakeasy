import { spawn } from "node:child_process";

/**
 * Microphone capture seam: ffmpeg (avfoundation) -> 16kHz mono Float32 PCM ->
 * exact 20ms frames.
 *
 * Engine-free by design: this module must never import from src/engines/.
 * Frames-in is the boundary; consumers wire frames into an STTSession
 * themselves, and an Electron AudioWorklet can replace this helper without
 * touching anything downstream of the frame callback.
 */

export const CAPTURE_SAMPLE_RATE = 16_000;
export const CAPTURE_FRAME_MS = 20;
export const CAPTURE_FRAME_SAMPLES =
  (CAPTURE_SAMPLE_RATE * CAPTURE_FRAME_MS) / 1_000;

const FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg";
const DEFAULT_DEVICE = ":0";
const BYTES_PER_SAMPLE = 4;
const FRAME_BYTES = CAPTURE_FRAME_SAMPLES * BYTES_PER_SAMPLE;
const STDERR_TAIL_BYTES = 4_096;

export type MicCaptureOptions = {
  /** Called once per exact 20ms frame (320 samples at 16kHz). */
  onFrame: (frame: Float32Array) => void;
  /** Called when ffmpeg fails to start or exits abnormally. */
  onError: (error: Error) => void;
  /** avfoundation input spec. Default ":0" (default audio device, no video). */
  device?: string;
  ffmpegPath?: string;
};

export type MicCapture = {
  /** Stops capture and resolves once the ffmpeg process has exited. */
  stop: () => Promise<void>;
};

export function startMicCapture(options: MicCaptureOptions): MicCapture {
  const child = spawn(
    options.ffmpegPath ?? FFMPEG_PATH,
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-f",
      "avfoundation",
      "-i",
      options.device ?? DEFAULT_DEVICE,
      "-ar",
      String(CAPTURE_SAMPLE_RATE),
      "-ac",
      "1",
      "-f",
      "f32le",
      "-",
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );

  // ffmpeg stdout chunks do not align to frame boundaries (or even to sample
  // boundaries); carry the remainder across chunks and emit exact frames.
  let carry: Buffer = Buffer.alloc(0);
  let stderrTail = "";
  let stopped = false;

  child.stdout.on("data", (chunk: Buffer) => {
    carry = carry.length === 0 ? chunk : Buffer.concat([carry, chunk]);
    while (carry.length >= FRAME_BYTES) {
      options.onFrame(decodeFrame(carry));
      carry = carry.subarray(FRAME_BYTES);
    }
  });

  child.stderr.on("data", (chunk: Buffer) => {
    stderrTail = (stderrTail + chunk.toString("utf8")).slice(
      -STDERR_TAIL_BYTES,
    );
  });

  child.on("error", (error) => {
    options.onError(
      new Error(`ffmpeg failed to start (${FFMPEG_PATH}): ${error.message}`),
    );
  });

  child.on("close", (code, signal) => {
    if (stopped || code === 0) {
      return;
    }
    const exit = code === null ? `signal ${signal}` : `code ${code}`;
    const detail = stderrTail.trim() || "no stderr output";
    options.onError(
      new Error(
        `ffmpeg exited with ${exit}: ${detail}. On macOS, check that your terminal has microphone permission (System Settings > Privacy & Security > Microphone).`,
      ),
    );
  });

  return {
    stop: () =>
      new Promise((resolve) => {
        if (stopped) {
          resolve();
          return;
        }
        stopped = true;
        if (child.exitCode !== null || child.signalCode !== null) {
          resolve();
          return;
        }
        child.once("close", () => {
          resolve();
        });
        child.kill("SIGTERM");
      }),
  };
}

/** Copy one frame out of the carry buffer; safe for any byte alignment. */
function decodeFrame(bytes: Buffer): Float32Array {
  const frame = new Float32Array(CAPTURE_FRAME_SAMPLES);
  for (let index = 0; index < CAPTURE_FRAME_SAMPLES; index += 1) {
    frame[index] = bytes.readFloatLE(index * BYTES_PER_SAMPLE);
  }
  return frame;
}
