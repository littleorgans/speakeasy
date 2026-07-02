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
/** avfoundation spec for the system-default audio input (never an index: */
/** device order varies per machine, e.g. :0 can be a silent virtual device). */
export const DEFAULT_MIC_DEVICE = ":default";
/**
 * Preferred input devices, matched case-insensitively by name substring in
 * order. Name matching survives device-order shifts that break hardcoded
 * indices; the system default (often a virtual device) is only the fallback.
 */
export const PREFERRED_MIC_NAMES = ["MacBook Pro Microphone"];

const FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg";
const BYTES_PER_SAMPLE = 4;
const FRAME_BYTES = CAPTURE_FRAME_SAMPLES * BYTES_PER_SAMPLE;
const STDERR_TAIL_BYTES = 4_096;
const STOP_SIGKILL_AFTER_MS = 500;

export type MicCaptureOptions = {
  /** Called once per exact 20ms frame (320 samples at 16kHz). */
  onFrame: (frame: Float32Array) => void;
  /** Called when ffmpeg fails to start or exits abnormally. */
  onError: (error: Error) => void;
  /** avfoundation input spec, e.g. ":default" or ":1". */
  device?: string;
  ffmpegPath?: string;
};

export type MicCapture = {
  /** Stops capture and resolves once the ffmpeg process has exited. */
  stop: () => Promise<void>;
};

export type AudioDevice = { index: number; name: string };

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
      options.device ?? DEFAULT_MIC_DEVICE,
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

  // Never leave a capture process behind, even on a hard process.exit().
  const killOnExit = (): void => {
    child.kill("SIGKILL");
  };
  process.once("exit", killOnExit);

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
    process.removeListener("exit", killOnExit);
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
        // avfoundation capture can ignore SIGTERM; escalate so stop() can
        // never hang a shutdown path.
        const killTimer = setTimeout(() => {
          child.kill("SIGKILL");
        }, STOP_SIGKILL_AFTER_MS);
        child.once("close", () => {
          clearTimeout(killTimer);
          resolve();
        });
        child.kill("SIGTERM");
      }),
  };
}

/**
 * Resolve the default mic: the first PREFERRED_MIC_NAMES match by current
 * index, else the system default. Listing failures fall back silently; the
 * capture itself surfaces real errors.
 */
export async function resolveDefaultMicDevice(
  ffmpegPath = FFMPEG_PATH,
): Promise<{ spec: string; device?: AudioDevice }> {
  try {
    const devices = await listAudioDevices(ffmpegPath);
    for (const preferred of PREFERRED_MIC_NAMES) {
      const match = devices.find((device) =>
        device.name.toLowerCase().includes(preferred.toLowerCase()),
      );
      if (match) {
        return { spec: `:${match.index}`, device: match };
      }
    }
  } catch {
    // Fall through to the system default.
  }
  return { spec: DEFAULT_MIC_DEVICE };
}

/** Enumerate avfoundation audio input devices (index + name). */
export function listAudioDevices(
  ffmpegPath = FFMPEG_PATH,
): Promise<AudioDevice[]> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      ffmpegPath,
      ["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      reject(
        new Error(`ffmpeg failed to start (${ffmpegPath}): ${error.message}`),
      );
    });
    // ffmpeg exits nonzero after -list_devices by design; the listing is on
    // stderr either way.
    child.on("close", () => {
      resolve(parseAudioDevices(stderr));
    });
  });
}

function parseAudioDevices(stderr: string): AudioDevice[] {
  const devices: AudioDevice[] = [];
  let inAudioSection = false;
  for (const line of stderr.split("\n")) {
    if (line.includes("AVFoundation audio devices")) {
      inAudioSection = true;
      continue;
    }
    if (line.includes("AVFoundation video devices")) {
      inAudioSection = false;
      continue;
    }
    if (!inAudioSection) {
      continue;
    }
    const match = /\[(\d+)\]\s+(.+)$/.exec(line);
    if (match) {
      devices.push({ index: Number(match[1]), name: match[2]!.trim() });
    }
  }
  return devices;
}

/** Copy one frame out of the carry buffer; safe for any byte alignment. */
function decodeFrame(bytes: Buffer): Float32Array {
  const frame = new Float32Array(CAPTURE_FRAME_SAMPLES);
  for (let index = 0; index < CAPTURE_FRAME_SAMPLES; index += 1) {
    frame[index] = bytes.readFloatLE(index * BYTES_PER_SAMPLE);
  }
  return frame;
}
