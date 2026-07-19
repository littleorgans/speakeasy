import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AudioSegment } from "./contract.ts";
import { writeWav } from "./synth.ts";

/**
 * Continuous audio sink for streamed TTS: fed one AudioSegment at a time as
 * synthesis produces them, so segment i plays while segment i+1 synthesizes.
 * Promoted out of the stream demo so every consumer (demo, sweep quality tool,
 * convo-engine) shares one player instead of re-implementing playback.
 *
 * ffplay is preferred: a single long-lived process reads raw f32le PCM from its
 * stdin, so playback is gapless across sentences (stream.ts already trims the
 * model's ragged end padding). afplay is the fallback when ffplay is absent; it
 * plays one temp wav per segment and keeps the per-spawn gap.
 */

/** Silence (ms) written on open() so CoreAudio opens off the request path. */
export const SINK_PRIMER_MS = 150;
/** Measured ffplay CoreAudio open latency; the cold-sink penalty in honest TTFA. */
export const DEVICE_OPEN_EST_MS = 500;

export type SegmentPlayer = {
  kind: string;
  /** Pre-open the device (off the request path) so it is hot by first write. */
  open(): void;
  write(segment: AudioSegment): void;
  /** Stop playback immediately, dropping buffered audio (barge-in). */
  interrupt(): void;
  end(): Promise<void>;
};

/** ffplay when available (gapless PCM stdin), else the afplay per-segment fallback. */
export function createSegmentPlayer(sampleRate: number): SegmentPlayer {
  return hasFfplay() ? ffplayPlayer(sampleRate) : afplayPlayer(sampleRate);
}

export function hasFfplay(): boolean {
  return spawnSync("ffplay", ["-version"], { stdio: "ignore" }).status === 0;
}

function ffplayPlayer(sampleRate: number): SegmentPlayer {
  let child: ChildProcess | undefined;
  let done: Promise<void> = Promise.resolve();
  let stopped = false;
  const ensure = (): ChildProcess => {
    if (child) {
      return child;
    }
    child = spawn(
      "ffplay",
      // ffplay is a playback tool: channels come from -ch_layout, not -ac.
      // -nostats/-nodisp keep it headless; -autoexit quits at stdin EOF.
      // prettier-ignore
      [
        "-hide_banner", "-loglevel", "error", "-nostats", "-nodisp",
        "-autoexit", "-f", "f32le", "-ar", String(sampleRate),
        "-ch_layout", "mono", "-i", "pipe:0",
      ],
      { stdio: ["pipe", "ignore", "ignore"] },
    );
    child.stdin?.on("error", () => {}); // report via exit, not an EPIPE throw
    done = onExit(child, "ffplay");
    return child;
  };
  return {
    kind: "ffplay",
    open() {
      if (stopped) {
        return;
      }
      const frames = Math.round((sampleRate * SINK_PRIMER_MS) / 1000);
      ensure().stdin?.write(Buffer.alloc(frames * Float32Array.BYTES_PER_ELEMENT));
    },
    write(segment) {
      if (stopped) {
        return;
      }
      const { samples } = segment;
      ensure().stdin?.write(
        Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength),
      );
    },
    interrupt() {
      // SIGKILL drops whatever ffplay has buffered so sound stops now, not at
      // the end of the current sentence.
      stopped = true;
      child?.kill("SIGKILL");
    },
    async end() {
      if (!stopped) {
        child?.stdin?.end();
      }
      await done.catch(() => {});
    },
  };
}

/** Fallback: one temp wav per segment played by afplay. Retains the spawn gap. */
function afplayPlayer(sampleRate: number): SegmentPlayer {
  let queue: Promise<void> = Promise.resolve();
  let dir: string | undefined;
  let index = 0;
  let stopped = false;
  let current: ChildProcess | undefined;
  const scratch = (): string => {
    dir ??= mkdtempSync(join(tmpdir(), "speakeasy-tts-"));
    return join(dir, `seg-${index++}.wav`);
  };
  return {
    kind: "afplay (fallback; per-segment spawn gap, no pre-open)",
    open() {}, // afplay plays whole wav files; nothing to pre-open
    write(segment) {
      if (stopped) {
        return;
      }
      const path = scratch();
      queue = queue.then(async () => {
        if (stopped) {
          return;
        }
        await writeWav(path, { samples: segment.samples, sampleRate });
        current = spawn("afplay", [path], { stdio: "ignore" });
        await onExit(current, "afplay").catch(() => {});
      });
    },
    interrupt() {
      stopped = true;
      current?.kill("SIGKILL");
    },
    end: () => queue.catch(() => {}),
  };
}

/** Resolve when the child exits 0; reject on spawn error or nonzero exit. */
function onExit(child: ChildProcess, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${name} exited with code ${code ?? "unknown"}`)),
    );
  });
}
