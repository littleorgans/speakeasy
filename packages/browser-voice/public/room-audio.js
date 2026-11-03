import { CaptureSession } from "/capture-session.js";

const AUDIO_HEADER_BYTES = 9;
const AUDIO_PACKET_KIND = 1;

/**
 * The room's audio graph: an AudioContext per open room, a CaptureSession on
 * the chosen mic and the playback worklet that plays the host's audio
 * packets. onFrame decides what each captured frame does, as the session's
 * hook; onDrained receives each drained message from the playback worklet.
 * Close invalidates a pending start, so no node is created on a closed
 * context.
 */
export class RoomAudio {
  constructor({ onFrame, onDrained }) {
    this.onFrame = onFrame;
    this.onDrained = onDrained;
    this.context = null;
    this.capture = null;
    this.playback = null;
    /** Bumped by close, so a pending start stops. */
    this.generation = 0;
  }

  get open() {
    return Boolean(this.capture?.open);
  }

  /** Opens a fresh graph on the mic `deviceId` (system default when empty); call after close. */
  async start(deviceId) {
    const generation = this.generation;
    const context = new AudioContext({ latencyHint: "interactive" });
    this.context = context;
    this.capture = new CaptureSession({ context, deviceId, onFrame: this.onFrame });
    await Promise.all([this.capture.start(), context.audioWorklet.addModule("/playback-worklet.js")]);
    if (generation !== this.generation) return;
    this.playback = new AudioWorkletNode(context, "speak-easy-playback", {
      outputChannelCount: [2],
    });
    this.playback.connect(context.destination);
    this.playback.port.onmessage = (event) => {
      if (event.data?.kind === "drained") this.onDrained(event.data);
    };
    await context.resume();
  }

  async close() {
    this.generation += 1;
    this.capture?.close();
    this.capture = null;
    this.playback?.disconnect();
    this.playback = null;
    const context = this.context;
    this.context = null;
    if (context && context.state !== "closed") await context.close();
  }

  /** Returns and clears the idle pre roll that leads a hold take. */
  takePreRoll() {
    return this.capture?.takePreRoll() ?? new Float32Array(0);
  }

  /** Forwards a host playback command (start, end, clear) to the playback worklet. */
  command({ action, playbackId, sampleRate }) {
    this.playback?.port.postMessage({ kind: action, playbackId, sampleRate });
  }

  /** Plays one binary host packet: kind, playback id and sample rate, then float samples. */
  play(packet) {
    if (packet.byteLength < AUDIO_HEADER_BYTES || !this.playback) return;
    const header = new DataView(packet);
    if (header.getUint8(0) !== AUDIO_PACKET_KIND) return;
    const playbackId = header.getUint32(1, true);
    const sampleRate = header.getUint32(5, true);
    const payload = packet.slice(AUDIO_HEADER_BYTES);
    if (payload.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) return;
    const samples = new Float32Array(payload);
    this.playback.port.postMessage(
      { kind: "audio", playbackId, sampleRate, samples },
      [samples.buffer],
    );
  }
}
