import { CAPTURE_AUDIO_CONSTRAINTS } from "/capture-constraints.js";
import { concat, PreRollRing } from "/pre-roll.js";

/**
 * One open microphone through the capture worklet, in a context the caller
 * owns. Idle frames feed the pre roll ring; a take holds its pre roll plus
 * every frame until the worklet answers the take's flush, so no frame of one
 * take reaches the next. Every frame also reaches onFrame, for level meters
 * and voice detection. `rate` (16 kHz unless given) is the frames' sample
 * rate; `constraints` replace the shared capture constraints, as a voice copy
 * recording does to keep the voice unprocessed.
 */
export class MicCapture {
  constructor({ context, onFrame = () => {}, rate = 16_000, constraints = CAPTURE_AUDIO_CONSTRAINTS }) {
    this.context = context;
    this.onFrame = onFrame;
    this.rate = rate;
    this.constraints = constraints;
    this.node = null;
    this.stream = null;
    this.source = null;
    this.echoCancellation = constraints.echoCancellation;
    this.ring = new PreRollRing(rate);
    this.take = null;
    this.draining = [];
  }

  get open() {
    return Boolean(this.stream);
  }

  get recording() {
    return Boolean(this.take);
  }

  /** Opens the mic, or reopens it when the echo cancellation choice changed. */
  async start(echoCancellation = this.echoCancellation) {
    if (this.stream && echoCancellation === this.echoCancellation) return;
    this.stop();
    this.echoCancellation = echoCancellation;
    if (!this.node) {
      await this.context.audioWorklet.addModule("/capture-worklet.js");
      this.node = new AudioWorkletNode(this.context, "speak-easy-capture", { processorOptions: { targetRate: this.rate } });
      this.node.port.onmessage = (event) => this.frame(event.data);
      const silent = this.context.createGain();
      silent.gain.value = 0;
      this.node.connect(silent).connect(this.context.destination);
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { ...this.constraints, echoCancellation },
    });
    if (this.stream || echoCancellation !== this.echoCancellation) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    this.stream = stream;
    this.source = this.context.createMediaStreamSource(stream);
    this.source.connect(this.node);
  }

  stop() {
    this.source?.disconnect();
    this.source = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.ring.clear();
    this.take = null;
  }

  /** Starts a take led by the last PRE_ROLL_MS of idle audio; returns that pre roll. */
  beginTake() {
    const preRoll = this.ring.take();
    this.take = { frames: [preRoll], preRollSamples: preRoll.length };
    return preRoll;
  }

  /** Resolves with the take's samples once the worklet flushed its tail. */
  endTake() {
    const take = this.take;
    if (!take) return Promise.resolve(null);
    this.take = null;
    return new Promise((resolve) => {
      this.draining.push({ take, resolve });
      this.node.port.postMessage({ kind: "flush" });
    });
  }

  cancelTake() {
    this.take = null;
  }

  frame(data) {
    if (data?.kind !== "frame" || !(data.samples instanceof Float32Array)) return;
    const draining = this.draining[0];
    if (draining) {
      draining.take.frames.push(data.samples);
      if (data.flush === true) {
        this.draining.shift();
        draining.resolve({ samples: concat(draining.take.frames), preRollSamples: draining.take.preRollSamples });
      }
      return;
    }
    if (!this.stream) return;
    if (this.take) this.take.frames.push(data.samples);
    else this.ring.push(data.samples);
    this.onFrame(data);
  }
}
