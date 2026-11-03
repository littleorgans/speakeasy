import { CAPTURE_AUDIO_CONSTRAINTS } from "/capture-constraints.js";
import { concat, PreRollRing } from "/pre-roll.js";

/**
 * One capture graph through the capture worklet, in a context the caller
 * owns: the worklet node, the mic, the idle pre roll ring and the current
 * take. Idle frames feed the ring; a take holds its pre roll plus every frame
 * until the worklet answers the take's flush, so no frame of one take reaches
 * the next. Each live frame reaches onFrame first, for level meters, voice
 * detection and streaming; onFrame returning false keeps the frame out of the
 * ring and the take. `rate` (16 kHz unless given) is the frames' sample rate;
 * `constraints` replace the shared capture constraints, as a voice copy
 * recording does to keep the voice unprocessed; `deviceId` picks the mic.
 * Stop discards any take and every late mic acquisition.
 */
export class CaptureSession {
  constructor({ context, onFrame = () => {}, rate = 16_000, constraints = CAPTURE_AUDIO_CONSTRAINTS, deviceId = "" }) {
    this.context = context;
    this.onFrame = onFrame;
    this.rate = rate;
    this.constraints = constraints;
    this.deviceId = deviceId;
    /** Resolves once the worklet node exists; reset on failure and by close. */
    this.loading = null;
    this.node = null;
    this.stream = null;
    this.source = null;
    this.echoCancellation = constraints.echoCancellation;
    /** The pending acquisition and its echo choice, shared until it settles. */
    this.acquiring = null;
    /** Bumped by stop, so a late acquisition is discarded. */
    this.generation = 0;
    this.ring = new PreRollRing(rate);
    this.take = null;
    /** Unanswered flushes, oldest first; a discarded take's entry holds no take. */
    this.flushes = [];
  }

  get open() {
    return Boolean(this.stream);
  }

  get recording() {
    return Boolean(this.take);
  }

  /** True while an ended take waits for its flush reply. */
  get draining() {
    return this.flushes.some((entry) => entry.take);
  }

  /** Label of the open mic; empty when closed. */
  get device() {
    return this.stream?.getAudioTracks()[0]?.label ?? "";
  }

  /** Opens the mic, or reopens it when the echo cancellation choice changed. */
  async start(echoCancellation = this.echoCancellation) {
    if (this.stream && echoCancellation === this.echoCancellation) return;
    if (this.acquiring && echoCancellation === this.echoCancellation) return this.acquiring;
    // A cold start keeps a take begun before it; a reopen discards it.
    if (this.stream || this.acquiring) this.stop();
    this.echoCancellation = echoCancellation;
    const acquiring = this.acquire(this.generation, echoCancellation);
    this.acquiring = acquiring;
    return acquiring;
  }

  async acquire(generation, echoCancellation) {
    try {
      await this.load();
      if (generation !== this.generation) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { ...(this.deviceId ? { deviceId: { exact: this.deviceId } } : {}), ...this.constraints, echoCancellation },
      });
      if (generation !== this.generation) {
        // The mic stopped while this one opened; a later start owns capture.
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      this.stream = stream;
      this.source = this.context.createMediaStreamSource(stream);
      this.source.connect(this.node);
    } catch (error) {
      if (generation === this.generation) throw error;
    } finally {
      if (generation === this.generation) this.acquiring = null;
    }
  }

  /** Loads the worklet and creates its node once; a failed load retries at the next start. */
  load() {
    const loading = this.loading ??= this.context.audioWorklet.addModule("/capture-worklet.js").then(() => {
      if (this.loading !== loading) return;
      this.node = new AudioWorkletNode(this.context, "speak-easy-capture", { processorOptions: { targetRate: this.rate } });
      this.node.port.onmessage = (event) => this.frame(event.data);
      const silent = this.context.createGain();
      silent.gain.value = 0;
      this.node.connect(silent).connect(this.context.destination);
    }, (error) => {
      if (this.loading === loading) this.loading = null;
      throw error;
    });
    return loading;
  }

  /** Releases the mic and ring and discards any take; the graph stays for reuse. */
  stop() {
    this.generation += 1;
    this.acquiring = null;
    this.source?.disconnect();
    this.source = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.ring.clear();
    this.take = null;
    for (const entry of this.flushes) {
      entry.take = null;
      entry.resolve(null);
    }
    // The worklet's buffered partial belongs to no take.
    if (this.node) this.flush({ take: null, resolve() {} });
  }

  /** Stops, then releases the graph, for an owner that closes its context. */
  close() {
    this.stop();
    this.node?.disconnect();
    this.node = null;
    this.loading = null;
    this.flushes = [];
  }

  /** Returns and clears the last PRE_ROLL_MS of idle audio. */
  takePreRoll() {
    return this.ring.take();
  }

  /** Starts a take led by the idle pre roll; returns that pre roll. */
  beginTake() {
    const preRoll = this.takePreRoll();
    this.take = { frames: [preRoll], preRollSamples: preRoll.length };
    return preRoll;
  }

  /** Resolves with the take's samples once the worklet flushed its tail; null when stopped first. */
  endTake() {
    const take = this.take;
    if (!take) return Promise.resolve(null);
    this.take = null;
    // Before the worklet loads, it holds nothing of this take.
    if (!this.node) return Promise.resolve(takeAudio(take));
    return new Promise((resolve) => this.flush({ take, resolve }));
  }

  flush(entry) {
    this.flushes.push(entry);
    this.node.port.postMessage({ kind: "flush" });
  }

  frame(data) {
    if (data?.kind !== "frame" || !(data.samples instanceof Float32Array)) return;
    const head = this.flushes[0];
    if (head) {
      head.take?.frames.push(data.samples);
      if (data.flush === true) {
        this.flushes.shift();
        if (head.take) head.resolve(takeAudio(head.take));
      }
      return;
    }
    if (!this.stream || this.onFrame(data) === false) return;
    if (this.take) this.take.frames.push(data.samples);
    else this.ring.push(data.samples);
  }
}

function takeAudio(take) {
  return { samples: concat(take.frames), preRollSamples: take.preRollSamples };
}
