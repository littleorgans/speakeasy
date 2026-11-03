const BUSY_RETRY_MS = 150;
/** The host keeps a stopped voice's reservation until native work settles. */
const BUSY_RETRY_LIMIT = 60;
const STOP_REPLY_MS = 1_000;

/**
 * Reads text aloud over the host's /speech-output channel into one playback
 * worklet node in the caller's context. Times are browser estimates on the
 * performance clock: the context's output timestamp maps rendered audio to
 * the moment it reaches the output device. Keeps each reading's PCM for replay.
 * A reading may carry voice settings and a voice copy reference; the result
 * keeps the settings the host applied. When the host no longer holds the
 * reference, `restore` uploads it again, once, and the reading retries.
 */
export class VoiceReadback {
  constructor({ context, onChange = () => {} }) {
    this.context = context;
    this.onChange = onChange;
    this.node = null;
    this.socket = null;
    this.job = null;
    this.serial = 0;
  }

  get active() {
    return Boolean(this.job);
  }

  /** True from the first rendered sample until playback drains or stops. */
  get playing() {
    return Boolean(this.job?.firstAudioAt != null && !this.job.stopping);
  }

  async prepare() {
    if (this.node) return;
    await this.context.audioWorklet.addModule("/playback-worklet.js");
    this.node ??= new AudioWorkletNode(this.context, "speak-easy-playback", { outputChannelCount: [2] });
    this.node.connect(this.context.destination);
    this.node.port.onmessage = (event) => this.rendered(event.data);
  }

  /** Resolves when the reading drains, stops or fails; never rejects. */
  async speak({ selection, catalogueRevision, text, settings, reference, restore }) {
    this.stop();
    // A replacement may arrive before the stopped playback's clear reply.
    if (this.job) this.finish(this.job, { outcome: "stopped" });
    const job = { selection, catalogueRevision, text, settings, reference, restore, requestedAt: performance.now(), tries: 0,
      chunks: [], sampleRate: null, firstAudioAt: null };
    job.done = new Promise((resolve) => { job.resolve = resolve; });
    this.job = job;
    try {
      await this.prepare();
      if (this.job !== job) return job.done;
      await this.open();
      if (this.job === job) this.request(job);
    } catch (error) {
      this.finish(job, { outcome: "error", message: error.message });
    }
    return job.done;
  }

  /** Clears playback now; the result reports silence relative to `from`. */
  stop(from = performance.now()) {
    const job = this.job;
    if (!job || job.stopping) return;
    job.stopping = true;
    job.stopFrom = from;
    if (job.playbackId != null) this.node.port.postMessage({ kind: "clear", playbackId: job.playbackId });
    if (this.socket?.readyState === WebSocket.OPEN && job.requestId) {
      this.socket.send(JSON.stringify({ type: "cancel-sample", requestId: job.requestId }));
    }
    if (job.playbackId == null || !job.started) this.finish(job, { outcome: "stopped" });
    // The worklet answers a clear at its next render quantum; never wait longer.
    else setTimeout(() => this.finish(job, { outcome: "stopped" }), STOP_REPLY_MS);
  }

  open() {
    if (this.socket?.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${scheme}://${location.host}/speech-output`);
      socket.binaryType = "arraybuffer";
      this.socket = socket;
      socket.addEventListener("open", () => resolve(), { once: true });
      socket.addEventListener("error", () => reject(new Error("Could not open the speech output channel.")), { once: true });
      socket.addEventListener("message", (event) => {
        if (event.data instanceof ArrayBuffer) this.audio(event.data);
        else this.accept(JSON.parse(event.data));
      });
      socket.addEventListener("close", () => {
        if (this.socket === socket) this.socket = null;
        if (this.job) this.finish(this.job, { outcome: "error", message: "Speech output channel closed." });
      });
    });
  }

  request(job) {
    job.requestId = `readback-${Date.now()}-${++this.serial}`;
    this.socket.send(JSON.stringify({ type: "select-output", requestId: job.requestId,
      catalogueRevision: job.catalogueRevision, selection: job.selection }));
  }

  accept(event) {
    const job = this.job;
    if (!job || job.stopping && event.type !== "sample-complete") return;
    if (event.type === "playback") {
      if (event.playbackId !== job.playbackId) return;
      if (event.action === "start" && !job.started) {
        job.started = true;
        job.sampleRate = event.sampleRate;
        this.node.port.postMessage({ kind: "start", playbackId: job.playbackId, sampleRate: event.sampleRate, reportFirstRender: true });
      } else if (event.action === "end" || event.action === "clear") {
        this.node.port.postMessage({ kind: event.action, playbackId: job.playbackId });
      }
      return;
    }
    if (event.requestId !== job.requestId && event.requestId !== null) return;
    if (event.type === "output-selected") {
      this.socket.send(JSON.stringify({ type: "speak-sample", requestId: job.requestId, catalogueRevision: event.catalogueRevision,
        selectionRevision: event.selectionRevision, selection: job.selection, text: job.text,
        ...(job.settings ? { settings: job.settings } : {}), ...(job.reference ? { reference: job.reference } : {}) }));
    } else if (event.type === "sample-accepted") {
      job.playbackId = event.playbackId;
      job.applied = event.settings;
    } else if (event.type === "sample-phase") {
      // The host reports a voice load as the voice opens, so a stopped reading keeps it.
      if (Number.isFinite(event.loadMs)) job.loadMs = event.loadMs;
    } else if (event.type === "sample-error") {
      if (event.code === "BUSY" && ++job.tries < BUSY_RETRY_LIMIT) {
        setTimeout(() => { if (this.job === job && !job.stopping) this.request(job); }, BUSY_RETRY_MS);
      } else if (event.code === "REFERENCE_MISSING" && job.restore && !job.restored) {
        job.restored = true;
        void job.restore().then((reference) => {
          if (this.job !== job || job.stopping) return;
          if (!reference) this.finish(job, { outcome: "error", message: event.message });
          else { job.reference = reference; this.request(job); }
        });
      } else this.finish(job, { outcome: "error", message: event.message });
    } else if (event.type === "sample-complete") {
      const hostFirstAudio = event.timing?.serverFirstAudioReadyMs;
      job.hostFirstAudioMs = hostFirstAudio?.status === "measured" ? hostFirstAudio.ms : null;
      if (event.outcome === "produced" && !job.started) this.finish(job, { outcome: "finished" });
      if (event.outcome === "produced") job.produced = true;
      this.acknowledge(job);
    }
  }

  audio(packet) {
    const job = this.job;
    if (!job?.started || job.stopping || packet.byteLength < 9 || (packet.byteLength - 9) % 4) return;
    const header = new DataView(packet);
    if (header.getUint8(0) !== 1 || header.getUint32(1, true) !== job.playbackId) return;
    const samples = new Float32Array(packet.slice(9));
    job.chunks.push(samples.slice());
    this.node.port.postMessage({ kind: "audio", playbackId: job.playbackId, samples }, [samples.buffer]);
  }

  rendered(message) {
    const job = this.job;
    if (!job || message.playbackId !== job.playbackId) return;
    if (message.kind === "first-render" && job.firstAudioAt == null && !job.stopping) {
      job.firstAudioAt = this.outputTime(message.audioTime);
      this.onChange();
    } else if (message.kind === "drained") {
      job.drained = message;
      if (job.stopping) {
        const silentAt = this.outputTime(message.audioTime);
        this.finish(job, { outcome: "stopped", stopMs: silentAt == null ? null : Math.max(0, silentAt - job.stopFrom) });
      } else {
        job.endAt = this.outputTime(message.audioTime);
        this.acknowledge(job);
      }
    }
  }

  /** The output device time of a context time, or null without a usable clock pair. */
  outputTime(audioTime) {
    let stamp;
    try { stamp = this.context.getOutputTimestamp?.(); } catch { return null; }
    if (!Number.isFinite(audioTime) || !(stamp?.contextTime > 0) || !(stamp?.performanceTime > 0)) return null;
    return stamp.performanceTime + (audioTime - stamp.contextTime) * 1_000;
  }

  acknowledge(job) {
    if (!job.produced || !job.drained || job.acknowledged) return;
    job.acknowledged = true;
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "playback-drained", playbackId: job.playbackId, audioEndMs: job.drained.audioEndMs }));
    }
    this.finish(job, { outcome: "finished" });
  }

  finish(job, result) {
    if (this.job !== job) return;
    this.job = null;
    const since = (at) => at == null ? null : Math.max(0, at - job.requestedAt);
    const samples = new Float32Array(job.chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of job.chunks) { samples.set(chunk, offset); offset += chunk.length; }
    job.resolve({ outcome: result.outcome, message: result.message ?? "", firstAudioMs: since(job.firstAudioAt),
      totalMs: result.outcome === "finished" ? since(job.endAt) : null, stopMs: result.stopMs ?? null,
      hostFirstAudioMs: job.hostFirstAudioMs ?? null, loadMs: job.loadMs ?? null, samples, sampleRate: job.sampleRate,
      settings: job.applied ?? null });
    this.onChange();
  }
}
