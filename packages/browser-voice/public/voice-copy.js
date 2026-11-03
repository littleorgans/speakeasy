import { MicCapture } from "/mic-capture.js";

/**
 * The voice copy panel, shown for a model whose voices copy a reference:
 * Listen to a synthetic pick, or record your own reference at the context's
 * rate with no browser processing, hear it, forget it, and type the exact
 * words you read. The recording uploads once to the host, which holds it in
 * memory only; a reading names it by its hash.
 */
export const RECORDING = "recording";
export const MAX_RECORDING_MS = 30_000;
const MIN_RECORDING_MS = 1_000;
export const READ_THIS = "The window is open, and the room is quiet. We can take our time today, and read each sentence slowly and clearly.";
export const CONSENT = "Copy only your own voice or the synthetic references. Your recording stays in this server's memory " +
  "until you press Forget or the server restarts.";
/** Raw capture: echo cancellation, noise suppression and gain control would change the voice being copied. */
const RAW = Object.freeze({ channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false });

export class VoiceCopy {
  /**
   * audio() resolves the page's audio context (opened on a press); play(samples, rate)
   * is the page's replay path; onChange re-renders the page.
   */
  constructor({ document, root, audio, play, onChange = () => {}, fetch = globalThis.fetch }) {
    Object.assign(this, { document, root, audio, play, onChange, fetch });
    this.model = null;
    this.voice = null;
    this.recording = null;
    this.capture = null;
    this.build();
  }

  build() {
    const make = (tag, className, text) => {
      const node = this.document.createElement(tag);
      if (className) node.className = className;
      if (text) node.textContent = text;
      return node;
    };
    this.consent = make("p", "quiet", CONSENT);
    this.record = make("button", "", "Record reference");
    this.record.type = "button";
    this.record.addEventListener("click", () => void (this.capture ? this.stop() : this.start()));
    this.listen = make("button", "", "Listen");
    this.listen.type = "button";
    this.listen.addEventListener("click", () => void this.hear());
    this.forget = make("button", "", "Forget");
    this.forget.type = "button";
    this.forget.addEventListener("click", () => void this.discard());
    this.status = make("span", "quiet");
    this.status.setAttribute("role", "status");
    const buttons = make("p", "replay");
    buttons.append(this.record, this.listen, this.forget, this.status);
    this.words = make("label", "setting");
    this.transcript = make("textarea");
    this.transcript.rows = 3;
    this.transcript.value = READ_THIS;
    this.transcript.addEventListener("input", () => this.onChange());
    this.words.append(make("span", "label", "Transcript: read this aloud, or type exactly what you said"), this.transcript);
    this.root.replaceChildren(buttons, this.words, this.consent);
  }

  /** Shows the panel for a voice copy model and the chosen voice; hidden otherwise. */
  show(model, voiceId) {
    this.model = model;
    this.voice = model?.voices.find((voice) => String(voice.id) === String(voiceId)) ?? null;
    const copy = Boolean(model?.voices.some((voice) => voice.origin));
    this.root.hidden = !copy;
    if (!copy) return;
    const mine = this.voice?.id === RECORDING;
    for (const node of [this.record, this.forget, this.words]) node.hidden = !mine;
    this.record.textContent = this.capture ? "Stop" : this.recording ? "Record again" : "Record reference";
    this.forget.disabled = !this.recording;
    this.listen.disabled = mine ? !this.recording : !this.voice?.origin;
    if (!this.capture) {
      this.status.textContent = mine ? (this.recording ? `Recorded ${seconds(this.recording.samples.length / this.recording.rate)}`
        : "Nothing recorded yet.") : this.voice?.transcript ? `Says: ${this.voice.transcript}` : "";
    }
  }

  async start() {
    try {
      const context = await this.audio();
      this.capture = new MicCapture({ context, rate: context.sampleRate, constraints: RAW });
      await this.capture.start(false);
      this.capture.beginTake();
      const began = performance.now();
      this.ticker = setInterval(() => {
        const elapsed = performance.now() - began;
        this.status.textContent = `Recording ${seconds(elapsed / 1_000)} of ${seconds(MAX_RECORDING_MS / 1_000)}`;
        if (elapsed >= MAX_RECORDING_MS) void this.stop();
      }, 200);
    } catch (error) {
      this.capture?.stop();
      this.capture = null;
      this.status.textContent = `Microphone unavailable: ${error?.message ?? error}`;
    }
    this.refresh();
  }

  async stop() {
    const capture = this.capture;
    if (!capture) return;
    clearInterval(this.ticker);
    this.capture = null;
    const take = await capture.endTake();
    capture.stop();
    const rate = capture.rate;
    const samples = take?.samples.slice(0, Math.round(rate * MAX_RECORDING_MS / 1_000)) ?? new Float32Array();
    if (samples.length < rate * MIN_RECORDING_MS / 1_000) {
      this.status.textContent = "Too short. Record at least a second.";
      this.refresh();
      return;
    }
    if (this.recording) await this.discard();
    this.recording = { samples, rate, id: null };
    await this.upload();
    this.refresh();
  }

  /** Sends the recording to the host; resolves with its id, or null with the reason shown. */
  async upload(recording = this.recording) {
    if (!recording) return null;
    try {
      const response = await this.fetch(`/speech-output/references?sampleRate=${recording.rate}`, { method: "POST",
        headers: { "content-type": "application/octet-stream" }, body: recording.samples.slice().buffer });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? response.status);
      if (this.recording !== recording) {
        await this.fetch(`/speech-output/references/${body.id}`, { method: "DELETE" });
        return null;
      }
      recording.id = body.id;
      return body.id;
    } catch (error) {
      this.status.textContent = `Upload failed: ${error.message}`;
      return null;
    }
  }

  async discard() {
    const recording = this.recording;
    this.recording = null;
    if (recording?.id) await this.fetch(`/speech-output/references/${recording.id}`, { method: "DELETE" }).catch(() => {});
    this.refresh();
  }

  async hear() {
    if (this.voice?.id === RECORDING) {
      if (this.recording) this.play(this.recording.samples, this.recording.rate);
      return;
    }
    try {
      const context = await this.audio();
      const response = await this.fetch(`/speech-output/picks/${this.model.id}/${this.voice.id}`);
      if (!response.ok) throw new Error(response.status);
      const buffer = await context.decodeAudioData(await response.arrayBuffer());
      this.play(buffer.getChannelData(0), buffer.sampleRate);
    } catch (error) { this.status.textContent = `Listen failed: ${error.message}`; }
  }

  /**
   * What a reading of the chosen voice needs: nothing for a pick, the
   * recording's id and transcript for My recording, or an error to show
   * instead of sending. Words and voice needs the transcript.
   */
  request(copyMode, voiceId = this.voice?.id) {
    if (voiceId !== RECORDING) return {};
    const recording = this.recording;
    if (!recording?.id) return { error: "Record a reference first." };
    const transcript = this.transcript.value.trim();
    if (copyMode !== "voice-only" && !transcript) return { error: "Type the words you read, or choose Copy: Voice only." };
    return { reference: { id: recording.id, transcript }, restore: async () => {
      if (this.recording !== recording) return null;
      const id = await this.upload(recording);
      return id && this.recording === recording ? { id, transcript } : null;
    } };
  }

  refresh() {
    this.show(this.model, this.voice?.id);
    this.onChange();
  }
}

function seconds(value) {
  return `${value.toFixed(1)} s`;
}
