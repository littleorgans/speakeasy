/** A typed text preview has its own socket and playback, without mic or LLM. */
export class SpeechOutputControls {
  constructor({ activeRoom = () => false } = {}) {
    this.activeRoom = activeRoom;
    this.catalog = null;
    this.draft = null;
    this.serial = 0;
    this.job = null;
    this.status = "Loading voices";
    this.elements = Object.fromEntries([
      "output-provider", "output-model", "output-model-summary", "output-voice", "output-transfer", "output-mode-note",
      "output-text", "output-try", "output-stop", "output-status", "output-error", "output-timing",
    ].map((id) => [id, document.getElementById(id)]));
    for (const key of ["output-provider", "output-model", "output-voice"]) {
      this.elements[key].addEventListener("change", () => {
        this.stop();
        this.choose(key);
      });
    }
    this.elements["output-text"].addEventListener("input", () => { this.stop(); this.render(); });
    this.elements["output-try"].addEventListener("click", () => void this.tryVoice());
    this.elements["output-stop"].addEventListener("click", () => this.stop());
  }

  async load(stored) {
    this.draft = stored ?? null;
    try {
      const response = await fetch("/speech-output/models");
      const catalog = await response.json();
      if (!response.ok) throw new Error(catalog.message ?? catalog.error ?? "Voices unavailable");
      this.catalog = catalog;
      this.models = catalog.models.filter((model) => model.provider.kind === "local" || model.configured === true);
      this.draft ??= catalog.default;
      this.status = "Ready to try";
      this.renderChoices();
    } catch (error) {
      this.error = error.message;
      this.status = "Voices unavailable";
    }
    this.render();
  }

  model() {
    return this.models?.find((model) => model.provider.kind === this.draft?.provider?.kind && model.id === this.draft.model);
  }

  selection() {
    const model = this.model();
    if (this.draft?.schema !== 1 || !model?.voices.some((voice) => voice.id === this.draft.voice) ||
        !Number.isFinite(this.draft.speed) || this.draft.speed < model.speed.min || this.draft.speed > model.speed.max) return null;
    return { ...this.draft, provider: { ...this.draft.provider } };
  }

  available() {
    return this.selection() && this.model().readiness?.state === "ready";
  }

  choose(changed) {
    const provider = this.elements["output-provider"].value;
    const candidates = this.models.filter((model) => model.provider.kind === provider);
    const model = changed === "output-provider" ? candidates[0] : candidates.find((entry) => entry.id === this.elements["output-model"].value);
    if (!model) return;
    const voice = changed === "output-voice" ? model.voices.find((entry) => String(entry.id) === this.elements["output-voice"].value)?.id : model.voices.find((entry) => entry.id === this.draft?.voice)?.id ?? model.defaultVoice;
    this.draft = { schema: 1, provider: { kind: provider }, model: model.id, voice, speed: model.defaultSpeed };
    this.error = "";
    this.status = "Ready to try";
    this.renderChoices();
    this.render();
  }

  renderChoices() {
    const provider = this.draft?.provider?.kind;
    const kinds = [...new Set(this.models.map((model) => model.provider.kind))];
    this.fill("output-provider", kinds.map((kind) => ({ id: kind, label: kind === "local" ? "Local" : kind })), provider);
    const models = this.models.filter((model) => model.provider.kind === provider);
    this.fill("output-model", models, this.draft?.model);
    this.fill("output-voice", this.model()?.voices ?? [], this.draft?.voice);
    if (!this.selection()) {
      this.error = "Saved speech output is unavailable. Choose a supported model and voice.";
      this.status = "Selection unavailable";
    }
  }

  fill(id, entries, value) {
    const options = entries.map((entry) => new Option(entry.label, String(entry.id)));
    if (!entries.some((entry) => entry.id === value)) options.unshift(new Option("Unavailable: choose again", ""));
    this.elements[id].replaceChildren(...options);
    this.elements[id].value = value == null || !entries.some((entry) => entry.id === value) ? "" : String(value);
    this.elements[id].disabled = false;
  }

  renderMode(engine) {
    this.elements["output-mode-note"].textContent = engine === "realtime"
      ? "Realtime replies use the Realtime voice above."
      : "Speech output applies the next time you enter the room.";
    this.render();
  }

  renderRoomPrivacy(settings, element) {
    const input = "Microphone audio is transcribed on this Mac.";
    const outputKind = settings.speechOutput?.provider?.kind;
    element.textContent = settings.engine === "realtime"
      ? `${input} Transcript text is sent to OpenAI Realtime; reply audio returns.`
      : `${input} Transcript text is sent to Mercury. ${outputKind === "local"
        ? "Reply speech is generated on this Mac."
        : outputKind ? `Reply text is sent to ${outputKind}; generated audio returns.` : "Reply speech uses the host configured output."}`;
  }

  textError() {
    const text = this.elements["output-text"].value;
    if (!text.trim()) return "Enter text to try.";
    if (this.catalog?.maxTextLength && text.length > this.catalog.maxTextLength) return `Use at most ${this.catalog.maxTextLength} characters.`;
    return "";
  }

  render() {
    const local = this.draft?.provider?.kind === "local";
    this.elements["output-transfer"].textContent = !this.draft ? "" : local
      ? "Speech is generated on this Mac."
      : `Text is sent to ${this.draft.provider.kind}. Generated audio comes back to this Mac.`;
    const busy = this.activeRoom();
    const unavailable = this.catalog && this.selection() && !this.available();
    this.renderModelSummary();
    this.elements["output-status"].textContent = busy ? "End the room to try a voice." : unavailable ? local ? "Voice needs setup" : "Voice unavailable" : this.status;
    this.elements["output-error"].textContent = this.error || (unavailable ? this.model().readiness.reason : this.catalog ? this.textError() : "") || "";
    this.elements["output-try"].disabled = Boolean(this.job) || busy || !this.available() || Boolean(this.textError());
    this.elements["output-stop"].hidden = !this.job;
    this.elements["output-timing"].textContent = `Browser TTFA: ${this.browserMs == null ? "Unavailable" : `${Math.round(this.browserMs)} ms`} · Host first audio: ${this.hostMs == null ? "Unavailable" : `${Math.round(this.hostMs)} ms`}. Browser rendering only; speaker latency is unmeasured.${this.applied ? ` ${this.applied.model} · ${this.applied.voice}` : ""}`;
  }

  renderModelSummary() {
    const model = this.model();
    const root = this.elements["output-model-summary"];
    root.hidden = !model;
    if (!model) return;
    const name = document.createElement("strong");
    name.textContent = model.label;
    const facts = document.createElement("p");
    facts.className = "output-note";
    facts.textContent = [model.id === this.catalog.default.model && model.provider.kind === "local" ? "Recommended" : "", model.streaming === "pcm" ? "Streams PCM" : model.streaming === "sentence" ? "Streams by sentence" : "", Number.isFinite(model.sampleRate) ? `${model.sampleRate / 1_000} kHz` : ""].filter(Boolean).join(" · ");
    const status = document.createElement("p");
    status.className = "output-note";
    status.textContent = model.readiness.state === "ready" ? model.provider.kind === "local" ? "Ready to load" : "Configured" : `Needs setup: ${model.readiness.reason ?? model.readiness.state}`;
    root.replaceChildren(name, facts, status);
  }

  async tryVoice() {
    if (this.activeRoom() || !this.available() || this.textError()) { this.render(); return; }
    this.stop();
    const job = { requestId: `voice-${Date.now()}-${++this.serial}`, selection: this.selection(), text: this.elements["output-text"].value, submittedAt: performance.now() };
    this.job = job;
    this.error = "";
    this.applied = null;
    this.browserMs = null;
    this.hostMs = null;
    this.status = "Preparing voice. Waiting for audio.";
    this.render();
    try {
      const context = new AudioContext({ latencyHint: "interactive" });
      job.context = context;
      await context.resume();
      if (this.job !== job) return;
      if (context.state !== "running") throw new Error("Audio playback is blocked. Try voice again to resume it.");
      context.addEventListener("statechange", () => {
        if (this.job !== job || job.rendered || context.state === "running") return;
        job.clockBlocked = true;
        this.status = "Audio playback paused. Browser TTFA unavailable.";
        this.render();
      });
      await context.audioWorklet.addModule("/playback-worklet.js");
      if (this.job !== job) return;
      job.node = new AudioWorkletNode(context, "speak-easy-playback", { outputChannelCount: [2] });
      job.node.connect(context.destination);
      // Both anchors are local. No server clock is subtracted from this clock.
      job.anchor = { audio: context.currentTime, monotonic: performance.now() };
      job.node.port.onmessage = (event) => this.rendered(job, event.data);
      this.openSocket(job);
    } catch (error) { this.fail(job, error.message); }
  }

  openSocket(job) {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/speech-output`);
    socket.binaryType = "arraybuffer";
    job.socket = socket;
    socket.addEventListener("open", () => {
      if (this.job !== job) { socket.close(); return; }
      socket.send(JSON.stringify({ type: "select-output", requestId: job.requestId, catalogueRevision: this.catalog.revision, selection: job.selection }));
    });
    socket.addEventListener("message", (event) => {
      if (this.job !== job) return;
      if (event.data instanceof ArrayBuffer) this.audio(job, event.data);
      else {
        try { this.accept(job, JSON.parse(event.data)); }
        catch { this.fail(job, "The speech host sent an unreadable event."); }
      }
    });
    socket.addEventListener("error", () => this.fail(job, "Could not open the speech output channel."));
    socket.addEventListener("close", () => { if (this.job === job) this.fail(job, "Speech output channel closed before completion."); });
  }

  accept(job, event) {
    if (this.job !== job) return;
    if (event.type === "playback") {
      if (event.requestId !== job.requestId || event.playbackId !== job.playbackId) return;
      if (event.action === "start") {
        if (job.started) return;
        job.started = true;
        job.sampleRate = event.sampleRate;
        job.node.port.postMessage({ kind: "start", playbackId: event.playbackId, sampleRate: event.sampleRate, reportFirstRender: true });
      } else if (event.playbackId === job.playbackId) {
        if (event.action === "end" || event.action === "clear") job.ended = true;
        job.node.port.postMessage({ kind: event.action, playbackId: event.playbackId });
        // The host clears audio before sending the terminal error/cancellation.
        if (event.action === "clear") job.cleared = true;
      }
      this.render();
      return;
    }
    if (event.requestId !== job.requestId && !(event.type === "sample-error" && event.requestId === null)) return;
    if (event.type === "output-selected") {
      if (job.selectionSent) return;
      job.selectionSent = true;
      job.socket.send(JSON.stringify({ type: "speak-sample", requestId: job.requestId, catalogueRevision: event.catalogueRevision, selectionRevision: event.selectionRevision, selection: job.selection, text: job.text }));
    } else if (event.type === "sample-accepted") {
      if (job.playbackId != null) return;
      job.playbackId = event.playbackId;
      job.generation = event.generation;
      this.applied = event.selection;
    } else if (event.playbackId != null && event.playbackId !== job.playbackId) return;
    else if (event.type === "sample-phase") this.status = event.phase === "opening" ? "Preparing voice. Waiting for audio." : "Waiting for audio";
    else if (event.type === "sample-error") { this.fail(job, event.message); return; }
    else if (event.type === "sample-complete") {
      if (event.outcome === "cancelled") { this.stop(); return; }
      job.produced = true;
      job.complete = event.outcome === "drained";
      const observation = event.timing?.serverFirstAudioReadyMs;
      this.hostMs = observation?.status === "measured" ? observation.ms : null;
      this.acknowledgeDrain(job);
      if (event.outcome === "produced" && !event.segments.length) {
        this.job = null;
        this.dispose(job);
        this.status = "No audio produced. Browser TTFA unavailable.";
      }
      this.finish(job);
    }
    this.render();
  }

  audio(job, packet) {
    if (this.job !== job || !job.started || job.ended || packet.byteLength < 9 || (packet.byteLength - 9) % 4) return;
    const header = new DataView(packet);
    if (header.getUint8(0) !== 1 || header.getUint32(1, true) !== job.playbackId || header.getUint32(5, true) !== job.sampleRate) return;
    const samples = new Float32Array(packet.slice(9));
    if (!samples.every(Number.isFinite)) { this.fail(job, "The speech host sent invalid audio."); return; }
    job.node.port.postMessage({ kind: "audio", playbackId: job.playbackId, sampleRate: job.sampleRate, samples }, [samples.buffer]);
  }

  rendered(job, message) {
    if (this.job !== job || job.cleared || message.playbackId !== job.playbackId) return;
    if (message.kind === "first-render" && this.browserMs == null && Number.isFinite(message.audioTime)) {
      const renderedAt = job.anchor.monotonic + (message.audioTime - job.anchor.audio) * 1_000;
      const elapsed = renderedAt - job.submittedAt;
      if (elapsed >= 0 && job.context.state === "running" && !job.clockBlocked) this.browserMs = elapsed;
      job.rendered = true;
      this.status = "Playing";
    } else if (message.kind === "drained") {
      job.drained = true;
      job.audioEndMs = message.audioEndMs;
      this.acknowledgeDrain(job);
      this.finish(job);
    }
    this.render();
  }

  acknowledgeDrain(job) {
    if (this.job !== job || !job.produced || !job.drained || job.drainSent || job.socket.readyState !== WebSocket.OPEN) return;
    job.drainSent = true;
    job.socket.send(JSON.stringify({ type: "playback-drained", playbackId: job.playbackId, audioEndMs: job.audioEndMs }));
  }

  finish(job) {
    if (this.job !== job || !job.complete || !job.drained) return;
    this.job = null;
    this.dispose(job);
    this.status = this.browserMs == null ? "Finished. No nonzero audio rendered; browser TTFA unavailable." : "Finished";
  }

  fail(job, message) {
    if (this.job !== job) return;
    this.stop();
    this.error = message;
    this.status = "Voice unavailable";
    this.render();
  }

  stop() {
    const job = this.job;
    this.job = null;
    if (!job) return;
    // Close local delivery before sending cancellation or awaiting the server.
    if (job.playbackId != null && !job.cleared) job.node?.port.postMessage({ kind: "clear", playbackId: job.playbackId });
    if (job.socket?.readyState === WebSocket.OPEN) job.socket.send(JSON.stringify({ type: "cancel-sample", requestId: job.requestId }));
    this.dispose(job);
    this.status = "Stopped";
    this.render();
  }

  dispose(job) {
    job.socket?.close();
    job.node?.disconnect();
    if (job.context && job.context.state !== "closed") void job.context.close();
  }
}

export const SETTINGS_KEY = "speak-easy.browser-room";

export function loadRoomSettings(elements, storage) {
  const defaults = {
    mode: "natural",
    engine: elements.engine.value,
    pauseMs: Number(elements.pause.value),
    voice: elements.voice.value,
    microphoneId: "",
    systemPrompt: "",
    barge: true,
  };
  try {
    const stored = { ...defaults, ...JSON.parse(storage.getItem(SETTINGS_KEY) ?? "{}") };
    const engines = [...elements.engine.options].map((option) => option.value);
    const voices = [...elements.voice.options].map((option) => option.value);
    return {
      mode: stored.mode === "hold" ? "hold" : "natural",
      engine: engines.includes(stored.engine) ? stored.engine : defaults.engine,
      pauseMs: clampPause(stored.pauseMs, elements.pause),
      voice: voices.includes(stored.voice) ? stored.voice : defaults.voice,
      microphoneId: typeof stored.microphoneId === "string" ? stored.microphoneId : "",
      systemPrompt: typeof stored.systemPrompt === "string" ? stored.systemPrompt : "",
      barge: stored.barge !== false,
      ...(stored.speechOutput ? { speechOutput: stored.speechOutput } : {}),
    };
  } catch {
    return defaults;
  }
}


export function clampPause(value, input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const pause = Number(value);
  return Math.min(max, Math.max(min, Number.isFinite(pause) ? Math.round(pause) : Number(input.value)));
}
