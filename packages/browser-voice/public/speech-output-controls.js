import { formatMs, formatObservation, transcriptStatus } from "/speech-observations.js";

/** A typed text preview has its own socket and playback, without mic or LLM. */
export class SpeechOutputControls {
  constructor({ activeRoom = () => false } = {}) {
    this.activeRoom = activeRoom;
    this.catalog = null;
    this.draft = null;
    this.serial = 0;
    this.job = null;
    this.result = null;
    this.roomSerial = 0;
    this.roomSelection = null;
    this.status = "Loading voices";
    this.elements = Object.fromEntries([
      "output-provider", "output-model", "output-model-summary", "output-voice", "output-transfer", "output-mode-note",
      "output-text", "output-try", "output-stop", "output-status", "output-error", "output-timing", "output-details", "output-observations",
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

  async load(stored = this.draft) {
    this.cancelRoomSelection();
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
    this.cancelRoomSelection();
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

  /** Room selection never shares a Try job, serial or selection revision. */
  roomSnapshot() {
    if (!this.available() || !this.catalog?.revision) {
      throw new Error("Speech output is unavailable. Choose a supported model and voice before entering.");
    }
    return { speechOutput: this.selection(), catalogueRevision: this.catalog.revision };
  }

  async selectForRoom(socket, snapshot) {
    this.cancelRoomSelection();
    snapshot ??= this.roomSnapshot();
    if (!this.available() || socket?.readyState !== WebSocket.OPEN || snapshot.catalogueRevision !== this.catalog?.revision ||
        !this.matchesSelection(snapshot.speechOutput, this.selection())) throw new Error("Room output selection changed. Enter again.");
    return new Promise((resolve, reject) => {
      const pending = { socket, resolve, reject, requestId: `room-${Date.now()}-${++this.roomSerial}`,
        selection: { ...snapshot.speechOutput, provider: { ...snapshot.speechOutput.provider } }, catalogueRevision: snapshot.catalogueRevision };
      this.roomSelection = pending;
      try {
        socket.send(JSON.stringify({ type: "select-output", requestId: pending.requestId,
          catalogueRevision: pending.catalogueRevision, selection: pending.selection }));
      } catch (error) { this.cancelRoomSelection(error.message); }
    });
  }

  acceptRoom(socket, event) {
    const pending = this.roomSelection;
    if (!pending || pending.socket !== socket) return;
    if ((event.type === "session" && event.phase === "stopped") || (event.type === "notice" && event.level === "error")) {
      this.cancelRoomSelection("Room selection stopped.");
    } else if (event.type === "output-selection-error" && (event.requestId === null || event.requestId === pending.requestId)) {
      this.cancelRoomSelection(event.message);
    } else if (event.type === "output-selected" && event.requestId === pending.requestId) {
      if (event.catalogueRevision !== pending.catalogueRevision || this.catalog?.revision !== pending.catalogueRevision ||
          !this.matchesSelection(event.selection, pending.selection) || !this.matchesSelection(this.selection(), pending.selection) ||
          !Number.isSafeInteger(event.selectionRevision) || event.selectionRevision <= 0) {
        this.cancelRoomSelection("Room selection acknowledgement does not match the displayed output.");
        return;
      }
      this.roomSelection = null;
      pending.resolve({ speechOutput: pending.selection, catalogueRevision: pending.catalogueRevision, selectionRevision: event.selectionRevision });
    }
  }

  cancelRoomSelection(message = "Room selection cancelled.") {
    const pending = this.roomSelection;
    this.roomSelection = null;
    pending?.reject(new Error(message));
  }

  matchesSelection(actual, expected) {
    return Boolean(actual && expected) && actual.schema === expected.schema && actual.provider?.kind === expected.provider?.kind &&
      actual?.provider?.runtimeKind === expected?.provider?.runtimeKind && actual?.model === expected?.model &&
      actual?.voice === expected?.voice && actual?.speed === expected?.speed;
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
    this.elements["output-timing"].textContent = `Browser TTFA: ${formatMs(this.browserMs)} · Host first audio: ${formatMs(this.hostMs)}. Browser rendering estimate; speaker latency is unmeasured.${this.applied ? ` ${this.applied.model} · ${this.applied.voice}` : ""}`;
    this.renderResult();
  }

  renderResult() {
    const result = this.result;
    this.elements["output-details"].hidden = !result;
    const lines = [];
    if (result) {
      lines.push(`Request: ${result.requestId} · Playback: ${result.playbackId} · Generation: ${result.generation} · Text: ${result.textId} · ${result.outcome}`);
      lines.push(`Selection: ${JSON.stringify(result.selection)}`, `Requested full text: ${result.text}`);
      for (const [key, value] of Object.entries(result.timing ?? {})) lines.push(`${key}: ${formatObservation(value)}`);
      for (const segment of result.segments) {
        lines.push(`Segment ${segment.index}: ${segment.sentence} · ${segment.sampleRate} Hz · Audio duration: ${formatMs(segment.audioDurationMs)}`);
        for (const [key, value] of Object.entries(segment.timing ?? {})) lines.push(`${key}: ${formatObservation(value)}`);
      }
      for (const request of result.requests) {
        lines.push(`Applied model: ${request.model} · Voice: ${request.voice} · Requested sentence: ${request.text}`);
        lines.push(`Returned transcript: ${typeof request.returnedTranscript === "string" ? JSON.stringify(request.returnedTranscript) : "Unavailable"} · Transcript: ${transcriptStatus(request.text, request.returnedTranscript)} (exact text comparison)`);
        lines.push([["Request wall", "providerRequestWallMs"], ["Provider processing", "providerProcessingMs"], ["Provider network", "providerNetworkMs"]]
          .map(([label, key]) => `${label}: ${formatObservation(request.timing?.[key])}`).join(" · "));
        if (request.sessionSetupWallMs || request.providerFirstAudioMs) lines.push(`Setup wall: ${formatObservation(request.sessionSetupWallMs)} · First provider PCM: ${formatObservation(request.providerFirstAudioMs)}`);
      }
    }
    this.elements["output-observations"].replaceChildren(...lines.map((line) => {
      const paragraph = document.createElement("p");
      paragraph.textContent = line;
      return paragraph;
    }));
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
    const job = { requestId: `voice-${Date.now()}-${++this.serial}`, selection: this.selection(), catalogueRevision: this.catalog.revision, text: this.elements["output-text"].value, submittedAt: performance.now() };
    this.job = job;
    this.result = null;
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
      socket.send(JSON.stringify({ type: "select-output", requestId: job.requestId, catalogueRevision: job.catalogueRevision, selection: job.selection }));
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
      if (event.catalogueRevision !== job.catalogueRevision || !this.matchesSelection(event.selection, job.selection) ||
          !Number.isSafeInteger(event.selectionRevision) || event.selectionRevision <= 0) { this.fail(job, "Voice selection acknowledgement does not match."); return; }
      job.selectionSent = true;
      job.socket.send(JSON.stringify({ type: "speak-sample", requestId: job.requestId, catalogueRevision: event.catalogueRevision, selectionRevision: event.selectionRevision, selection: job.selection, text: job.text }));
    } else if (event.type === "sample-accepted") {
      if (job.playbackId != null) return;
      if (!job.selectionSent || !this.matchesSelection(event.selection, job.selection) || event.text !== job.text ||
          typeof event.textId !== "string" || !event.textId.trim() || !Number.isSafeInteger(event.generation) || event.generation <= 0) {
        this.fail(job, "Voice acceptance does not match this trial."); return;
      }
      job.playbackId = event.playbackId;
      job.generation = event.generation;
      job.textId = event.textId;
      this.applied = job.selection;
    } else if (event.playbackId != null && event.playbackId !== job.playbackId) return;
    else if (event.type === "sample-phase") this.status = event.phase === "opening" ? "Preparing voice. Waiting for audio." : "Waiting for audio";
    else if (event.type === "sample-error") { this.fail(job, event.message); return; }
    else if (event.type === "sample-complete") {
      if (job.playbackId == null || event.playbackId !== job.playbackId || event.textId !== job.textId ||
          !this.matchesSelection(event.selection, job.selection) || !["produced", "drained", "cancelled"].includes(event.outcome)) return;
      if (event.outcome === "cancelled") { this.stop(); return; }
      if (job.complete || (job.produced && event.outcome === "produced")) return;
      this.result = JSON.parse(JSON.stringify({ requestId: job.requestId, playbackId: job.playbackId, generation: job.generation,
        selection: job.selection, textId: job.textId, text: job.text, outcome: event.outcome,
        timing: event.timing, segments: event.segments ?? [], requests: event.requests ?? [] }));
      job.produced = true;
      job.complete = event.outcome === "drained";
      const observation = event.timing?.serverFirstAudioReadyMs;
      this.hostMs = observation?.status === "measured" ? observation.ms : null;
      this.acknowledgeDrain(job);
      if (event.outcome === "produced" && !this.result.segments.length) {
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
    if (message.kind === "first-render" && !job.rendered && Number.isFinite(message.audioTime)) {
      // currentTime need not advance with performance.now(), especially at startup.
      // Use the browser's related output stream positions, never message receipt time.
      let timestamp;
      try { timestamp = job.context.getOutputTimestamp?.(); }
      catch { /* A nonzero render remains observed when output timing is unavailable. */ }
      if (job.context.state === "running" && !job.clockBlocked &&
          Number.isFinite(timestamp?.contextTime) && timestamp.contextTime > 0 &&
          Number.isFinite(timestamp?.performanceTime) && timestamp.performanceTime > 0) {
        const renderedAt = timestamp.performanceTime + (message.audioTime - timestamp.contextTime) * 1_000;
        const elapsed = renderedAt - job.submittedAt;
        if (Number.isFinite(elapsed) && elapsed >= 0) this.browserMs = elapsed;
      }
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
    this.status = this.browserMs != null ? "Finished" : job.rendered
      ? "Finished. Browser audio clock mapping unavailable; browser TTFA unavailable."
      : "Finished. No nonzero audio rendered; browser TTFA unavailable.";
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
