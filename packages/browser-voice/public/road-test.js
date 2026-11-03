import { HoldReleaseGate } from "/hold-release.js";
import { LiveWords } from "/live-words.js";
import { MicCapture } from "/mic-capture.js";
import { trailItems, trailSummary } from "/live-trail.js";
import { formatDuration, inputCard, liveLine, outputCard, outputLatency, readEvents } from "/road-test-text.js";
import { TalkOverDetector } from "/talk-over.js";
import { VoiceReadback } from "/voice-readback.js";

/**
 * Road test: pick a local speech in model and voice, hold to speak, see the
 * transcript and its timings, hear the voice read it back, interrupt it by
 * key or by talking over it, and replay both sides. A model that truly
 * streams, the setup's own or another the owner picks, shows words while the
 * owner speaks, one tone quieter, until the final text replaces them in
 * place; a collapsed trail lists each revision. Final text arrives as one
 * comparison row; the voice uses the speech output channel.
 */

const SAMPLE_RATE = 16_000;
const MIN_TAKE_MS = 250;
const CUT_IN_MAX_MS = 15_000;
const BUSY_RETRY_MS = 150;
const BUSY_RETRY_LIMIT = 80;
const HISTORY_LIMIT = 20;
const PLACEHOLDER = "The transcript appears here, and the voice reads it back.";
const LISTENING = "Listening. Release when you finish.";
const DONE = "Done";
/** Earlier takes columns; the transcript spans a line of its own, and narrow screens label each cell with these. */
const HISTORY_COLUMNS = [
  { label: "Speech in", cell: (run) => [run.setup.label, liveDiffers(run) ? `Live words ${run.liveChoice.setup.label}` : null] },
  { label: "Voice", cell: (run) => [run.output ? run.voiceLabel : null] },
  { label: "Transcript", kind: "said-cell", cell: (run) => [run.input.error || run.input.text || "Nothing heard"], trail: true },
  { label: "Press to first words", kind: "num", cell: (run) => [formatDuration(pressToFirstWords(run))] },
  { label: "Release to text", kind: "num", cell: (run) => [formatDuration(run.input.finalMs),
    run.input.loadMs != null ? `Model start ${formatDuration(run.input.loadMs)}` : null] },
  { label: "Text to first audio", kind: "num", cell: (run) => [formatDuration(run.output?.firstAudioMs),
    run.output?.loadMs != null ? `Model start ${formatDuration(run.output.loadMs)}` : null] },
  { label: "Release to voice", kind: "num", cell: (run) => [formatDuration(releaseToVoice(run))] },
];

export class RoadTestPage {
  constructor() {
    this.elements = Object.fromEntries([
      "status", "input-model", "input-card", "output-model", "output-voice", "output-card", "hold", "stop", "level",
      "transcript", "take-models", "input-timings", "output-timings", "interrupt-timings", "replay-take",
      "replay-talk-over", "replay-voice", "echo", "talk-over", "history", "history-head", "history-section", "level-track",
      "live-model", "live-settings", "live-line", "trail", "trail-summary", "trail-list",
    ].map((id) => [id, document.getElementById(id)]));
    this.setups = [];
    this.voices = [];
    this.liveSelects = new Map();
    this.historyRows = new WeakMap();
    this.catalogueRevision = null;
    this.context = null;
    this.mic = null;
    this.readback = null;
    this.live = new LiveWords({ onEvent: (event) => this.liveEvent(event) });
    this.detector = new TalkOverDetector();
    this.phase = "idle";
    this.run = null;
    this.history = [];
    this.replaying = null;
    this.serial = 0;
    this.gate = new HoldReleaseGate({ onCommit: () => void this.finishTake(), onChange: () => this.render() });
  }

  async start() {
    this.bind();
    this.elements["history-head"].replaceChildren(...HISTORY_COLUMNS.map(({ label, kind }) => {
      const head = document.createElement("th");
      head.scope = "col";
      head.textContent = label;
      if (kind) head.className = kind;
      return head;
    }));
    try {
      const [setups, voices] = await Promise.all([json("/prompt-reader/setups"), json("/speech-output/models")]);
      this.setups = setups.setups;
      this.voices = voices.models.filter((model) => model.provider.kind === "local");
      this.catalogueRevision = voices.revision;
      fill(this.elements["input-model"], this.setups.map((setup) => ({ value: setup.id, label: setup.label,
        disabled: setup.readiness?.ready !== true })), this.setups.find((setup) => setup.readiness?.ready)?.id);
      this.resetLive();
      fill(this.elements["output-model"], this.voices.map((model) => ({ value: model.id, label: model.label,
        disabled: model.readiness?.state !== "ready" })), voices.default?.model);
      this.fillVoices();
      const ready = this.setups.some((setup) => setup.readiness?.ready);
      this.status(ready ? "Ready" : `No speech in model is ready: ${this.setups[0]?.readiness?.reason ?? "none configured"}`);
    } catch (error) {
      this.status(`Models unavailable: ${error.message}`);
    }
    this.render();
  }

  bind() {
    const { hold, stop } = this.elements;
    hold.addEventListener("pointerdown", (event) => { hold.setPointerCapture?.(event.pointerId); this.press(event.timeStamp); });
    hold.addEventListener("pointerup", (event) => this.release(event.timeStamp));
    hold.addEventListener("pointercancel", (event) => this.release(event.timeStamp));
    stop.addEventListener("click", (event) => this.stopVoice(event.timeStamp));
    this.elements["output-model"].addEventListener("change", () => { this.fillVoices(); this.render(); });
    this.elements["input-model"].addEventListener("change", () => { this.resetLive(); this.render(); });
    this.elements["live-model"].addEventListener("change", () => { this.fillLiveSettings(); this.render(); });
    for (const id of ["output-voice", "talk-over"]) this.elements[id].addEventListener("change", () => this.render());
    this.elements.echo.addEventListener("change", () => void this.reopenMic());
    for (const which of ["take", "talk-over"]) this.elements[`replay-${which}`].addEventListener("click", () => this.replay(which));
    this.elements["replay-voice"].addEventListener("click", () => this.replay("voice"));
    document.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (!event.repeat) this.press(event.timeStamp);
      } else if (event.code === "Escape") this.stopVoice(event.timeStamp);
    });
    document.addEventListener("keyup", (event) => {
      if (event.code === "Space") { event.preventDefault(); this.release(event.timeStamp); }
    });
  }

  fillVoices() {
    const model = this.outputModel();
    fill(this.elements["output-voice"], (model?.voices ?? []).map((voice) => ({ value: String(voice.id), label: voice.label })),
      String(model?.defaultVoice ?? ""));
  }

  setup() {
    return this.setups.find((setup) => setup.id === this.elements["input-model"].value);
  }

  /** Live words default to today's pairing for the chosen setup: its own stream, its base's, or off. */
  resetLive() {
    fill(this.elements["live-model"], [{ value: "off", label: "Off" }, ...this.setups.filter((setup) => setup.live)
      .map((setup) => ({ value: setup.id, label: setup.label, disabled: setup.readiness?.ready !== true }))],
    this.setup()?.liveDefault ?? "off");
    this.fillLiveSettings();
  }

  /** One column per setting of the chosen live model, each at today's value with its one line note. */
  fillLiveSettings() {
    this.liveSelects.clear();
    const columns = (this.liveModel()?.live.settings ?? []).map((setting) => {
      const column = document.createElement("label");
      column.className = "setting";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = setting.label;
      const select = document.createElement("select");
      fill(select, setting.options.map((option) => ({ value: String(option.value), label: option.label })), String(setting.default));
      select.addEventListener("change", () => this.render());
      const note = document.createElement("span");
      note.className = "quiet";
      note.textContent = setting.note;
      column.append(label, select, note);
      this.liveSelects.set(setting.id, select);
      return column;
    });
    this.elements["live-settings"].replaceChildren(...columns);
  }

  liveModel() {
    return this.setups.find((setup) => setup.live && setup.id === this.elements["live-model"].value);
  }

  /** The live model and settings a press uses; null when live words are off. */
  liveChoice() {
    const setup = this.liveModel();
    if (!setup) return null;
    return { setup, stream: setup.live, settings: Object.fromEntries(setup.live.settings.map((setting) =>
      [setting.id, Number(this.liveSelects.get(setting.id)?.value ?? setting.default)])) };
  }

  outputModel() {
    return this.voices.find((model) => model.id === this.elements["output-model"].value);
  }

  selection() {
    const model = this.outputModel();
    const voice = model?.voices.find((entry) => String(entry.id) === this.elements["output-voice"].value);
    if (!model || !voice) return null;
    return { schema: 1, provider: { kind: "local" }, model: model.id, voice: voice.id, speed: model.defaultSpeed };
  }

  /** The audio context, mic and playback node open at the first press, a user gesture. */
  async openAudio() {
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    await this.context.resume();
    this.mic ??= new MicCapture({ context: this.context, onFrame: (frame) => this.frame(frame) });
    this.readback ??= new VoiceReadback({ context: this.context, onChange: () => this.render() });
    await Promise.all([this.mic.start(this.elements.echo.checked), this.readback.prepare()]);
    this.elements["level-track"].hidden = false;
  }

  async reopenMic() {
    if (!this.mic?.open || this.mic.recording) return;
    try { await this.mic.start(this.elements.echo.checked); }
    catch (error) { this.status(micError(error)); }
    this.render();
  }

  press(at) {
    if (!this.setup()?.readiness?.ready || this.phase === "cut-in" || !this.gate.press()) return;
    this.stopReplay();
    // A press over the voice stops it like Esc; the new take shows that stop.
    const interrupt = this.stopVoice(at);
    this.phase = "recording";
    this.run = { setup: this.setup(), liveChoice: this.liveChoice(), selection: this.selection(), voiceLabel: this.voiceLabel(),
      pressedAt: at, ...(interrupt ? { interrupt } : {}) };
    this.status(LISTENING);
    this.render();
    void this.openAudio().then(() => {
      if (this.phase === "recording" && this.gate.capturing && !this.mic.recording) this.beginTake(this.run);
    }, (error) => {
      this.gate.cancel();
      this.phase = "idle";
      this.status(micError(error));
      this.render();
    });
  }

  release(at) {
    if (this.phase !== "recording" || !this.gate.release()) return;
    this.run.releasedAt = at;
    // The owner has finished: a feed still running settles during the release tail, before the final pass.
    this.live.end();
  }

  /** Starts recording; with live words on, the chosen live model hears the take, led by its pre roll. */
  beginTake(target) {
    const preRoll = this.mic.beginTake();
    const choice = this.run.liveChoice;
    if (choice) target.live = { takeId: this.live.begin(choice.setup.id, preRoll, choice.settings), updates: [] };
  }

  /** Live words for the latest take or what was said over the voice; ignored once that text is final. */
  liveEvent(event) {
    const run = this.run;
    const target = [run?.live, run?.interrupt?.live].find((live) => live && live.takeId === event.takeId);
    if (!target) return;
    const now = performance.now();
    const listening = this.phase === "recording" && target === run.live;
    if (event.type === "stream") {
      // The stream and settings the host applied: what the live words line names.
      target.stream = { setupId: event.setupId, settings: event.settings };
    } else if (event.type === "loading") {
      target.loadingAt = now;
      if (listening) this.status(`Listening. Loading ${run.liveChoice.setup.label} for live words.`);
    } else if (event.type === "ready") {
      if (event.cold && target.loadingAt != null) target.loadMs = now - target.loadingAt;
      if (listening) this.status(LISTENING);
    } else if (event.type === "error") {
      target.error = event.message;
      if (listening) this.status(`Listening. Live words stopped: ${event.message}`);
    } else if (event.type === "partial" && !(target === run.live ? run.input : run.interrupt.text)) {
      const words = `${event.stable}${event.tentative}`.trim();
      if (words) {
        target.words = words;
        target.firstWordsAt ??= now;
        if (target === run.live) target.updates.push({ atMs: now - run.pressedAt, words });
      }
    }
    this.render();
  }

  async finishTake() {
    const run = this.run;
    this.live.end();
    const take = await this.mic?.endTake();
    if (this.run !== run) return;
    if (!take || (take.samples.length - take.preRollSamples) / SAMPLE_RATE * 1_000 < MIN_TAKE_MS) {
      this.phase = "idle";
      this.status("Too short. Hold for the whole phrase.");
      this.render();
      return;
    }
    run.take = take.samples;
    this.phase = "transcribing";
    this.status(`Transcribing with ${run.setup.label}`);
    this.render();
    const result = await this.transcribe(run, take.samples, run.releasedAt);
    if (this.run !== run) return;
    // Model start counts a load the live words did while the owner spoke.
    if (run.live?.loadMs != null) result.loadMs = (result.loadMs ?? 0) + run.live.loadMs;
    run.input = result;
    if (result.error) {
      this.phase = "idle";
      this.status(result.error);
    } else if (!result.text.trim()) {
      this.phase = "idle";
      this.status("Nothing heard. Try again closer to the microphone.");
    } else await this.speak(run);
    this.remember(run);
    this.render();
  }

  /** One comparison row for one setup. Waits while the voice releases the host. */
  async transcribe(run, samples, from, onText = (text) => this.showTranscript(text)) {
    const result = { text: "", finalMs: null, finalAt: null, loadMs: null, firstUse: false, processingMs: null, error: "" };
    const setupIds = JSON.stringify([run.setup.id]);
    for (let tries = 0; ; tries++) {
      const requestId = `road-${Date.now()}-${++this.serial}`;
      let response;
      try {
        response = await fetch(`/prompt-reader/comparisons?${new URLSearchParams({ requestId, setupIds })}`, {
          method: "POST", headers: { "content-type": "application/octet-stream" }, body: samples.buffer.slice(0) });
      } catch (error) { return { ...result, error: `Transcription failed: ${error.message}` }; }
      if ((response.status === 409 || response.status === 429) && tries < BUSY_RETRY_LIMIT) {
        await new Promise((resolve) => setTimeout(resolve, BUSY_RETRY_MS));
        continue;
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        return { ...result, error: `Transcription failed: ${body.error ?? response.status}` };
      }
      let loadingAt = null;
      let startMs = 0;
      await readEvents(response, (event) => {
        if (event.type !== "row" || event.row.setupId !== run.setup.id) return;
        const row = event.row;
        // A cleanup row loads Ultra, then its own model: every load phase counts.
        if (row.status === "loading") {
          loadingAt = performance.now();
          this.status(`Loading ${run.setup.label}`);
        } else if (row.status === "processing") {
          if (loadingAt != null) startMs += performance.now() - loadingAt;
          loadingAt = null;
          this.status(`Transcribing with ${run.setup.label}`);
        }
        if (row.status === "complete") {
          result.text = row.text ?? "";
          result.finalAt = performance.now();
          result.finalMs = result.finalAt - from;
          // Observed here: the host's load figure leaves out worker process startup.
          result.loadMs = row.timings.firstUse ? startMs : null;
          result.processingMs = row.timings.processingMs;
          onText(result.text);
        } else if (["error", "cancelled", "skipped"].includes(row.status)) {
          result.error = `${run.setup.label} failed: ${row.error?.message ?? row.status}`;
        }
      });
      return result;
    }
  }

  async speak(run) {
    if (!run.selection) { this.phase = "idle"; this.status("Choose a voice to hear the transcript."); return; }
    this.phase = "speaking";
    this.detector.reset();
    this.status(`Reading back with ${run.voiceLabel}`);
    const speaking = this.readback.speak({ selection: run.selection, catalogueRevision: this.catalogueRevision, text: run.input.text });
    run.voiceRequestedAt = performance.now();
    this.render();
    const output = await speaking;
    run.output = output;
    // Only a stopped reading times its interrupt; a finished one leaves a stop carried from the last take alone.
    if (output.outcome === "stopped" && run.interrupt) run.interrupt.stopMs = output.stopMs;
    run.outputLatency = outputLatency(this.context);
    if (this.run !== run && this.phase !== "cut-in") return;
    if (output.outcome === "error") this.status(`Voice failed: ${output.message}`);
    else if (output.outcome === "finished") this.status(DONE);
    if (this.phase === "speaking") this.phase = "idle";
  }

  /** Stop key, button, or a new press: timed from the press to silence at the output. */
  stopVoice(at) {
    if (!this.readback?.active || !this.run) return undefined;
    const interrupt = { kind: "stop", from: at };
    this.run.interrupt = interrupt;
    this.readback.stop(at);
    this.status("Voice stopped.");
    return interrupt;
  }

  frame(frame) {
    this.elements.level.style.width = `${Math.min(100, (Number(frame.level) || 0) * 400)}%`;
    if (this.mic?.recording) this.live.push(frame.samples);
    const run = this.run;
    if (this.phase === "speaking" && this.readback.playing && this.elements["talk-over"].checked) {
      const event = this.detector.push(frame.level, frame.audioTime);
      if (event?.type === "onset") this.cutIn(run, event.onsetTime);
    } else if (this.phase === "cut-in") {
      const event = this.detector.push(frame.level, frame.audioTime);
      if (event?.type === "end") void this.finishCutIn(run);
    }
  }

  /** The owner talked over the voice: stop it and record what they say. */
  cutIn(run, onsetTime) {
    const detectedAt = performance.now();
    run.interrupt = { kind: "talk-over", detectMs: Math.max(0, (this.context.currentTime - onsetTime) * 1_000) };
    this.phase = "cut-in";
    this.readback.stop(detectedAt);
    this.beginTake(run.interrupt);
    run.cutInTimer = setTimeout(() => void this.finishCutIn(run), CUT_IN_MAX_MS);
    this.status("Heard you. Listening until you pause.");
    this.render();
  }

  async finishCutIn(run) {
    if (this.phase !== "cut-in" || this.run !== run) return;
    clearTimeout(run.cutInTimer);
    const endedAt = performance.now();
    this.live.end();
    this.phase = "cut-in-transcribing";
    this.status(`Transcribing what you said over the voice`);
    this.render();
    const take = await this.mic.endTake();
    if (!take) return;
    run.cutInTake = take.samples;
    const result = await this.transcribe(run, take.samples, endedAt, () => {});
    run.interrupt.text = result.error || result.text || "Nothing heard";
    this.phase = "idle";
    this.status(DONE);
    this.render();
  }

  /** Replays the latest take, what was said over the voice, or the voice. */
  replay(which) {
    const run = this.run;
    const [samples, rate] = { take: [run?.take, SAMPLE_RATE], "talk-over": [run?.cutInTake, SAMPLE_RATE],
      voice: [run?.output?.samples, run?.output?.sampleRate] }[which];
    if (!samples?.length || !rate || !this.context) return;
    this.stopReplay();
    const buffer = this.context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => { if (this.replaying === source) this.replaying = null; };
    this.replaying = source;
    source.start();
  }

  stopReplay() {
    const source = this.replaying;
    this.replaying = null;
    if (source) { source.onended = null; source.stop(); }
  }

  remember(run) {
    if (!run.input || run.remembered) return;
    run.remembered = true;
    this.history.unshift(run);
    this.history.length = Math.min(this.history.length, HISTORY_LIMIT);
  }

  voiceLabel() {
    const model = this.outputModel();
    const voice = model?.voices.find((entry) => String(entry.id) === this.elements["output-voice"].value);
    return model ? `${model.label}${model.voices.length > 1 && voice ? `, ${voice.label}` : ""}` : "";
  }

  showTranscript(text) {
    const element = this.elements.transcript;
    element.textContent = text;
    element.classList.toggle("placeholder", false);
    element.classList.toggle("partial", false);
  }

  status(message) {
    this.elements.status.textContent = message;
  }

  render() {
    const { hold, stop } = this.elements;
    const setup = this.setup();
    const model = this.outputModel();
    this.elements["input-card"].textContent = setup ? inputCard(setup) : "";
    this.elements["output-card"].textContent = model ? outputCard(model) : "";
    hold.setAttribute("aria-pressed", String(this.gate.capturing));
    hold.disabled = !setup?.readiness?.ready || this.phase === "cut-in";
    stop.disabled = !this.readback?.active;
    const run = this.run;
    if (!run?.input) {
      // Live words, one tone quieter, until the final text replaces them in place.
      this.elements.transcript.textContent = run ? run.live?.words ?? "" : PLACEHOLDER;
      this.elements.transcript.classList.toggle("placeholder", !run);
      this.elements.transcript.classList.toggle("partial", Boolean(run));
    }
    // Names the models that produced the latest take, so changed selects never claim its figures.
    this.elements["take-models"].textContent = run ? [run.setup.label, run.selection && run.voiceLabel].filter(Boolean).join(" · ") : "";
    this.renderLive(run);
    this.elements["replay-take"].disabled = !run?.take;
    this.elements["replay-talk-over"].hidden = !run?.cutInTake;
    this.elements["replay-voice"].disabled = !run?.output?.samples?.length;
    this.renderTimings(run);
    this.renderHistory();
  }

  /** The line naming what made the latest take's live words (or would make the next), and its trail. */
  renderLive(run) {
    const choice = run ? run.liveChoice : this.liveChoice();
    const applied = run?.live?.stream;
    this.elements["live-line"].textContent = liveLine(choice && { ...choice, settings: applied?.settings ?? choice.settings },
      run?.setup ?? this.setup());
    const updates = run?.live?.updates ?? [];
    this.elements.trail.hidden = updates.length === 0;
    this.elements["trail-summary"].textContent = updates.length ? trailSummary(updates) : "";
    this.elements["trail-list"].replaceChildren(...trailItems(document, updates, run?.input?.text ?? null, formatDuration));
  }

  renderTimings(run) {
    const input = run?.input;
    // Where the live and final models differ, each figure names its model.
    const differs = run ? liveDiffers(run) : false;
    describe(this.elements["input-timings"], [
      ...((run ? run.liveChoice : this.liveChoice()) ? [["Press to first words", formatDuration(pressToFirstWords(run)), null,
        differs ? run.liveChoice.setup.label : null]] : []),
      ["Release to text", formatDuration(input?.finalMs), null, differs ? run.setup.label : null],
      ["Model time", formatDuration(input?.processingMs)],
      ...(input?.loadMs != null ? [["Model start", formatDuration(input.loadMs)]] : []),
    ]);
    const output = run?.output;
    const latency = run?.outputLatency ?? outputLatency(this.context);
    describe(this.elements["output-timings"], [
      ["Release to voice", formatDuration(releaseToVoice(run))],
      ["Text to first audio", formatDuration(output?.firstAudioMs)],
      ["Text to end", formatDuration(output?.totalMs) ?? (output?.outcome === "stopped" ? "stopped" : null)],
      ["Output latency", formatDuration(latency?.outputMs)],
      ["Base latency", formatDuration(latency?.baseMs)],
      ...(output?.loadMs != null ? [["Model start", formatDuration(output.loadMs)]] : []),
    ]);
    const interrupt = run?.interrupt;
    const stopMs = interrupt?.stopMs;
    const said = interrupt?.text ?? interrupt?.live?.words
      ?? (this.phase === "cut-in" ? "Listening" : this.phase === "cut-in-transcribing" ? "Transcribing" : null);
    // Talk over is off by default; while off, only the stop key or button can interrupt.
    describe(this.elements["interrupt-timings"], interrupt?.kind === "talk-over" ? [
      ["Voice heard after", formatDuration(interrupt.detectMs)],
      ["Silent after", formatDuration(stopMs)],
      ["You said", said, interrupt.text == null && interrupt.live?.words ? "wide partial" : "wide"],
    ] : [
      ["Stop to silence", formatDuration(interrupt?.kind === "stop" ? stopMs : null)],
      ...(this.elements["talk-over"].checked ? [["Voice heard after", null], ["Silent after", null]] : []),
    ]);
  }

  /**
   * Earlier takes only; the latest is already shown above. A cold row names
   * its model start. Rows are built once, so an opened trail stays open.
   */
  renderHistory() {
    const earlier = this.history.filter((run) => run !== this.run);
    const rows = earlier.map((run) => {
      if (this.historyRows.has(run)) return this.historyRows.get(run);
      const row = document.createElement("tr");
      for (const { label, kind, cell, trail } of HISTORY_COLUMNS) {
        const [value, note] = cell(run);
        const data = document.createElement("td");
        data.dataset.label = label;
        if (kind) data.className = kind;
        const text = document.createElement("span");
        text.textContent = value ?? "–";
        data.append(text);
        if (note) {
          const quiet = document.createElement("span");
          quiet.className = "note";
          quiet.textContent = note;
          data.append(quiet);
        }
        if (trail && run.live?.updates?.length) data.append(historyTrail(run));
        row.append(data);
      }
      this.historyRows.set(run, row);
      return row;
    });
    this.elements.history.replaceChildren(...rows);
    this.elements["history-section"].hidden = earlier.length === 0;
  }
}

/**
 * From the press to the first words on screen: live words where they came,
 * else the final text. Only a take with live words on shows it, and a take
 * where no words came shows none.
 */
function pressToFirstWords(run) {
  if (!run?.liveChoice) return null;
  const firstWordsAt = run.live?.firstWordsAt ?? (run.input?.text?.trim() ? run.input.finalAt : null);
  return firstWordsAt != null && run.pressedAt != null ? firstWordsAt - run.pressedAt : null;
}

/** Whether another model than the final one made the take's live words. */
function liveDiffers(run) {
  return Boolean(run.liveChoice && run.liveChoice.setup.id !== run.setup.id);
}

/** An earlier take's trail, collapsed to its one line summary. */
function historyTrail(run) {
  const details = document.createElement("details");
  details.className = "trail";
  const summary = document.createElement("summary");
  summary.className = "note";
  summary.textContent = trailSummary(run.live.updates);
  const list = document.createElement("ol");
  list.replaceChildren(...trailItems(document, run.live.updates, run.input?.text ?? null, formatDuration));
  details.append(summary, list);
  return details;
}

/** From the owner's release to the voice's first sound at the output. */
function releaseToVoice(run) {
  const firstAudioMs = run?.output?.firstAudioMs;
  if (firstAudioMs == null || run.voiceRequestedAt == null || run.releasedAt == null) return null;
  return run.voiceRequestedAt + firstAudioMs - run.releasedAt;
}

/** Fills a timing list; missing values show a quiet dash, and a note names the figure's model. */
function describe(list, entries) {
  const rows = entries.map(([label, value, kind, note]) => {
    const row = document.createElement("div");
    if (kind) row.className = kind;
    const term = document.createElement("dt");
    term.textContent = label;
    if (note) {
      const quiet = document.createElement("span");
      quiet.className = "note";
      quiet.textContent = note;
      term.append(quiet);
    }
    const detail = document.createElement("dd");
    detail.textContent = value ?? "–";
    if (value == null) detail.className = "missing";
    row.append(term, detail);
    return row;
  });
  list.replaceChildren(...rows);
}

function fill(select, entries, value) {
  select.replaceChildren(...entries.map((entry) => {
    const option = new Option(entry.label, entry.value);
    option.disabled = Boolean(entry.disabled);
    return option;
  }));
  if (entries.some((entry) => entry.value === value)) select.value = value;
}

async function json(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `${url} answered ${response.status}`);
  return body;
}

function micError(error) {
  return error?.name === "NotAllowedError" ? "Microphone blocked. Allow it in the browser and press again."
    : `Microphone unavailable: ${error?.message ?? error}`;
}

if (typeof window !== "undefined" && document.getElementById("hold")) void new RoadTestPage().start();
