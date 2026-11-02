import { HoldReleaseGate } from "/hold-release.js";
import { MicCapture } from "/mic-capture.js";
import { formatDuration, inputCard, outputCard, outputLatency, readEvents, transcriptMode } from "/road-test-text.js";
import { TalkOverDetector } from "/talk-over.js";
import { VoiceReadback } from "/voice-readback.js";

/**
 * Road test: pick a local speech in model and voice, hold to speak, see the
 * transcript and its timings, hear the voice read it back, interrupt it by
 * key or by talking over it, and replay both sides. Final text arrives as
 * one comparison row; the voice uses the speech output channel.
 */

const SAMPLE_RATE = 16_000;
const MIN_TAKE_MS = 250;
const CUT_IN_MAX_MS = 15_000;
const BUSY_RETRY_MS = 150;
const BUSY_RETRY_LIMIT = 80;
const HISTORY_LIMIT = 20;
const PLACEHOLDER = "Hold Space, speak, release. The transcript appears here and the voice reads it back.";

export class RoadTestPage {
  constructor() {
    this.elements = Object.fromEntries([
      "status", "input-model", "input-card", "output-model", "output-voice", "output-card", "hold", "stop", "level",
      "transcript", "transcript-mode", "input-timings", "output-timings", "interrupt-timings", "replay-take",
      "replay-voice", "echo", "talk-over", "history", "history-section", "level-track",
    ].map((id) => [id, document.getElementById(id)]));
    this.setups = [];
    this.voices = [];
    this.catalogueRevision = null;
    this.context = null;
    this.mic = null;
    this.readback = null;
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
    try {
      const [setups, voices] = await Promise.all([json("/prompt-reader/setups"), json("/speech-output/models")]);
      this.setups = setups.setups;
      this.voices = voices.models.filter((model) => model.provider.kind === "local");
      this.catalogueRevision = voices.revision;
      fill(this.elements["input-model"], this.setups.map((setup) => ({ value: setup.id, label: setup.label,
        disabled: setup.readiness?.ready !== true })), this.setups.find((setup) => setup.readiness?.ready)?.id);
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
    for (const id of ["input-model", "output-voice"]) this.elements[id].addEventListener("change", () => this.render());
    this.elements.echo.addEventListener("change", () => void this.reopenMic());
    this.elements["replay-take"].addEventListener("click", () => this.replay("take"));
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
    if (this.readback?.active) this.stopVoice(at);
    this.phase = "recording";
    this.run = { setup: this.setup(), selection: this.selection(), voiceLabel: this.voiceLabel(), pressedAt: at };
    this.status("Listening. Release when you finish.");
    this.render();
    void this.openAudio().then(() => {
      if (this.phase === "recording" && this.gate.capturing && !this.mic.recording) this.mic.beginTake();
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
  }

  async finishTake() {
    const run = this.run;
    const take = await this.mic?.endTake();
    if (this.run !== run) return;
    if (!take || (take.samples.length - take.preRollSamples) / SAMPLE_RATE * 1_000 < MIN_TAKE_MS) {
      this.phase = "idle";
      this.status("Too short. Hold Space for the whole phrase.");
      this.render();
      return;
    }
    run.take = take.samples;
    this.phase = "transcribing";
    this.status(`Transcribing with ${run.setup.label}`);
    this.render();
    const result = await this.transcribe(run, take.samples, run.releasedAt);
    if (this.run !== run) return;
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
    const result = { text: "", finalMs: null, firstWordsMs: null, loadMs: null, firstUse: false, processingMs: null, error: "" };
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
        if (typeof row.text === "string" && row.text.trim() && result.firstWordsMs == null) {
          result.firstWordsMs = performance.now() - from;
          onText(row.text);
        }
        if (row.status === "complete") {
          result.text = row.text ?? "";
          result.finalMs = performance.now() - from;
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
    run.outputLatency = outputLatency(this.context);
    if (this.run !== run && this.phase !== "cut-in") return;
    if (output.outcome === "error") this.status(`Voice failed: ${output.message}`);
    else if (output.outcome === "finished") this.status("Done. Hold Space for the next take.");
    if (this.phase === "speaking") this.phase = "idle";
  }

  /** Stop key, button, or a new press: timed from the press to silence at the output. */
  stopVoice(at) {
    if (!this.readback?.active || !this.run) return;
    this.run.interrupt = { kind: "stop", from: at };
    this.readback.stop(at);
    this.status("Voice stopped.");
  }

  frame(frame) {
    this.elements.level.style.width = `${Math.min(100, (Number(frame.level) || 0) * 400)}%`;
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
    this.mic.beginTake();
    run.cutInTimer = setTimeout(() => void this.finishCutIn(run), CUT_IN_MAX_MS);
    this.status("Heard you. Listening until you pause.");
    this.render();
  }

  async finishCutIn(run) {
    if (this.phase !== "cut-in" || this.run !== run) return;
    clearTimeout(run.cutInTimer);
    const endedAt = performance.now();
    this.phase = "cut-in-transcribing";
    this.status(`Transcribing what you said over the voice`);
    this.render();
    const take = await this.mic.endTake();
    if (!take) return;
    run.cutInTake = take.samples;
    const result = await this.transcribe(run, take.samples, endedAt, () => {});
    run.interrupt.text = result.error || result.text || "Nothing heard";
    this.phase = "idle";
    this.status("Done. Hold Space for the next take.");
    this.render();
  }

  replay(which) {
    const run = this.run;
    const samples = which === "take" ? run?.cutInTake ?? run?.take : run?.output?.samples;
    const rate = which === "take" ? SAMPLE_RATE : run?.output?.sampleRate;
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
      this.elements.transcript.textContent = run ? "" : PLACEHOLDER;
      this.elements.transcript.classList.toggle("placeholder", !run);
    }
    this.elements["transcript-mode"].textContent = transcriptMode(run?.setup ?? setup);
    this.elements["replay-take"].disabled = !(run?.cutInTake ?? run?.take);
    this.elements["replay-voice"].disabled = !run?.output?.samples?.length;
    this.renderTimings(run);
    this.renderHistory();
  }

  renderTimings(run) {
    const input = run?.input;
    describe(this.elements["input-timings"], [
      ...(run?.setup?.card?.transcript === "partials" ? [["Release to first words", formatDuration(input?.firstWordsMs)]] : []),
      ["Release to text", formatDuration(input?.finalMs)],
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
    ]);
    const interrupt = run?.interrupt;
    const stopMs = output?.stopMs;
    const said = interrupt?.text ?? (this.phase === "cut-in" ? "Listening" : this.phase === "cut-in-transcribing" ? "Transcribing" : null);
    describe(this.elements["interrupt-timings"], interrupt?.kind === "talk-over" ? [
      ["Voice heard after", formatDuration(interrupt.detectMs)],
      ["Silent after", formatDuration(stopMs)],
      ["You said", said, "wide"],
    ] : [
      ["Stop to silence", formatDuration(interrupt?.kind === "stop" ? stopMs : null)],
      ["Voice heard after", null],
      ["Silent after", null],
    ]);
  }

  /** Earlier takes only; the latest is already shown above. */
  renderHistory() {
    const earlier = this.history.filter((run) => run !== this.run);
    const rows = earlier.map((run) => {
      const row = document.createElement("tr");
      for (const [text, kind] of [[run.setup.label], [run.output ? run.voiceLabel : "–"], [run.input.error || run.input.text || "Nothing heard", "said-cell"],
        [formatDuration(run.input.finalMs) ?? "–", "num"], [formatDuration(run.output?.firstAudioMs) ?? "–", "num"],
        [formatDuration(releaseToVoice(run)) ?? "–", "num"]]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        if (kind) cell.className = kind;
        row.append(cell);
      }
      return row;
    });
    this.elements.history.replaceChildren(...rows);
    this.elements["history-section"].hidden = earlier.length === 0;
  }
}

/** From the owner's release to the voice's first sound at the output. */
function releaseToVoice(run) {
  const firstAudioMs = run?.output?.firstAudioMs;
  if (firstAudioMs == null || run.voiceRequestedAt == null || run.releasedAt == null) return null;
  return run.voiceRequestedAt + firstAudioMs - run.releasedAt;
}

/** Fills a timing list; missing values show a quiet dash. */
function describe(list, entries) {
  const rows = entries.map(([label, value, kind]) => {
    const row = document.createElement("div");
    if (kind) row.className = kind;
    const term = document.createElement("dt");
    term.textContent = label;
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
