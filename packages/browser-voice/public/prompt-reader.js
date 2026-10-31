import { CAPTURE_AUDIO_CONSTRAINTS } from "/capture-constraints.js";
import { HoldReleaseGate } from "/hold-release.js";

/**
 * Prompt reader: walks a prompt set in id order and records one take per
 * prompt through the capture path TM dictation uses: the capture worklet,
 * CAPTURE_AUDIO_CONSTRAINTS, the mic opened at each press and stopped at
 * each commit, and the HoldReleaseGate tail after release. A saved take
 * becomes a corpus pair on the host; the prompt text is its expected
 * transcript. Play replays the unsaved take, or a saved take of the
 * current prompt (latest first; Earlier take steps back), without touching
 * the mic or any take. A ready take is also transcribed for display only:
 * the text shows under the prompt with words that differ marked, and is
 * never saved; saving never waits for it.
 */

const SAMPLE_RATE = 16_000;
const MIN_TAKE_MS = 300;
const CLIP_LEVEL = 0.99;
/** A transcript later than this shows as unavailable. */
const TRANSCRIBE_TIMEOUT_MS = 15_000;

class PromptReaderPage {
  constructor() {
    this.elements = {
      body: document.body,
      progress: document.getElementById("progress"),
      meta: document.getElementById("prompt-meta"),
      text: document.getElementById("prompt-text"),
      heard: document.getElementById("heard"),
      heardText: document.getElementById("heard-text"),
      heardNote: document.getElementById("heard-note"),
      level: document.getElementById("level"),
      status: document.getElementById("status"),
      output: document.getElementById("output"),
      record: document.getElementById("record"),
      save: document.getElementById("save"),
      play: document.getElementById("play"),
      playLabel: document.getElementById("play-label"),
      earlier: document.getElementById("earlier"),
      previous: document.getElementById("previous"),
      next: document.getElementById("next"),
    };
    this.prompts = [];
    this.index = 0;
    this.context = null;
    this.capture = null;
    this.stream = null;
    this.micGeneration = 0;
    this.source = null;
    this.frames = [];
    /** Flush requests the worklet has not answered yet. */
    this.flushes = 0;
    /** True while the ended take waits for its flush reply. */
    this.draining = false;
    this.take = null;
    this.saving = false;
    this.failed = false;
    this.playback = null;
    /** Saved take offset from the latest; 0 plays the newest. */
    this.takeOffset = 0;
    this.gate = new HoldReleaseGate({
      onCommit: () => this.finishTake(),
      onChange: () => this.render(),
    });
  }

  async start() {
    this.bind();
    const response = await fetch("/prompt-reader/prompts");
    const body = await response.json();
    if (!response.ok) {
      this.fail(body.error ?? `prompts unavailable (${response.status})`);
      return;
    }
    this.prompts = body.prompts;
    this.elements.output.textContent = body.outputDir;
    const firstOpen = this.prompts.findIndex((prompt) => !prompt.recorded);
    this.index = firstOpen === -1 ? this.prompts.length : firstOpen;
    this.setStatus("Hold Space while you read the prompt aloud, then release.");
    this.render();
  }

  bind() {
    window.addEventListener("keydown", (event) => {
      if (isSpace(event)) {
        event.preventDefault();
        if (!event.repeat) void this.press();
      } else if (event.key === "Enter" && !event.repeat) {
        event.preventDefault();
        void this.save();
      } else if ((event.key === "p" || event.key === "P") && !event.repeat) {
        void this.play();
      } else if ((event.key === "t" || event.key === "T") && !event.repeat) {
        this.stepTake();
      } else if (event.key === "ArrowRight") {
        this.move(1);
      } else if (event.key === "ArrowLeft") {
        this.move(-1);
      }
    });
    window.addEventListener("keyup", (event) => {
      if (isSpace(event)) {
        event.preventDefault();
        this.gate.release();
      }
    });
    // A hidden tab cancels the take, as TM dictation cancels on hidden.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.cancelTake("Take cancelled: the tab was hidden.");
    });
    const record = this.elements.record;
    record.addEventListener("pointerdown", () => void this.press());
    for (const type of ["pointerup", "pointerleave", "pointercancel"]) {
      record.addEventListener(type, () => this.gate.release());
    }
    this.elements.save.addEventListener("click", () => void this.save());
    this.elements.play.addEventListener("click", () => void this.play());
    this.elements.earlier.addEventListener("click", () => this.stepTake());
    this.elements.previous.addEventListener("click", () => this.move(-1));
    this.elements.next.addEventListener("click", () => this.move(1));
  }

  current() {
    return this.prompts[this.index];
  }

  /** The saved take Play would use, or null when the prompt has none. */
  savedTake() {
    const takes = this.current()?.takes ?? [];
    if (!takes.length) return null;
    const offset = this.takeOffset % takes.length;
    return { name: takes[takes.length - 1 - offset], number: takes.length - offset, count: takes.length };
  }

  stepTake() {
    if (this.take || this.gate.capturing || (this.current()?.takes?.length ?? 0) < 2) return;
    this.stopPlayback();
    this.takeOffset += 1;
    const saved = this.savedTake();
    this.setStatus(`Take ${saved.number} of ${saved.count} selected. P plays it.`);
    this.render();
  }

  /** Play the unsaved take, else the selected saved take; never opens the mic. */
  async play() {
    if (this.gate.capturing || this.saving) return;
    const saved = this.take ? null : this.savedTake();
    if (!this.take && !saved) return;
    this.stopPlayback();
    const token = {};
    this.playback = token;
    try {
      this.context ??= new AudioContext({ latencyHint: "interactive" });
      await this.context.resume();
      let buffer;
      if (this.take) {
        buffer = this.context.createBuffer(1, this.take.samples.length, SAMPLE_RATE);
        buffer.copyToChannel(this.take.samples.slice(), 0);
      } else {
        const response = await fetch(`/prompt-reader/takes/${encodeURIComponent(saved.name)}`);
        if (!response.ok) throw new Error(`take unavailable (${response.status})`);
        buffer = await this.context.decodeAudioData(await response.arrayBuffer());
      }
      if (this.playback !== token) return;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.context.destination);
      source.onended = () => {
        if (this.playback?.source === source) this.playback = null;
        this.render();
      };
      token.source = source;
      source.start();
      this.render();
    } catch (error) {
      if (this.playback === token) this.playback = null;
      this.fail(`Playback failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  stopPlayback() {
    const playing = this.playback;
    this.playback = null;
    if (playing?.source) {
      playing.source.onended = null;
      playing.source.stop();
    }
  }

  async press() {
    if (this.saving || this.draining || !this.current() || !this.gate.press()) return;
    this.stopPlayback();
    const generation = ++this.micGeneration;
    this.frames = [];
    this.take = null;
    this.setStatus("Recording. Release when you finish the prompt.");
    try {
      await this.openMic(generation);
    } catch (error) {
      if (generation !== this.micGeneration) return;
      this.gate.cancel();
      this.closeMic();
      this.fail(describeMicError(error));
    }
  }

  async openMic(generation) {
    // AudioContext and worklet are created once, at the first press, and
    // resumed at every press; the mic itself opens per press.
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    if (!this.capture) {
      await this.context.audioWorklet.addModule("/capture-worklet.js");
      if (generation !== this.micGeneration) return;
      this.capture = new AudioWorkletNode(this.context, "speak-easy-capture");
      this.capture.port.onmessage = (event) => this.onFrame(event.data);
      const silent = this.context.createGain();
      silent.gain.value = 0;
      this.capture.connect(silent).connect(this.context.destination);
    }
    await this.context.resume();
    if (generation !== this.micGeneration) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { ...CAPTURE_AUDIO_CONSTRAINTS },
    });
    if (generation !== this.micGeneration || !this.gate.capturing) {
      // This press ended before its mic opened; a later press owns capture.
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    this.stream = stream;
    this.source = this.context.createMediaStreamSource(stream);
    this.source.connect(this.capture);
  }

  // Frames before a flush reply belong to the mic that closed: the draining
  // take keeps those after any earlier reply; a cancelled one drops them.
  onFrame(data) {
    if (data?.kind !== "frame" || !(data.samples instanceof Float32Array)) return;
    this.elements.level.value = Number(data.level) || 0;
    if (this.flushes === 0) {
      if (this.gate.capturing && this.stream) this.frames.push(data.samples);
      return;
    }
    if (this.draining && this.flushes === 1) this.frames.push(data.samples);
    if (data.flush !== true) return;
    this.flushes -= 1;
    if (this.flushes === 0 && this.draining) {
      this.draining = false;
      this.completeTake();
    }
  }

  closeMic() {
    this.micGeneration += 1;
    this.source?.disconnect();
    this.source = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.device = this.stream?.getAudioTracks()[0]?.label ?? this.device ?? "";
    this.stream = null;
    this.elements.level.value = 0;
    // The worklet answers with its buffered partial and clears it, so the
    // next take starts with no samples from this one.
    if (this.capture) {
      this.flushes += 1;
      this.capture.port.postMessage({ kind: "flush" });
    }
  }

  /** The take ends once the worklet's flush reply has drained its tail. */
  finishTake() {
    this.draining = Boolean(this.capture);
    this.closeMic();
    if (!this.draining) this.completeTake();
  }

  completeTake() {
    const samples = concat(this.frames);
    this.frames = [];
    const durationMs = (samples.length / SAMPLE_RATE) * 1_000;
    if (durationMs < MIN_TAKE_MS) {
      this.setStatus("Take too short. Hold Space for the whole prompt.");
      this.render();
      return;
    }
    const peak = samples.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
    this.take = { samples, device: this.device, transcript: { state: "pending" } };
    void this.transcribe(this.take);
    const warnings = [
      peak >= CLIP_LEVEL ? "it clipped" : "",
      this.current()?.recorded ? "this prompt already has a take; saving adds another" : "",
    ].filter(Boolean);
    this.setStatus(
      `Take ready: ${(durationMs / 1_000).toFixed(1)} s, peak ${peak.toFixed(2)}${warnings.length ? `; ${warnings.join("; ")}` : ""}. Enter saves; P plays; hold Space to retake.`,
    );
    this.render();
  }

  /** Display only; the result lands on its own take, so a later take ignores it. */
  async transcribe(take) {
    const settle = (transcript) => {
      if (take.transcript.state !== "pending") return;
      take.transcript = transcript;
      if (this.take === take) this.render();
    };
    const timer = window.setTimeout(() => settle({ state: "failed" }), TRANSCRIBE_TIMEOUT_MS);
    try {
      const response = await fetch("/prompt-reader/transcriptions", {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: take.samples.buffer,
      });
      const body = await response.json();
      settle(response.ok && typeof body.text === "string"
        ? { state: "done", text: body.text, model: String(body.model ?? ""), decodeMs: Number(body.decodeMs) || 0 }
        : { state: "failed" });
    } catch {
      settle({ state: "failed" });
    } finally {
      window.clearTimeout(timer);
    }
  }

  cancelTake(message) {
    if (!this.gate.capturing && !this.take && !this.draining) return;
    this.gate.cancel();
    this.draining = false;
    this.closeMic();
    this.frames = [];
    this.take = null;
    this.setStatus(message);
    this.render();
  }

  async save() {
    const prompt = this.current();
    if (!prompt || !this.take || this.saving) return;
    this.stopPlayback();
    this.saving = true;
    this.render();
    try {
      const query = new URLSearchParams({ promptId: prompt.id, device: this.take.device ?? "" });
      const response = await fetch(`/prompt-reader/recordings?${query}`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: this.take.samples.buffer,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `save failed (${response.status})`);
      prompt.recorded = true;
      prompt.takes = [...(prompt.takes ?? []), body.audio];
      this.take = null;
      this.index += 1;
      this.setStatus(`Saved ${prompt.id} as ${body.audio}.`);
    } catch (error) {
      this.fail(error instanceof Error ? error.message : String(error));
    } finally {
      this.saving = false;
      this.render();
    }
  }

  move(step) {
    if (this.gate.capturing || this.saving) return;
    const next = Math.min(Math.max(this.index + step, 0), this.prompts.length);
    if (next === this.index) return;
    this.stopPlayback();
    this.takeOffset = 0;
    this.index = next;
    this.take = null;
    this.setStatus("Hold Space while you read the prompt aloud, then release.");
    this.render();
  }

  fail(message) {
    this.failed = true;
    this.elements.status.textContent = message;
    this.render();
  }

  setStatus(message) {
    this.failed = false;
    this.elements.status.textContent = message;
  }

  render() {
    const { elements } = this;
    const prompt = this.current();
    const recorded = this.prompts.filter((candidate) => candidate.recorded).length;
    elements.progress.textContent = `${recorded} of ${this.prompts.length} recorded`;
    elements.body.dataset.state = this.gate.capturing
      ? "recording"
      : this.failed
        ? "error"
        : this.take
          ? "review"
          : "ready";
    if (prompt) {
      elements.meta.textContent = `${prompt.id} · ${prompt.stratum} · ${this.index + 1} of ${this.prompts.length}${prompt.recorded ? " · recorded" : ""}`;
      elements.text.textContent = prompt.text;
    } else {
      elements.meta.textContent = "";
      elements.text.textContent = "End of the prompt set. Score the folder with pnpm bench --corpus <dir>.";
    }
    this.renderTranscript(prompt);
    elements.record.disabled = !prompt || this.saving;
    elements.save.disabled = !this.take || this.saving;
    const saved = this.take ? null : this.savedTake();
    elements.play.disabled = this.gate.capturing || this.saving || (!this.take && !saved);
    elements.playLabel.textContent = this.playback
      ? "Playing"
      : saved && saved.count > 1
        ? `Play take ${saved.number} of ${saved.count}`
        : "Play";
    elements.earlier.hidden = !saved || saved.count < 2;
    elements.previous.disabled = this.index === 0;
    elements.next.disabled = !prompt;
  }

  renderTranscript(prompt) {
    const { heard, heardText, heardNote } = this.elements;
    const transcript = this.take?.transcript;
    heard.hidden = !transcript || !prompt;
    if (heard.hidden) return;
    if (transcript.state !== "done") {
      heardText.replaceChildren();
      heardNote.textContent = transcript.state === "pending" ? "Transcribing" : "Transcript unavailable";
      return;
    }
    heardText.replaceChildren(
      ...diffWords(prompt.text, transcript.text).map(({ word, differs }, index) => {
        const span = document.createElement("span");
        span.textContent = `${index ? " " : ""}${word}`;
        if (differs) span.className = "differs";
        return span;
      }),
    );
    heardNote.textContent = transcript.text
      ? `${transcript.model} · ${Math.round(transcript.decodeMs)} ms`
      : `${transcript.model} heard no speech · ${Math.round(transcript.decodeMs)} ms`;
  }
}

/** Case and punctuation insensitive word key; empty for pure punctuation. */
function wordKey(word) {
  return word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
}

/**
 * The heard words in order, each marked when it is outside the longest
 * common word sequence shared with the prompt.
 */
function diffWords(expected, heard) {
  const want = expected.split(/\s+/).map(wordKey).filter(Boolean);
  const words = heard.split(/\s+/).filter(Boolean);
  const keys = words.map(wordKey);
  const lengths = Array.from({ length: keys.length + 1 }, () => new Array(want.length + 1).fill(0));
  for (let i = keys.length - 1; i >= 0; i--) {
    for (let j = want.length - 1; j >= 0; j--) {
      lengths[i][j] = keys[i] && keys[i] === want[j]
        ? lengths[i + 1][j + 1] + 1
        : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }
  const differs = keys.map((key) => key !== "");
  for (let i = 0, j = 0; i < keys.length && j < want.length; ) {
    if (keys[i] && keys[i] === want[j]) {
      differs[i] = false;
      i++;
      j++;
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  return words.map((word, index) => ({ word, differs: differs[index] }));
}

function isSpace(event) {
  return event.code === "Space" || event.key === " ";
}

function concat(frames) {
  const samples = new Float32Array(frames.reduce((sum, frame) => sum + frame.length, 0));
  let offset = 0;
  for (const frame of frames) {
    samples.set(frame, offset);
    offset += frame.length;
  }
  return samples;
}

function describeMicError(error) {
  if (error?.name === "NotAllowedError") return "Microphone permission denied.";
  if (error?.name === "NotFoundError") return "No microphone found.";
  if (!window.isSecureContext) return "The microphone needs a secure origin; open the page on 127.0.0.1.";
  return `Microphone failed: ${error instanceof Error ? error.message : String(error)}`;
}

void new PromptReaderPage().start();
