import { CAPTURE_AUDIO_CONSTRAINTS } from "/capture-constraints.js";
import { HoldReleaseGate } from "/hold-release.js";

/**
 * Prompt reader: walks a prompt set in id order and records one take per
 * prompt through the capture path TM dictation uses: the capture worklet,
 * CAPTURE_AUDIO_CONSTRAINTS, the mic opened at each press and stopped at
 * each commit, and the HoldReleaseGate tail after release. A saved take
 * becomes a corpus pair on the host; the prompt text is its expected
 * transcript.
 */

const SAMPLE_RATE = 16_000;
const MIN_TAKE_MS = 300;
const CLIP_LEVEL = 0.99;

class PromptReaderPage {
  constructor() {
    this.elements = {
      body: document.body,
      progress: document.getElementById("progress"),
      meta: document.getElementById("prompt-meta"),
      text: document.getElementById("prompt-text"),
      level: document.getElementById("level"),
      status: document.getElementById("status"),
      output: document.getElementById("output"),
      record: document.getElementById("record"),
      save: document.getElementById("save"),
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
    this.take = null;
    this.saving = false;
    this.failed = false;
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
    this.elements.previous.addEventListener("click", () => this.move(-1));
    this.elements.next.addEventListener("click", () => this.move(1));
  }

  current() {
    return this.prompts[this.index];
  }

  async press() {
    if (this.saving || !this.current() || !this.gate.press()) return;
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

  onFrame(data) {
    if (data?.kind !== "frame" || !(data.samples instanceof Float32Array)) return;
    this.elements.level.value = Number(data.level) || 0;
    if (this.gate.capturing && this.stream) this.frames.push(data.samples);
  }

  closeMic() {
    this.micGeneration += 1;
    this.source?.disconnect();
    this.source = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.device = this.stream?.getAudioTracks()[0]?.label ?? this.device ?? "";
    this.stream = null;
    this.elements.level.value = 0;
  }

  finishTake() {
    this.closeMic();
    const samples = concat(this.frames);
    this.frames = [];
    const durationMs = (samples.length / SAMPLE_RATE) * 1_000;
    if (durationMs < MIN_TAKE_MS) {
      this.setStatus("Take too short. Hold Space for the whole prompt.");
      this.render();
      return;
    }
    const peak = samples.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
    this.take = { samples, device: this.device };
    const warnings = [
      peak >= CLIP_LEVEL ? "it clipped" : "",
      this.current()?.recorded ? "this prompt already has a take; saving adds another" : "",
    ].filter(Boolean);
    this.setStatus(
      `Take ready: ${(durationMs / 1_000).toFixed(1)} s, peak ${peak.toFixed(2)}${warnings.length ? `; ${warnings.join("; ")}` : ""}. Enter saves; hold Space to retake.`,
    );
    this.render();
  }

  cancelTake(message) {
    if (!this.gate.capturing && !this.take) return;
    this.gate.cancel();
    this.closeMic();
    this.frames = [];
    this.take = null;
    this.setStatus(message);
    this.render();
  }

  async save() {
    const prompt = this.current();
    if (!prompt || !this.take || this.saving) return;
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
    elements.record.disabled = !prompt || this.saving;
    elements.save.disabled = !this.take || this.saving;
    elements.previous.disabled = this.index === 0;
    elements.next.disabled = !prompt;
  }
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
