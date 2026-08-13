import { HoldReleaseGate } from "/hold-release.js";

const AUDIO_HEADER_BYTES = 9;
const AUDIO_PACKET_KIND = 1;
const SETTINGS_KEY = "speak-easy.browser-room";

class SpeakEasyRoom {
  constructor() {
    this.socket = null;
    this.socketPromise = null;
    this.audioContext = null;
    this.micStream = null;
    this.captureNode = null;
    this.playbackNode = null;
    this.sessionPhase = "stopped";
    this.state = "idle";
    this.muted = false;
    this.activeSettings = null;
    this.level = 0;
    this.turnNumber = 0;
    this.currentUser = null;
    this.currentAssistant = null;
    this.toastTimer = 0;
    this.elements = this.collectElements();
    this.hold = new HoldReleaseGate({
      onCommit: () => {
        this.renderControl();
        if (this.sessionPhase === "ready") this.send({ type: "commit-input" });
      },
      onChange: () => {
        document.body.dataset.holding = String(this.hold.held);
        this.renderControl();
      },
    });
    this.settings = this.loadSettings();
    this.createSignalBars();
    this.bindControls();
    this.applySettings();
    this.animateSignal();
  }

  collectElements() {
    const byId = (id) => document.getElementById(id);
    return {
      connection: byId("connection-label"),
      statusWord: byId("status-word"),
      kicker: byId("state-kicker"),
      caption: byId("state-caption"),
      main: byId("main-control"),
      mainLabel: byId("main-control-label"),
      mainHint: byId("main-control-hint"),
      mute: byId("mute-button"),
      muteLabel: byId("mute-label"),
      end: byId("end-button"),
      settingsButton: byId("settings-button"),
      settingsDialog: byId("settings-dialog"),
      modeOptions: [...document.querySelectorAll('input[name="conversation-mode"]')],
      pauseField: byId("pause-field"),
      pause: byId("pause-input"),
      microphone: byId("microphone-select"),
      voice: byId("voice-select"),
      systemPrompt: byId("system-prompt"),
      bargeField: byId("barge-field"),
      barge: byId("barge-toggle"),
      saveSettings: byId("save-settings"),
      signalBars: byId("signal-bars"),
      turnList: byId("turn-list"),
      empty: byId("empty-conversation"),
      transcriptScroll: byId("transcript-scroll"),
      clear: byId("clear-button"),
      metricFinal: byId("metric-final"),
      metricToken: byId("metric-token"),
      metricAudio: byId("metric-audio"),
      runtime: byId("runtime-label"),
      toast: byId("toast"),
    };
  }

  bindControls() {
    this.elements.main.addEventListener("click", () => {
      if (this.sessionPhase === "stopped") void this.start();
      else if (!this.usesHoldMode() && (this.state === "thinking" || this.state === "speaking")) {
        this.interrupt();
      } else if (!this.usesHoldMode() && this.sessionPhase === "ready") {
        this.toggleMute();
      }
    });
    this.elements.main.addEventListener("pointerdown", (event) => {
      if (!this.usesHoldMode() || this.sessionPhase !== "ready") return;
      event.preventDefault();
      this.elements.main.setPointerCapture?.(event.pointerId);
      this.beginHold();
    });
    window.addEventListener("pointerup", () => this.endHold());
    window.addEventListener("pointercancel", () => this.endHold());
    this.elements.mute.addEventListener("click", () => this.toggleMute());
    this.elements.end.addEventListener("click", () => void this.end());
    this.elements.clear.addEventListener("click", () => this.clearTranscript());
    this.elements.settingsButton.addEventListener("click", () => void this.openSettings());
    this.elements.saveSettings.addEventListener("click", () => this.saveSettings());
    this.elements.modeOptions.forEach((option) => {
      option.addEventListener("change", () => this.renderSettingsMode());
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !this.elements.settingsDialog.open) {
        this.interrupt();
      }
      if (
        event.code === "Space" &&
        !event.repeat &&
        !this.elements.settingsDialog.open &&
        !isTextEntry(event.target) &&
        this.usesHoldMode() &&
        this.sessionPhase === "ready"
      ) {
        event.preventDefault();
        this.beginHold();
      }
    });
    window.addEventListener("keyup", (event) => {
      if (event.code === "Space" && this.hold.held) {
        event.preventDefault();
        this.endHold();
      }
    });
    window.addEventListener("beforeunload", () => {
      this.send({ type: "stop" });
      this.socket?.close();
    });
  }

  async start() {
    this.activeSettings = { ...this.settings };
    this.setSession("loading");
    try {
      await this.setupAudio();
      await this.connect();
      this.send({
        type: "start",
        mode: this.activeSettings.mode,
        pauseMs: this.activeSettings.pauseMs,
        voice: this.activeSettings.voice,
        barge: this.activeSettings.barge,
        ...(this.activeSettings.systemPrompt
          ? { systemPrompt: this.activeSettings.systemPrompt }
          : {}),
      });
    } catch (error) {
      this.showError(this.errorMessage(error));
      await this.teardownAudio();
      this.activeSettings = null;
      this.setSession("stopped");
    }
  }

  async end() {
    this.hold.cancel();
    this.send({ type: "stop" });
    await this.teardownAudio();
    this.setState("idle");
    this.setSession("stopped");
  }

  interrupt() {
    if (this.state !== "thinking" && this.state !== "speaking") return;
    this.send({ type: "interrupt" });
  }

  toggleMute() {
    if (this.sessionPhase !== "ready" || this.usesHoldMode()) return;
    this.muted = !this.muted;
    document.body.dataset.muted = String(this.muted);
    this.elements.muteLabel.textContent = this.muted ? "Unmute mic" : "Mute mic";
    this.elements.mainHint.textContent = this.muted ? "Microphone paused" : "Tap to mute";
    if (this.muted) this.level = 0;
  }

  beginHold() {
    if (!this.usesHoldMode() || this.sessionPhase !== "ready" || !this.hold.press()) return;
    if (this.state === "thinking" || this.state === "speaking") this.interrupt();
  }

  endHold() {
    this.hold.release();
  }

  usesHoldMode() {
    return (this.activeSettings ?? this.settings).mode === "hold";
  }

  async connect() {
    if (this.socket?.readyState === WebSocket.OPEN) return;
    if (this.socketPromise) return this.socketPromise;
    this.socketPromise = new Promise((resolve, reject) => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${scheme}://${location.host}/voice`);
      socket.binaryType = "arraybuffer";
      const failed = () => reject(new Error("Could not open the local voice channel"));
      socket.addEventListener("open", () => {
        socket.removeEventListener("error", failed);
        this.socket = socket;
        this.wireSocket(socket);
        resolve();
      }, { once: true });
      socket.addEventListener("error", failed, { once: true });
    }).finally(() => {
      this.socketPromise = null;
    });
    return this.socketPromise;
  }

  wireSocket(socket) {
    socket.addEventListener("message", (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.acceptAudioPacket(event.data);
        return;
      }
      try {
        this.acceptEvent(JSON.parse(event.data));
      } catch {
        this.showError("The voice host sent an unreadable event");
      }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      void this.teardownAudio();
      this.setState("idle");
      this.setSession("stopped");
      this.showError("The local voice channel closed");
    });
  }

  send(message) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  acceptEvent(event) {
    switch (event.type) {
      case "session":
        this.setSession(event.phase, event.label);
        if (event.phase === "stopped") void this.teardownAudio();
        break;
      case "state":
        this.setState(event.state);
        break;
      case "transcript":
        this.updateTranscript(event);
        break;
      case "metrics":
        this.updateMetrics(event.metrics);
        break;
      case "interrupted":
        this.markInterrupted();
        break;
      case "notice":
        if (event.level === "error") this.showError(event.message);
        else console.info(`[speak-easy] ${event.message}`);
        break;
      case "playback":
        this.handlePlayback(event);
        break;
    }
  }

  setSession(phase, label) {
    this.sessionPhase = phase;
    if (phase === "stopped") {
      this.hold.cancel();
      this.activeSettings = null;
      this.setState("idle");
    }
    document.body.dataset.session = phase;
    const copy = {
      stopped: "Room closed",
      loading: "Preparing local engines",
      ready: "Private room open",
    };
    this.elements.connection.textContent = copy[phase] ?? phase;
    this.elements.end.disabled = phase === "stopped";
    if (label) this.elements.runtime.textContent = label;
    this.renderControl();
  }

  setState(state) {
    this.state = state;
    document.body.dataset.state = state;
    const states = {
      idle: {
        word: "Come in.",
        kicker: "A quiet line, ready when you are",
        caption: "Start the room, speak naturally, then pause. Your reply will arrive as audio.",
      },
      listening: {
        word: "I’m listening.",
        kicker: "Your side of the line is open",
        caption: "Take your time. A natural pause closes the turn and starts the reply.",
      },
      thinking: {
        word: "One moment.",
        kicker: "The room is shaping a reply",
        caption: "Tap the signal control or press Escape if you want to take the floor back.",
      },
      speaking: {
        word: "Here it comes.",
        kicker: "The reply is playing now",
        caption: "Speak over it or tap the signal control to interrupt immediately.",
      },
    };
    const holdStates = {
      idle: {
        word: "Come in.",
        kicker: "A quiet line, ready when you are",
        caption: "Enter the room, then hold the signal control or Space while you speak.",
      },
      listening: {
        word: "Your turn.",
        kicker: "The line opens when you hold",
        caption: "Hold the signal control or Space. Release when your thought is complete.",
      },
      thinking: {
        word: "One moment.",
        kicker: "The room is shaping a reply",
        caption: "Hold the signal control or Space to interrupt and take the floor.",
      },
      speaking: {
        word: "Here it comes.",
        kicker: "The reply is playing now",
        caption: "Hold the signal control or Space to interrupt and speak immediately.",
      },
    };
    const copy = (this.usesHoldMode() ? holdStates : states)[state] ?? states.idle;
    this.elements.statusWord.textContent = copy.word;
    this.elements.kicker.textContent = copy.kicker;
    this.elements.caption.textContent = copy.caption;
    this.renderControl();
  }

  renderControl() {
    const holdMode = this.usesHoldMode();
    document.body.dataset.mode = holdMode ? "hold" : "natural";
    this.elements.mute.hidden = holdMode;
    this.elements.mute.disabled = this.sessionPhase !== "ready" || holdMode;
    if (this.sessionPhase === "loading") {
      this.elements.mainLabel.textContent = "Warming up";
      this.elements.mainHint.textContent = "Loading local speech models";
      this.elements.main.setAttribute("aria-label", "Preparing the voice room");
      return;
    }
    if (this.sessionPhase === "stopped") {
      this.elements.mainLabel.textContent = "Enter room";
      this.elements.mainHint.textContent = "Microphone permission required";
      this.elements.main.setAttribute("aria-label", "Enter voice room");
      return;
    }
    if (this.usesHoldMode()) {
      this.elements.mainLabel.textContent = this.hold.held
        ? "Release to send"
        : this.hold.releasing
          ? "Sending"
          : "Hold to speak";
      this.elements.mainHint.textContent = this.hold.capturing
        ? this.hold.held ? "Listening now" : "Finishing your words"
        : "Button or Space";
      this.elements.main.setAttribute(
        "aria-label",
        this.hold.held
          ? "Release to send your message"
          : this.hold.releasing
            ? "Sending your message"
            : "Hold to speak",
      );
      return;
    }
    if (this.state === "thinking" || this.state === "speaking") {
      this.elements.mainLabel.textContent = "Interrupt";
      this.elements.mainHint.textContent = "Tap or press Escape";
      this.elements.main.setAttribute("aria-label", "Interrupt the reply");
      return;
    }
    this.elements.mainLabel.textContent = this.muted ? "Mic paused" : "Listening";
    this.elements.mainHint.textContent = this.muted ? "Tap to resume" : "Tap to mute";
    this.elements.main.setAttribute("aria-label", this.muted ? "Resume microphone" : "Pause microphone");
  }

  updateTranscript(event) {
    this.elements.empty.hidden = true;
    if (event.role === "user") {
      if (!this.currentUser) this.currentUser = this.createTurn("user");
      this.applyTranscript(this.currentUser, event);
      if (event.final) this.currentUser = null;
    } else {
      if (!this.currentAssistant) this.currentAssistant = this.createTurn("assistant");
      this.applyTranscript(this.currentAssistant, event);
      if (event.final) this.currentAssistant = null;
    }
    this.elements.transcriptScroll.scrollTo({
      top: this.elements.transcriptScroll.scrollHeight,
      behavior: "smooth",
    });
  }

  createTurn(role) {
    this.turnNumber += 1;
    const item = document.createElement("li");
    item.className = "turn";
    item.dataset.role = role;
    item.dataset.final = "false";
    item.dataset.index = String(this.turnNumber).padStart(2, "0");
    const label = document.createElement("p");
    label.className = "turn-label";
    label.textContent = role === "user" ? "You" : "Speak Easy";
    const text = document.createElement("p");
    text.className = "turn-text";
    item.append(label, text);
    this.elements.turnList.append(item);
    return item;
  }

  applyTranscript(item, event) {
    const text = item.querySelector(".turn-text");
    text.textContent = event.mode === "append" ? text.textContent + event.text : event.text;
    item.dataset.final = String(event.final);
  }

  markInterrupted() {
    if (this.currentAssistant) {
      this.currentAssistant.dataset.interrupted = "true";
      this.currentAssistant.dataset.final = "true";
      const note = document.createElement("span");
      note.className = "turn-note";
      note.textContent = "Interrupted";
      this.currentAssistant.append(note);
      this.currentAssistant = null;
    }
  }

  updateMetrics(metrics) {
    const format = (value) => `${Math.max(0, Math.round(value))} ms`;
    this.elements.metricFinal.textContent = format(metrics.endpointToFinalMs);
    this.elements.metricToken.textContent = format(metrics.endpointToFirstTokenMs);
    this.elements.metricAudio.textContent = format(metrics.endpointToFirstAudioMs);
  }

  clearTranscript() {
    this.elements.turnList.replaceChildren();
    this.elements.empty.hidden = false;
    this.turnNumber = 0;
    this.currentUser = null;
    this.currentAssistant = null;
  }

  handlePlayback(event) {
    this.playbackId = event.playbackId;
    this.playbackNode?.port.postMessage({
      kind: event.action,
      playbackId: event.playbackId,
      sampleRate: event.sampleRate,
    });
  }

  acceptAudioPacket(packet) {
    if (packet.byteLength < AUDIO_HEADER_BYTES || !this.playbackNode) return;
    const header = new DataView(packet);
    if (header.getUint8(0) !== AUDIO_PACKET_KIND) return;
    const playbackId = header.getUint32(1, true);
    const sampleRate = header.getUint32(5, true);
    const payload = packet.slice(AUDIO_HEADER_BYTES);
    if (payload.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) return;
    const samples = new Float32Array(payload);
    this.playbackNode.port.postMessage(
      { kind: "audio", playbackId, sampleRate, samples },
      [samples.buffer],
    );
  }

  async setupAudio() {
    await this.teardownAudio();
    const device = this.settings.microphoneId;
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        ...(device ? { deviceId: { exact: device } } : {}),
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const context = new AudioContext({ latencyHint: "interactive" });
    await Promise.all([
      context.audioWorklet.addModule("/capture-worklet.js"),
      context.audioWorklet.addModule("/playback-worklet.js"),
    ]);
    this.audioContext = context;
    this.captureNode = new AudioWorkletNode(context, "speak-easy-capture");
    this.playbackNode = new AudioWorkletNode(context, "speak-easy-playback", {
      outputChannelCount: [2],
    });
    const source = context.createMediaStreamSource(this.micStream);
    const silent = context.createGain();
    silent.gain.value = 0;
    source.connect(this.captureNode).connect(silent).connect(context.destination);
    this.playbackNode.connect(context.destination);
    this.captureNode.port.onmessage = (event) => {
      this.level = Number(event.data?.level) || 0;
      if (
        !this.muted &&
        (!this.usesHoldMode() || this.hold.capturing) &&
        this.sessionPhase === "ready" &&
        this.socket?.readyState === WebSocket.OPEN &&
        event.data?.samples instanceof Float32Array
      ) {
        this.socket.send(event.data.samples.buffer);
      }
    };
    this.playbackNode.port.onmessage = (event) => {
      if (event.data?.kind === "drained") {
        this.send({
          type: "playback-drained",
          playbackId: event.data.playbackId,
          audioEndMs: event.data.audioEndMs,
        });
      }
    };
    await context.resume();
    await this.refreshDevices();
  }

  async teardownAudio() {
    this.level = 0;
    this.micStream?.getTracks().forEach((track) => track.stop());
    this.micStream = null;
    this.captureNode?.disconnect();
    this.playbackNode?.disconnect();
    this.captureNode = null;
    this.playbackNode = null;
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close();
    }
    this.audioContext = null;
  }

  openSettings() {
    this.elements.settingsDialog.showModal();
    if (navigator.mediaDevices?.enumerateDevices) void this.refreshDevices();
  }

  async refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === "audioinput",
    );
    const selected = this.elements.microphone.value || this.settings.microphoneId;
    this.elements.microphone.replaceChildren(new Option("System default", ""));
    devices.forEach((device, index) => {
      this.elements.microphone.add(
        new Option(device.label || `Microphone ${index + 1}`, device.deviceId),
      );
    });
    this.elements.microphone.value = devices.some((device) => device.deviceId === selected)
      ? selected
      : "";
  }

  loadSettings() {
    const defaults = {
      mode: "natural",
      pauseMs: Number(this.elements.pause.value),
      voice: this.elements.voice.value,
      microphoneId: "",
      systemPrompt: "",
      barge: true,
    };
    try {
      const stored = { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") };
      const voices = [...this.elements.voice.options].map((option) => option.value);
      return {
        mode: stored.mode === "hold" ? "hold" : "natural",
        pauseMs: clampPause(stored.pauseMs, this.elements.pause),
        voice: voices.includes(stored.voice) ? stored.voice : defaults.voice,
        microphoneId: typeof stored.microphoneId === "string" ? stored.microphoneId : "",
        systemPrompt: typeof stored.systemPrompt === "string" ? stored.systemPrompt : "",
        barge: stored.barge !== false,
      };
    } catch {
      return defaults;
    }
  }

  applySettings() {
    const mode = this.elements.modeOptions.find((option) => option.value === this.settings.mode);
    if (mode) mode.checked = true;
    this.elements.pause.value = String(this.settings.pauseMs);
    this.elements.microphone.value = this.settings.microphoneId;
    this.elements.voice.value = this.settings.voice;
    this.elements.systemPrompt.value = this.settings.systemPrompt;
    this.elements.barge.checked = this.settings.barge;
    this.renderSettingsMode();
    this.setState(this.state);
  }

  saveSettings() {
    const mode = this.elements.modeOptions.find((option) => option.checked)?.value ?? "natural";
    if (mode === "natural" && !this.elements.pause.reportValidity()) return;
    this.settings = {
      mode,
      pauseMs: clampPause(this.elements.pause.value, this.elements.pause),
      microphoneId: this.elements.microphone.value,
      voice: this.elements.voice.value,
      systemPrompt: this.elements.systemPrompt.value.trim(),
      barge: this.elements.barge.checked,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));
    if (this.sessionPhase === "ready") {
      this.showToast("Saved. End and re-enter the room to apply voice or conversation changes.");
    }
    this.setState(this.state);
  }

  renderSettingsMode() {
    const mode = this.elements.modeOptions.find((option) => option.checked)?.value;
    this.elements.pauseField.hidden = mode === "hold";
    this.elements.pause.disabled = mode === "hold";
    this.elements.bargeField.hidden = mode === "hold";
  }

  createSignalBars() {
    for (let index = 0; index < 34; index += 1) {
      const bar = document.createElement("span");
      bar.className = "signal-bar";
      bar.style.setProperty("--offset", String(index));
      this.elements.signalBars.append(bar);
    }
    this.bars = [...this.elements.signalBars.children];
  }

  animateSignal() {
    const now = performance.now() / 220;
    const gated = this.muted || (this.usesHoldMode() && !this.hold.capturing);
    const activeLevel = gated ? 0 : Math.min(1, this.level * 5.5);
    this.bars.forEach((bar, index) => {
      const falloff = 0.34 + 0.66 * Math.sin((index / this.bars.length) * Math.PI);
      const movement = 0.72 + 0.28 * Math.sin(now + index * 0.54);
      const thinkingPulse = this.state === "thinking" ? 0.13 * movement : 0;
      const speakingPulse = this.state === "speaking" ? 0.22 * movement : 0;
      const scale = Math.min(1, activeLevel * falloff * movement + thinkingPulse + speakingPulse);
      bar.style.setProperty("--bar-scale", scale.toFixed(3));
    });
    this.level *= 0.91;
    requestAnimationFrame(() => this.animateSignal());
  }

  showError(message) {
    this.showToast(message);
  }

  showToast(message) {
    clearTimeout(this.toastTimer);
    this.elements.toast.textContent = message;
    this.elements.toast.hidden = false;
    this.toastTimer = window.setTimeout(() => {
      this.elements.toast.hidden = true;
    }, 6500);
  }

  errorMessage(error) {
    if (error instanceof DOMException && error.name === "NotAllowedError") {
      return "Microphone access was declined. Allow it in the browser, then try again.";
    }
    return error instanceof Error ? error.message : String(error);
  }
}

function clampPause(value, input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const pause = Number(value);
  return Math.min(max, Math.max(min, Number.isFinite(pause) ? Math.round(pause) : Number(input.value)));
}

function isTextEntry(target) {
  return target instanceof HTMLElement &&
    (target.matches("input, textarea, select") || target.isContentEditable);
}

new SpeakEasyRoom();
