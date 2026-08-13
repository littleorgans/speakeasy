class SpeakEasyPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.playbackId = 0;
    this.inputRate = 24_000;
    this.step = this.inputRate / sampleRate;
    this.queue = [];
    this.readIndex = 0;
    this.fraction = 0;
    this.ended = false;
    this.reportedDrain = false;
    this.fade = 0;
    this.playedInputSamples = 0;
    this.port.onmessage = (event) => this.handle(event.data);
  }

  handle(message) {
    if (!message || typeof message !== "object") return;
    if (message.kind === "start") {
      this.playbackId = message.playbackId;
      this.inputRate = message.sampleRate;
      this.step = this.inputRate / sampleRate;
      this.queue.length = 0;
      this.readIndex = 0;
      this.fraction = 0;
      this.ended = false;
      this.reportedDrain = false;
      this.fade = 48;
      this.playedInputSamples = 0;
    } else if (message.kind === "audio" && message.playbackId === this.playbackId) {
      this.queue.push(message.samples);
    } else if (message.kind === "end" && message.playbackId === this.playbackId) {
      this.ended = true;
      this.maybeReportDrain();
    } else if (message.kind === "clear" && message.playbackId === this.playbackId) {
      this.queue.length = 0;
      this.readIndex = 0;
      this.fraction = 0;
      this.ended = true;
      this.maybeReportDrain();
    }
  }

  read() {
    if (this.queue.length === 0) return 0;
    const current = this.queue[0];
    const a = current[this.readIndex] ?? 0;
    const b =
      current[this.readIndex + 1] ??
      this.queue[1]?.[0] ??
      a;
    const value = a + (b - a) * this.fraction;
    this.fraction += this.step;
    while (this.fraction >= 1) {
      this.fraction -= 1;
      this.readIndex += 1;
      this.playedInputSamples += 1;
    }
    while (this.queue.length && this.readIndex >= this.queue[0].length) {
      this.readIndex -= this.queue[0].length;
      this.queue.shift();
    }
    return value;
  }

  maybeReportDrain() {
    if (this.ended && this.queue.length === 0 && !this.reportedDrain) {
      this.reportedDrain = true;
      this.port.postMessage({
        kind: "drained",
        playbackId: this.playbackId,
        audioEndMs: Math.floor((this.playedInputSamples / this.inputRate) * 1000),
      });
    }
  }

  process(_inputs, outputs) {
    const channels = outputs[0] ?? [];
    const primary = channels[0];
    if (!primary) return true;
    for (let index = 0; index < primary.length; index += 1) {
      let value = this.read();
      if (this.fade > 0) {
        value *= 1 - this.fade / 48;
        this.fade -= 1;
      }
      primary[index] = value;
      for (let channel = 1; channel < channels.length; channel += 1) {
        channels[channel][index] = value;
      }
    }
    this.maybeReportDrain();
    return true;
  }
}

registerProcessor("speak-easy-playback", SpeakEasyPlayback);
