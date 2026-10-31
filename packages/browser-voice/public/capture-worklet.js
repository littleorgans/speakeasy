const TARGET_RATE = 16_000;
const FRAME_MS = 20;

class SpeakEasyCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.outputSamples = Math.round((TARGET_RATE * FRAME_MS) / 1_000);
    this.source = new Float32Array(4096);
    this.sourceLength = 0;
    this.endFrame = 0;
    this.port.onmessage = (event) => {
      if (event.data?.kind === "flush") this.flush();
    };
  }

  // endFrame: context frame just past `input`. Each frame message carries
  // audioTime, the context time (s) at that frame's last source sample.
  ingest(input, endFrame) {
    if (this.sourceLength + input.length > this.source.length) {
      const expanded = new Float32Array(this.source.length * 2);
      expanded.set(this.source.subarray(0, this.sourceLength));
      this.source = expanded;
    }
    this.source.set(input, this.sourceLength);
    this.sourceLength += input.length;
    this.endFrame = endFrame;

    const needed = Math.ceil(this.outputSamples * this.ratio);
    while (this.sourceLength >= needed) {
      const consumed = Math.floor(this.outputSamples * this.ratio);
      this.post(this.outputSamples, consumed);
    }
  }

  // Answers every flush with exactly one frame marked `flush: true`: the
  // buffered partial resampled, or empty when nothing is buffered. Then the
  // buffer is cleared, so the next frame holds no earlier samples.
  flush() {
    const count = Math.floor(this.sourceLength / this.ratio);
    this.post(count, this.sourceLength, { flush: true });
  }

  post(count, consumed, marker = {}) {
    const frame = new Float32Array(count);
    let energy = 0;
    if (Math.abs(this.ratio - 3) < 0.0001) {
      for (let index = 0; index < frame.length; index += 1) {
        const sourceIndex = index * 3;
        const sample =
          (this.source[sourceIndex] +
            this.source[sourceIndex + 1] +
            this.source[sourceIndex + 2]) /
          3;
        frame[index] = sample;
        energy += sample * sample;
      }
    } else {
      for (let index = 0; index < frame.length; index += 1) {
        const position = index * this.ratio;
        const left = Math.floor(position);
        const blend = position - left;
        const a = this.source[left];
        const b = left + 1 < this.sourceLength ? this.source[left + 1] : a;
        const sample = a + (b - a) * blend;
        frame[index] = sample;
        energy += sample * sample;
      }
    }
    this.source.copyWithin(0, consumed, this.sourceLength);
    this.sourceLength -= consumed;
    this.port.postMessage(
      {
        kind: "frame",
        samples: frame,
        level: frame.length ? Math.sqrt(energy / frame.length) : 0,
        audioTime: (this.endFrame - this.sourceLength) / sampleRate,
        ...marker,
      },
      [frame.buffer],
    );
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input?.length) this.ingest(input, currentFrame + input.length);
    for (const channel of outputs[0] ?? []) channel.fill(0);
    return true;
  }
}

registerProcessor("speak-easy-capture", SpeakEasyCapture);
