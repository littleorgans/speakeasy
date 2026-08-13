const TARGET_RATE = 16_000;
const FRAME_MS = 20;

class SpeakEasyCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.outputSamples = Math.round((TARGET_RATE * FRAME_MS) / 1_000);
    this.source = new Float32Array(4096);
    this.sourceLength = 0;
  }

  ingest(input) {
    if (this.sourceLength + input.length > this.source.length) {
      const expanded = new Float32Array(this.source.length * 2);
      expanded.set(this.source.subarray(0, this.sourceLength));
      this.source = expanded;
    }
    this.source.set(input, this.sourceLength);
    this.sourceLength += input.length;

    const needed = Math.ceil(this.outputSamples * this.ratio);
    while (this.sourceLength >= needed) {
      const frame = new Float32Array(this.outputSamples);
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
          const b = this.source[left + 1] ?? a;
          const sample = a + (b - a) * blend;
          frame[index] = sample;
          energy += sample * sample;
        }
      }
      const consumed = Math.floor(this.outputSamples * this.ratio);
      this.source.copyWithin(0, consumed, this.sourceLength);
      this.sourceLength -= consumed;
      this.port.postMessage(
        { kind: "frame", samples: frame, level: Math.sqrt(energy / frame.length) },
        [frame.buffer],
      );
    }
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input?.length) this.ingest(input);
    for (const channel of outputs[0] ?? []) channel.fill(0);
    return true;
  }
}

registerProcessor("speak-easy-capture", SpeakEasyCapture);
