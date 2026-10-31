import { readFileSync } from "node:fs";
import vm from "node:vm";

export type FrameMessage = {
  kind: string;
  samples: Float32Array;
  level: number;
  audioTime: number;
  flush?: boolean;
};

export type CaptureInstance = {
  port: { onmessage?: (event: { data: unknown }) => void };
  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
};

export function feed(
  context: { currentFrame: number },
  worklet: CaptureInstance,
  fromBlock: number,
  toBlock: number,
  value: number,
): void {
  for (let block = fromBlock; block < toBlock; block += 1) {
    context.currentFrame = block * 128;
    worklet.process([[new Float32Array(128).fill(value)]], [[new Float32Array(128)]]);
  }
}

export function flush(worklet: CaptureInstance): void {
  const handler = worklet.port.onmessage;
  if (!handler) throw new Error("capture processor accepts no port messages");
  handler({ data: { kind: "flush" } });
}

/** Runs the shipped capture worklet in a VM; `onPost` sees each posted frame. */
export function loadCapture(
  sampleRate: number,
  onPost: (message: FrameMessage) => void = () => {},
): {
  context: { currentFrame: number };
  messages: FrameMessage[];
  Processor: new () => CaptureInstance;
} {
  const messages: FrameMessage[] = [];
  let registered: (new () => CaptureInstance) | undefined;
  class AudioWorkletProcessorFake {
    readonly port: CaptureInstance["port"] & {
      postMessage(message: FrameMessage): void;
    } = {
      postMessage: (message: FrameMessage) => {
        messages.push(message);
        onPost(message);
      },
    };
  }
  const context = vm.createContext({
    AudioWorkletProcessor: AudioWorkletProcessorFake,
    Float32Array,
    Math,
    sampleRate,
    currentFrame: 0,
    registerProcessor: (_name: string, Processor: new () => CaptureInstance) => {
      registered = Processor;
    },
  }) as { currentFrame: number };
  const source = readFileSync(
    new URL("../public/capture-worklet.js", import.meta.url),
    "utf8",
  );
  vm.runInContext(source, context);
  if (!registered) throw new Error("capture processor did not register");
  return { context, messages, Processor: registered };
}
