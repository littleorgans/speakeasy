/** speak-easy declarations for the vendored backend.js; see NOTICE. */
import type { InferenceSession } from "onnxruntime-node";

export class OrtBackend {
  constructor(ort: unknown, enc: InferenceSession, dec: InferenceSession);
  release(): void;
}
