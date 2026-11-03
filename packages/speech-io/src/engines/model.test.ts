import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import type { SilenceGatePolicy } from "./offline-session.ts";
import { resolveGraniteFiles } from "./ort-granite-models.ts";
import { ORT_MODELS, type OrtModel } from "./ort-models.ts";
import { resolveOrtTdtPaths } from "./ort-tdt-models.ts";
import { resolveWhistleFiles } from "./ort-whistle-models.ts";
import { resolveModelPaths, SHERPA_MODELS, type SherpaModel } from "./sherpa-models.ts";

/**
 * Every registry id, pinned: the sha256 of canonical JSON (keys sorted,
 * arrays in order) of its resolved asset under a fixed root, licence, cap,
 * silence gate, kind and threads. A change to any of them means a moved
 * install, a new download or a rehash of an existing one. On a mismatch
 * the message carries the canonical JSON.
 */

const ROOT = "/models";

const EXPECTED: Record<string, string> = {
  "en-parakeet-tdt-0.6b-v2-int8": "6c097e3ee031b3bcdd721d0c24cf18346a52d5946a952d531455516b14782439",
  "en-moonshine-base-int8": "c1462888c5db949d2785dd64f57fcea90b577e398cb78af5a1f568226578d55d",
  "en-parakeet-ultra-int8": "e5c3a4f87cfc1dac50a5957485d80435592b5139d4429142a47a15d402176d45",
  "en-canary-180m-flash-int8": "c67807b9f5ea13ced0b23085bdeed2cd2ef2f4e847026ae533cdbbed1c65d383",
  "en-qwen3-asr-0.6b-int8-2026-03-25": "7e7fc17e0da87be8e5b5661fb4da26a5e197175f20fbee7efa6e04ccf58e2e17",
  "en-2023-06-21": "812860ac185b17eb6a42689f0b9f67fa4d351305bba53993b17bc99ed1b001f6",
  "en-kroko-2025-08-06": "c2e363ed380e3da21a01ecee63a4f1840c8e8f4e13deec92d4297c41fd5dc5c2",
  "en-2023-06-26": "1c7b0666a58a347d1ebf2dcc73fb82683ad33b71e680ff471fd1dd69683472d0",
  "en-2023-02-21": "74e84e62ad8a9821201ad23c599cee26fd41bd7417b1f0ed5d20ff37cb0c1e13",
  "whistle-onnx-fp32": "147acc4f0efa79747f78edc877f6a6f3f03c732822fcf695096837661c833d9a",
  "en-granite-speech-5.0-470m-turboctc-int8": "e1db39304a4d6d12ecb1d30a81753c5335d363b62a30575e079091577142aaec",
  "en-fluid-speech-mini": "c2fa98506b18907a7740e0f00d4ded4249e922ec7103a5fad920cd0c3c5e0c30",
  "en-phonon-2-int8": "56b9874d4e9934ab033eb58868e2db2eca063f17f7a933e352a531063b00cf03",
  "en-parakeet-redux": "5da926b1124119142bc41f1f188ec240539f12d445b25c96638d16a0abc46740",
};

function resolved(model: SherpaModel | OrtModel) {
  if (!("family" in model)) return resolveModelPaths(model, ROOT);
  switch (model.family) {
    case "granite-ctc":
      return resolveGraniteFiles(model, ROOT);
    case "whistle":
      return resolveWhistleFiles(model, ROOT);
    case "parakeet-tdt":
      return resolveOrtTdtPaths(model, ROOT);
    default:
      return resolveModelPaths(model, ROOT);
  }
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, inner: unknown) =>
    inner !== null && typeof inner === "object" && !Array.isArray(inner)
      ? Object.fromEntries(Object.entries(inner).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
      : inner,
  );
}

function pinnedView(model: SherpaModel | OrtModel): string {
  // Streaming descriptors have no gate and sherpa ones no threads.
  const optional = model as { silenceGate?: SilenceGatePolicy; threads?: number };
  return canonical({
    asset: resolved(model),
    licence: model.licence,
    maxAudioMs: model.maxAudioMs,
    silenceGate: optional.silenceGate ?? null,
    kind: model.kind,
    threads: optional.threads ?? null,
  });
}

test("every model id resolves to its pinned asset and metadata", () => {
  const registry: Record<string, SherpaModel | OrtModel> = { ...SHERPA_MODELS, ...ORT_MODELS };
  assert.deepEqual(Object.keys(registry), Object.keys(EXPECTED));
  for (const [id, model] of Object.entries(registry)) {
    const view = pinnedView(model);
    assert.equal(createHash("sha256").update(view).digest("hex"), EXPECTED[id], `${id}: ${view}`);
  }
});
