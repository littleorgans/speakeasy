import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import {
  DEFAULT_LOCAL_OUTPUT,
  LocalSpeechOutputFactory,
  SpeechOutputFactory,
  type SpeechOutputConfig,
  type SpeechOutputContext,
} from "@speakeasy/speech-io";
import { createConversationRuntime, type RuntimeOptions } from "./index.ts";
import { CascadeResponder } from "./responder/cascade.ts";
import { OpenAIRealtimeResponder } from "./responder/openai-realtime.ts";
import { FakeSTT, FakeTTS } from "./test-support.ts";

const env = {
  INCEPTIONLABS_API_KEY: "fake-mercury-key",
  CEREBRAS_API_KEY: "fake-cerebras-key",
  CARTESIA_API_KEY: "fake-cartesia-key",
  OPENAI_API_KEY: "fake-openai-key",
};

function composition() {
  const tts = new FakeTTS();
  const selections: SpeechOutputConfig[] = [];
  const contexts: (SpeechOutputContext | undefined)[] = [];
  let quiet = 0;
  const options: RuntimeOptions = { speechOutputFactory: {
    create: (config = DEFAULT_LOCAL_OUTPUT, context) => {
      selections.push(config);
      contexts.push(context);
      return tts;
    },
    quiesce: async (config) => { quiet++; assert.deepEqual(config, selections[0]); },
  } };
  return { tts, selections, contexts, options, quiet: () => quiet };
}

test("explicit Mercury output uses injected composition and snapshots selection through quiescence", async () => {
  const c = composition();
  const stt = new FakeSTT();
  const signal = new AbortController().signal;
  const context = { signal, getKey: () => "fake-output-key", beforeRequest: () => {} };
  c.options.speechOutputContext = context;
  const speechOutput = { ...DEFAULT_LOCAL_OUTPUT, provider: { ...DEFAULT_LOCAL_OUTPUT.provider } };
  const runtime = await createConversationRuntime({ llmProvider: "mercury", speechOutput }, env, stt, c.options);
  speechOutput.voice = "mutation";
  assert.equal(runtime.stt, stt);
  assert.ok(runtime.responder instanceof CascadeResponder);
  assert.match(runtime.label, /local piper-kristin-medium · kristin/);
  assert.equal(c.selections[0]?.voice, "kristin");
  assert.equal(c.contexts[0], context);
  const session = await runtime.responder.open();
  await session.close();
  assert.equal(c.tts.session.closed, 1);
  await runtime.quiesce?.();
  assert.equal(c.quiet(), 1);
});

test("room and tryout reuse one real local factory with fake synthesis", async () => {
  let loads = 0;
  const local = new LocalSpeechOutputFactory({ createSynth: async () => {
    loads++;
    return { numSpeakers: 1, sampleRate: 22_050, synth: async () => {
      throw new Error("runtime construction/open must not synthesize");
    } };
  } });
  const factory = new SpeechOutputFactory({ local });
  const trial = factory.create(DEFAULT_LOCAL_OUTPUT);
  const runtime = await createConversationRuntime({ llmProvider: "mercury", speechOutput: DEFAULT_LOCAL_OUTPUT },
    env, new FakeSTT(), { speechOutputFactory: factory });
  assert.equal(loads, 0);
  const sample = await trial.open();
  const room = await runtime.responder.open();
  assert.equal(loads, 1);
  await room.close();
  await runtime.quiesce?.();
  await sample.close();
  await factory.close();
});

test("runtime retirement remains pending after responder close until the factory is quiet", async () => {
  const c = composition();
  let release!: () => void;
  const native = new Promise<void>((resolve) => { release = resolve; });
  c.options.speechOutputFactory!.quiesce = () => native;
  const runtime = await createConversationRuntime({ llmProvider: "mercury", speechOutput: DEFAULT_LOCAL_OUTPUT },
    env, new FakeSTT(), c.options);
  const session = await runtime.responder.open();
  await session.close();
  let quiet = false;
  const retirement = runtime.quiesce!().then(() => { quiet = true; });
  await setImmediate();
  assert.equal(quiet, false);
  release();
  await retirement;
});

test("hosted Mercury selection reaches the same factory without legacy voice conversion", async () => {
  const c = composition();
  const speechOutput: SpeechOutputConfig = {
    schema: 1, provider: { kind: "openai" }, model: "gpt-realtime-2.1-mini", voice: "marin", speed: 0.8,
  };
  const runtime = await createConversationRuntime({ llmProvider: "mercury", speechOutput, voice: "legacy" },
    env, new FakeSTT(), c.options);
  assert.deepEqual(c.selections, [speechOutput]);
  assert.match(runtime.label, /openai gpt-realtime-2.1-mini · marin/);
});

test("legacy Mercury and default Cerebras keep their prior defaults and factory stays unused", async () => {
  const c = composition();
  const legacy = await createConversationRuntime({ llmProvider: "mercury" }, env, new FakeSTT(), c.options);
  assert.match(legacy.label, /Mercury .*kokoro-v0.19/);
  assert.equal(legacy.quiesce, undefined);
  const defaultRuntime = await createConversationRuntime({}, env, new FakeSTT(), c.options);
  assert.match(defaultRuntime.label, /Cerebras .*kokoro-v0.19/);
  const cerebras = await createConversationRuntime({ speechOutput: DEFAULT_LOCAL_OUTPUT }, env, new FakeSTT(), c.options);
  assert.match(cerebras.label, /Cerebras .*kokoro-v0.19/);
  assert.deepEqual(c.selections, []);
});

test("Realtime ignores cascade output and retains its existing voice validation", async () => {
  const c = composition();
  const runtime = await createConversationRuntime({ responder: "realtime", llmProvider: "mercury",
    voice: "cedar", speechOutput: DEFAULT_LOCAL_OUTPUT }, env, new FakeSTT(), c.options);
  assert.ok(runtime.responder instanceof OpenAIRealtimeResponder);
  assert.match(runtime.label, /OpenAI Realtime .*cedar/);
  assert.equal(runtime.quiesce, undefined);
  assert.deepEqual(c.selections, []);
  await assert.rejects(createConversationRuntime({ responder: "realtime", voice: "invalid" }, env, new FakeSTT()),
    /Unsupported Realtime voice/);
});

test("legacy Cartesia selection and required LLM credentials retain their existing behavior", async () => {
  const runtime = await createConversationRuntime({ llmProvider: "mercury", ttsEngine: "cartesia", voice: "katie" },
    env, new FakeSTT());
  assert.match(runtime.label, /Cartesia sonic-3.5 · katie/);
  await assert.rejects(createConversationRuntime({ llmProvider: "mercury", speechOutput: DEFAULT_LOCAL_OUTPUT },
    {}, new FakeSTT()), /INCEPTIONLABS_API_KEY is required/);
});
