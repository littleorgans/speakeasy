import assert from "node:assert/strict";
import { test } from "node:test";
import { ProviderError } from "@speakeasy/speech-io/providers";
import type {
  CatalogueEntry,
  ProviderErrorCode,
  SpeechCatalogue,
  SpeechConfigRegistries,
  SpeechDirection,
  SpeechInputConfig,
  SpeechOutputConfig,
} from "@speakeasy/speech-io/providers";
import { createProviderCredentials } from "./provider-availability.ts";
import { ProviderPolicy } from "./provider-policy.ts";
import type {
  ProviderDispatchContext,
  ProviderDispatchRequest,
  ProviderSelection,
  ProviderSource,
  VerifiedOwnerReplayGrant,
} from "./provider-policy.ts";

const localInput: SpeechInputConfig = { schema: 1, provider: { kind: "local", presetId: "test-preset" } };
const apiInput: SpeechInputConfig = { schema: 1, provider: { kind: "cartesia", model: "test-input" } };
const siblingInput: SpeechInputConfig = { schema: 1, provider: { kind: "openai", model: "test-input" } };
const secondModel: SpeechInputConfig = { schema: 1, provider: { kind: "cartesia", model: "other-input" } };
const localOutput: SpeechOutputConfig = {
  schema: 1, provider: { kind: "local" }, model: "test-local-output", voice: 0, speed: 1,
};
const apiOutput: SpeechOutputConfig = {
  schema: 1, provider: { kind: "cartesia" }, model: "test-output", voice: "voice-one", speed: 1,
};
const siblingOutput: SpeechOutputConfig = {
  schema: 1, provider: { kind: "openai" }, model: "test-output", voice: "voice-one", speed: 1,
};
const registeredInput: SpeechInputConfig = {
  schema: 1, provider: { kind: "local", runtimeKind: "mlx-parakeet", model: "registered-input" },
};
const siblingRegisteredInput: SpeechInputConfig = {
  schema: 1, provider: { kind: "local", runtimeKind: "speechcpp-qwen3-asr", model: "registered-input" },
};
const registeredOutput: SpeechOutputConfig = {
  ...localOutput, provider: { kind: "local", runtimeKind: "sherpa-vits" },
};
const siblingRegisteredOutput: SpeechOutputConfig = {
  ...localOutput, provider: { kind: "local", runtimeKind: "sherpa-kokoro" },
};
const live: ProviderSource = { kind: "live-capture", sessionId: "session-one", operationId: "capture-one" };
const response: ProviderSource = { kind: "live-response-text", sessionId: "session-one", operationId: "room-one" };
const typed: ProviderSource = { kind: "typed-text", sessionId: "session-one" };
const saved: ProviderSource = { kind: "saved-take", corpusId: "test-corpus", takeId: "take-one" };

function fixture(grants: VerifiedOwnerReplayGrant[] = []) {
  const environment: Record<string, string | undefined> = {
    CARTESIA_API_KEY: "fixture-cartesia", OPENAI_API_KEY: "fixture-openai",
  };
  const inputConfigs = [localInput, apiInput, siblingInput, secondModel, registeredInput, siblingRegisteredInput];
  const outputConfigs = [localOutput, apiOutput, siblingOutput, registeredOutput, siblingRegisteredOutput];
  function entry<C extends SpeechInputConfig | SpeechOutputConfig>(
    config: C, index: number, direction: SpeechDirection,
  ): CatalogueEntry<C> {
    return { id: String(index), label: "Test choice", config: structuredClone(config), selectable: true,
      transfer: config.provider.kind === "local" ? "stays-on-mac"
        : direction === "input" ? "audio-to-api" : "text-to-api-audio-returned", samples: [] };
  }
  const catalogue: SpeechCatalogue = {
    schema: 1, revision: "catalogue-one",
    input: inputConfigs.map((config, index) => entry(config, index, "input")),
    output: outputConfigs.map((config, index) => entry(config, index, "output")),
  };
  const registries: SpeechConfigRegistries = {
    input: {
      localPresets: [{ id: "test-preset" }],
      localModels: inputConfigs.flatMap(({ provider }) =>
        provider.kind === "local" && provider.runtimeKind !== undefined ? [provider] : []),
      apiModels: inputConfigs.flatMap((config) => config.provider.kind === "local" ? [] : [config.provider]),
    },
    output: outputConfigs.map((config) => ({
      ...config.provider, model: config.model,
      voices: [config.voice, "voice-two", 0, "0"], speed: { min: 0.5, max: 2 },
    })),
  };
  let catalogueReads = 0;
  let registryReads = 0;
  const policy = new ProviderPolicy({
    serverRunId: "run-one", credentials: createProviderCredentials(() => environment),
    catalogue: () => { catalogueReads++; return catalogue; },
    registries: () => { registryReads++; return registries; }, replayGrants: grants,
  });
  function select(direction: SpeechDirection, config: SpeechInputConfig | SpeechOutputConfig): ProviderSelection {
    return policy.select({ sessionId: "session-one", direction, config, catalogueRevision: catalogue.revision });
  }
  function request(selection: ProviderSelection, source: ProviderSource): ProviderDispatchRequest {
    return { ...selection, source };
  }
  return { policy, select, request, environment, catalogue, registries,
    reads: () => ({ catalogueReads, registryReads }), calls: 0 };
}

type Fixture = ReturnType<typeof fixture>;

async function denied(f: Fixture, request: ProviderDispatchRequest, code: ProviderErrorCode): Promise<void> {
  const before = f.calls;
  await assert.rejects(f.policy.dispatch(request, () => { f.calls++; }),
    (error: unknown) => error instanceof ProviderError && error.code === code);
  assert.equal(f.calls, before);
}

function grant(): VerifiedOwnerReplayGrant {
  return { grantId: "owner-grant-one", ownerDecisionRef: "explicit-owner-decision", serverRunId: "run-one",
    scope: { corpusId: "test-corpus", takeIds: ["take-one"], inputConfigs: [apiInput], outputConfigs: [apiOutput] } };
}

test("construction, catalogue reads, selection and authorization never invoke adapters", () => {
  const f = fixture();
  assert.deepEqual(f.reads(), { catalogueReads: 0, registryReads: 0 });
  const selection = f.select("input", apiInput);
  const authorization = f.policy.authorize(f.request(selection, live));
  assert.equal(authorization.selectionRevision, selection.selectionRevision);
  assert.equal(f.calls, 0);
});

test("local live input and saved replay remain available without API keys or selection", async () => {
  const f = fixture();
  delete f.environment.CARTESIA_API_KEY;
  delete f.environment.OPENAI_API_KEY;
  for (const source of [live, saved, { kind: "unclassified-upload" } as const]) {
    const result = await f.policy.dispatch({ sessionId: "session-one", direction: "input", config: localInput,
      catalogueRevision: f.catalogue.revision, source }, ({ authorization, apiKey }) => {
      assert.equal(apiKey, undefined);
      assert.equal(authorization.grantId, null);
      f.calls++;
      return "local result";
    });
    assert.equal(result, "local result");
  }
  await f.policy.dispatch({ sessionId: "session-one", direction: "output", config: localOutput,
    catalogueRevision: f.catalogue.revision, source: typed }, ({ apiKey }) => {
    assert.equal(apiKey, undefined);
    f.calls++;
  });
  assert.equal(f.calls, 4);
});

test("registered local input binds catalogue availability to its exact runtime and revision", async () => {
  const f = fixture();
  const selection = f.select("input", registeredInput);
  const request = f.request(selection, saved);
  await f.policy.dispatch(request, ({ authorization, apiKey }) => {
    assert.deepEqual(authorization.config, registeredInput);
    assert.equal(authorization.grantId, null);
    assert.equal(apiKey, undefined);
    f.calls++;
  });
  const entry = f.catalogue.input.find((item) => item.config.provider.kind === "local" &&
    "runtimeKind" in item.config.provider && item.config.provider.runtimeKind === "mlx-parakeet");
  assert.ok(entry);
  entry.selectable = false;
  await denied(f, request, "PROVIDER_UNAVAILABLE");
  entry.selectable = true;
  f.catalogue.input = f.catalogue.input.filter((item) => item !== entry);
  await denied(f, request, "PROVIDER_UNAVAILABLE");
  f.catalogue.input.push(entry);
  f.catalogue.revision = "registration-revised";
  await denied(f, request, "SELECTION_STALE");
  f.catalogue.revision = selection.catalogueRevision;
  f.registries.input.localModels = [];
  await denied(f, request, "CONFIG_UNSUPPORTED");
  await denied(f, f.request(f.select("input", apiInput), saved), "SAVED_REPLAY_NOT_AUTHORIZED");
});

test("registered local output availability cannot borrow a legacy or sibling runtime entry", async () => {
  const f = fixture();
  const config = { ...registeredOutput, voice: "voice-two", speed: 0.75 };
  const selection = f.select("output", config);
  const request = f.request(selection, typed);
  let recheck: (() => unknown) | undefined;
  await f.policy.dispatch(request, (context) => {
    assert.deepEqual(context.authorization.config, config);
    assert.equal(context.apiKey, undefined);
    recheck = context.recheck;
    f.calls++;
  });
  const entry = f.catalogue.output.find((item) => item.config.provider.kind === "local" &&
    "runtimeKind" in item.config.provider && item.config.provider.runtimeKind === "sherpa-vits");
  assert.ok(entry);
  entry.selectable = false;
  await denied(f, request, "PROVIDER_UNAVAILABLE");
  assert.ok(recheck);
  assert.throws(recheck, (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
  entry.selectable = true;
  f.catalogue.output = f.catalogue.output.filter((item) => item !== entry);
  await denied(f, request, "PROVIDER_UNAVAILABLE");
  f.catalogue.output.push(entry);
  f.catalogue.revision = "registration-revised";
  await denied(f, request, "SELECTION_STALE");
  f.catalogue.revision = selection.catalogueRevision;
  await denied(f, f.request(f.select("output", apiOutput), saved), "SAVED_REPLAY_NOT_AUTHORIZED");
});

test("restored preference and client selected flags grant no API permission", async () => {
  const f = fixture();
  const request: ProviderDispatchRequest = { sessionId: "session-one", direction: "input",
    config: apiInput, catalogueRevision: f.catalogue.revision, source: live };
  await denied(f, request, "SELECTION_REQUIRED");
  await denied(f, { ...request, config: { ...apiInput, selected: true } }, "CONFIG_UNSUPPORTED");
  // A guessed revision cannot create a server selection record.
  await denied(f, { ...request, selectionRevision: 1 }, "SELECTION_REQUIRED");
});

test("dispatch invokes exactly the selected provider with truthful transfer boundaries", async () => {
  const f = fixture();
  for (const [direction, config, source] of [
    ["input", apiInput, live], ["output", apiOutput, typed],
  ] as const) {
    const selection = f.select(direction, config);
    const result = await f.policy.dispatch(f.request(selection, source), ({ authorization, apiKey }) => {
      assert.deepEqual(authorization.config, config);
      assert.equal(authorization.direction, direction);
      assert.equal(apiKey?.(), "fixture-cartesia");
      f.calls++;
      return "one provider result";
    });
    assert.equal(result, "one provider result");
  }
  assert.equal(f.calls, 2);
});

test("room response text dispatches exactly the selected hosted output", async () => {
  const f = fixture();
  for (const output of [apiOutput, siblingOutput]) {
    const config = { ...output, voice: 0, speed: 0.75 };
    const before = f.calls;
    const selection = f.select("output", config);
    assert.equal(f.calls, before);
    const result: string = await f.policy.dispatch(f.request(selection, response), (context) => {
      assert.deepEqual(context.authorization, {
        ...selection, source: response, grantId: null,
      });
      assert.deepEqual(context.recheck(), context.authorization);
      assert.equal(context.apiKey?.(), output === apiOutput ? "fixture-cartesia" : "fixture-openai");
      f.calls++;
      return "room output";
    });
    assert.equal(result, "room output");
  }
  assert.equal(f.calls, 2);
});

test("room response text denies input, another session and blank operations before execution", async () => {
  const f = fixture();
  const input = f.select("input", apiInput);
  await denied(f, f.request(input, response), "SOURCE_UNVERIFIED");
  const output = f.select("output", apiOutput);
  for (const source of [
    { ...response, sessionId: "other-session" },
    { ...response, operationId: "" },
    { ...response, operationId: " \t\n" },
  ]) await denied(f, f.request(output, source), "SOURCE_UNVERIFIED");
  assert.equal(f.calls, 0);
});

test("room response text needs exact explicit selection and cannot authorize saved or unclassified text", async () => {
  const f = fixture();
  const config = { ...apiOutput, voice: 0, speed: 0.75 };
  const restored: ProviderDispatchRequest = { sessionId: "session-one", direction: "output", config,
    catalogueRevision: f.catalogue.revision, source: response };
  await denied(f, restored, "SELECTION_REQUIRED");
  await denied(f, { ...restored, selectionRevision: 1 }, "SELECTION_REQUIRED");
  const selection = f.select("output", config);
  const request = f.request(selection, response);
  for (const changed of [
    { ...config, provider: { kind: "openai" } },
    { ...config, voice: "0" },
    { ...config, voice: "voice-two" },
    { ...config, speed: 1 },
  ]) await denied(f, { ...request, config: changed }, "SELECTION_STALE");
  await denied(f, { ...request, selectionRevision: selection.selectionRevision + 1 }, "SELECTION_STALE");
  f.catalogue.revision = "catalogue-two";
  await denied(f, request, "SELECTION_STALE");
  f.catalogue.revision = selection.catalogueRevision;
  await denied(f, f.request(selection, saved), "SAVED_REPLAY_NOT_AUTHORIZED");
  await denied(f, f.request(selection, { kind: "unclassified-upload" }), "SOURCE_UNVERIFIED");
  f.policy.revoke("session-one", "output");
  await denied(f, request, "SELECTION_REQUIRED");
  assert.equal(f.calls, 0);
});

test("captured room context blocks later transmissions after key removal, revoke and reselection", async () => {
  const f = fixture();
  const selection = f.select("output", apiOutput);
  const request = f.request(selection, response);
  const context: ProviderDispatchContext = await f.policy.dispatch(request, (context) => context);
  assert.ok(context.apiKey);
  const getKey = context.apiKey;
  const transmit = () => {
    context.recheck();
    const key = getKey();
    f.calls++;
    return key;
  };
  const assertBlocked = (code: ProviderErrorCode) => {
    const before = f.calls;
    for (const handle of [context.recheck, getKey, transmit]) {
      assert.throws(handle, (error: unknown) => error instanceof ProviderError && error.code === code);
    }
    assert.equal(f.calls, before);
  };
  assert.equal(transmit(), "fixture-cartesia");
  request.source = { ...response, operationId: "replacement-room" };
  assert.deepEqual(context.recheck().source, response);
  delete f.environment.CARTESIA_API_KEY;
  assertBlocked("PROVIDER_UNAVAILABLE");
  f.environment.CARTESIA_API_KEY = "replacement-fixture";
  assert.equal(transmit(), "replacement-fixture");
  f.policy.revoke("session-one", "output");
  assertBlocked("SELECTION_REQUIRED");
  f.select("output", apiOutput);
  assertBlocked("SELECTION_STALE");
  assert.equal(f.calls, 2);
});

test("selection binds session, direction, provider, model and revision", async () => {
  const f = fixture();
  const selection = f.select("input", apiInput);
  const request = f.request(selection, live);
  await denied(f, { ...request, sessionId: "session-two", source: { ...live, sessionId: "session-two" } }, "SELECTION_REQUIRED");
  await denied(f, { ...request, direction: "output", config: apiOutput, source: typed }, "SELECTION_REQUIRED");
  await denied(f, { ...request, config: siblingInput }, "SELECTION_STALE");
  await denied(f, { ...request, config: secondModel }, "SELECTION_STALE");
  await denied(f, { ...request, selectionRevision: selection.selectionRevision + 1 }, "SELECTION_STALE");
  f.select("input", siblingInput);
  await denied(f, request, "SELECTION_STALE");
  f.select("input", apiInput);
  await denied(f, request, "SELECTION_STALE");
});

test("output selection binds typed voice and speed while allowing registry choices", async () => {
  const f = fixture();
  const config = { ...apiOutput, voice: 0, speed: 0.75 };
  const selection = f.select("output", config);
  const request = f.request(selection, typed);
  f.policy.authorize(request);
  for (const changed of [{ ...config, voice: "0" }, { ...config, voice: "voice-two" }, { ...config, speed: 1 }]) {
    await denied(f, { ...request, config: changed }, "SELECTION_STALE");
  }
});

test("catalogue revision changes, missing keys and deferred adapters deny before execution", async () => {
  const f = fixture();
  const selection = f.select("input", apiInput);
  const request = f.request(selection, live);
  f.catalogue.revision = "catalogue-two";
  await denied(f, request, "SELECTION_STALE");
  await denied(f, { ...request, catalogueRevision: "catalogue-two" }, "SELECTION_STALE");
  f.catalogue.revision = "catalogue-one";
  delete f.environment.CARTESIA_API_KEY;
  await denied(f, request, "PROVIDER_UNAVAILABLE");
  f.environment.CARTESIA_API_KEY = "fixture-cartesia";
  const entry = f.catalogue.input.find((item) => item.config.provider.kind === "cartesia");
  assert.ok(entry);
  entry.selectable = false;
  entry.reason = "adapter-deferred";
  await denied(f, request, "PROVIDER_UNAVAILABLE");
});

test("saved API replay denies by default in both directions despite explicit selection", async () => {
  const f = fixture();
  for (const [direction, config] of [["input", apiInput], ["output", apiOutput]] as const) {
    const selection = f.select(direction, config);
    const request = f.request(selection, saved);
    await denied(f, request, "SAVED_REPLAY_NOT_AUTHORIZED");
    // Browser approval fields are outside the policy request contract and ignored.
    await denied(f, { ...request, approved: true, grant: grant() } as ProviderDispatchRequest, "SAVED_REPLAY_NOT_AUTHORIZED");
  }
});

test("only a trusted grant plus exact explicit selection permits its named take and configs", async () => {
  const f = fixture([grant()]);
  for (const [direction, config] of [["input", apiInput], ["output", apiOutput]] as const) {
    const selection = f.select(direction, config);
    const result: string | null = await f.policy.dispatch(f.request(selection, saved), ({ authorization }) => {
      f.calls++;
      return authorization.grantId;
    });
    assert.equal(result, "owner-grant-one");
  }
  const selected = f.select("input", apiInput);
  await denied(f, { ...f.request(selected, saved), source: { ...saved, takeId: "take-two" } }, "SAVED_REPLAY_NOT_AUTHORIZED");
  await denied(f, { ...f.request(selected, saved), source: { ...saved, corpusId: "other-corpus" } }, "SAVED_REPLAY_NOT_AUTHORIZED");
  const sibling = f.select("input", siblingInput);
  await denied(f, f.request(sibling, saved), "SAVED_REPLAY_NOT_AUTHORIZED");
  const changedOutput = f.select("output", { ...apiOutput, voice: "voice-two" });
  await denied(f, f.request(changedOutput, saved), "SAVED_REPLAY_NOT_AUTHORIZED");
  f.policy.revoke("session-one");
  await denied(f, f.request(changedOutput, saved), "SELECTION_REQUIRED");
});

test("wrong run, missing owner evidence or wrong direction cannot broaden a grant", async () => {
  for (const change of [
    (g: VerifiedOwnerReplayGrant) => { g.serverRunId = "other-run"; },
    (g: VerifiedOwnerReplayGrant) => { g.ownerDecisionRef = ""; },
    (g: VerifiedOwnerReplayGrant) => { g.grantId = ""; },
    (g: VerifiedOwnerReplayGrant) => { g.scope.inputConfigs = []; },
  ]) {
    const scoped = grant();
    change(scoped);
    const f = fixture([scoped]);
    await denied(f, f.request(f.select("input", apiInput), saved), "SAVED_REPLAY_NOT_AUTHORIZED");
  }
});

test("unknown uploads, sample playback and sources from another workflow cannot enable API dispatch", async () => {
  const f = fixture([grant()]);
  const input = f.select("input", apiInput);
  for (const source of [{ kind: "unclassified-upload" }, { kind: "catalogue-sample", sampleId: "sample-one" },
    typed, { ...live, sessionId: "other-session" }, { ...live, operationId: "" },
    { kind: "live", selected: true }, null,
  ]) await denied(f, { ...f.request(input, live), source } as ProviderDispatchRequest, "SOURCE_UNVERIFIED");
  const output = f.select("output", apiOutput);
  await denied(f, f.request(output, live), "SOURCE_UNVERIFIED");
});

test("session selection and host grant snapshots cannot be broadened by caller mutation", async () => {
  const scoped = grant();
  const f = fixture([scoped]);
  scoped.scope.takeIds.push("take-two");
  const selection = f.select("input", apiInput);
  selection.config = siblingInput;
  await denied(f, f.request(selection, live), "SELECTION_STALE");
  const original = f.select("input", apiInput);
  await denied(f, f.request(original, { ...saved, takeId: "take-two" }), "SAVED_REPLAY_NOT_AUTHORIZED");
});

test("guarded call time getters deny removed credentials and revoked deferred requests", async () => {
  const f = fixture();
  const selection = f.select("output", apiOutput);
  const request = f.request(selection, typed);
  let transmit: (() => string) | undefined;
  await f.policy.dispatch(request, ({ apiKey }) => { transmit = apiKey; });
  assert.ok(transmit);
  assert.equal(transmit(), "fixture-cartesia");
  delete f.environment.CARTESIA_API_KEY;
  assert.throws(transmit, (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
  f.environment.CARTESIA_API_KEY = "replacement-fixture";
  assert.equal(transmit(), "replacement-fixture");
  f.policy.revoke("session-one", "output");
  assert.throws(transmit, (error: unknown) => error instanceof ProviderError && error.code === "SELECTION_REQUIRED");
});

test("queued dispatch snapshots config and source; changing selection stops its later transmissions", async () => {
  const f = fixture();
  const selection = f.select("input", apiInput);
  const request = f.request(selection, live);
  let later: (() => unknown) | undefined;
  await f.policy.dispatch(request, ({ authorization, recheck }) => {
    request.config = siblingInput;
    request.source = saved;
    assert.deepEqual(authorization.config, apiInput);
    assert.deepEqual(recheck().source, live);
    later = recheck;
  });
  f.select("input", siblingInput);
  assert.ok(later);
  assert.throws(later, (error: unknown) => error instanceof ProviderError && error.code === "SELECTION_STALE");
});

test("adapter failures are sanitized and never trigger retries or sibling calls", async () => {
  const f = fixture();
  const selection = f.select("input", apiInput);
  await assert.rejects(f.policy.dispatch(f.request(selection, live), () => {
    f.calls++;
    throw new Error("sensitive request headers and response body");
  }), (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_FAILED" &&
    !JSON.stringify(error).includes("sensitive"));
  assert.equal(f.calls, 1);
});
