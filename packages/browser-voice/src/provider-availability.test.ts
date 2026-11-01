import assert from "node:assert/strict";
import { test } from "node:test";
import { ProviderError } from "@speakeasy/speech-io/providers";
import {
  createProviderCredentials,
  PROVIDER_CREDENTIAL_NAMES,
} from "./provider-availability.ts";
import type { EffectiveEnvironment } from "./provider-availability.ts";
import type { ApiProviderKind } from "@speakeasy/speech-io/providers";

test("missing, empty and whitespace credentials are unconfigured for both directions", () => {
  for (const value of [undefined, "", " \t "]) {
    const environment = Object.fromEntries(Object.values(PROVIDER_CREDENTIAL_NAMES).map((name) => [name, value]));
    const credentials = createProviderCredentials(() => environment);
    assert.deepEqual(Object.values(credentials.snapshot()), Array(6).fill("unconfigured"));
    assert.throws(credentials.getter("cartesia"),
      (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
  }
});

test("one effective environment supplies every provider without serializing values", () => {
  for (const kind of Object.keys(PROVIDER_CREDENTIAL_NAMES) as ApiProviderKind[]) {
    const credentials = createProviderCredentials(() => ({ [PROVIDER_CREDENTIAL_NAMES[kind]]: "fixture-credential-value" }));
    const status = credentials.snapshot();
    assert.equal(status[kind], "configured");
    assert.equal(Object.values(status).filter((value) => value === "configured").length, 1);
    assert.equal(JSON.stringify(status).includes("fixture-credential-value"), false);
    assert.equal(credentials.getter(kind)(), "fixture-credential-value");
  }
});

test("construction and getter creation have no effect; lookup happens at each call", () => {
  let environment: EffectiveEnvironment = {};
  let reads = 0;
  const credentials = createProviderCredentials(() => { reads++; return environment; });
  const getKey = credentials.getter("openai");
  assert.equal(reads, 0);
  environment = { OPENAI_API_KEY: "first-fixture-value" };
  assert.equal(credentials.configured("openai"), true);
  assert.equal(getKey(), "first-fixture-value");
  environment = { OPENAI_API_KEY: "replacement-fixture-value" };
  assert.equal(getKey(), "replacement-fixture-value");
  environment = {};
  assert.equal(credentials.configured("openai"), false);
  assert.throws(getKey, (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
});

test("unresearched alias names cannot enable providers", () => {
  const credentials = createProviderCredentials(() => ({
    GOOGLE_API_KEY: "alias-fixture", ELEVEN_LABS_API_KEY: "alias-fixture",
  }));
  assert.equal(credentials.configured("gemini"), false);
  assert.equal(credentials.configured("elevenlabs"), false);
});

test("effective environment failures yield safe errors without raw diagnostics", () => {
  const credentials = createProviderCredentials(() => { throw new Error("sensitive environment failure"); });
  for (const run of [() => credentials.snapshot(), credentials.getter("cartesia")]) {
    assert.throws(run, (error: unknown) => error instanceof ProviderError &&
      error.code === "PROVIDER_UNAVAILABLE" && !error.message.includes("sensitive"));
  }
});
