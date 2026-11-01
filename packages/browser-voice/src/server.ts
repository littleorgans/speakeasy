import { loadEnvFile } from "node:process";
import { createConversationRuntime } from "@speakeasy/convo-engine";
import { WebSocket } from "ws";
import { startBrowserVoiceServer } from "./host.ts";
import { createSetupRuntime, readBrowserVoiceOptions } from "./server-options.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { createProviderRuntime } from "./provider-runtime.ts";
import { LocalSpeechOutputFactory } from "../../speech-io/src/tts/factory.ts";
import { getLocalOutputCatalogue } from "../../speech-io/src/tts/catalogue.ts";

loadLocalEnv();
const options = readBrowserVoiceOptions(process.argv.slice(2), process.env);
const setup = await createSetupRuntime(options.setupRuntimeFile);
const credentials = createProviderCredentials();
const localOutput = new LocalSpeechOutputFactory({ root: options.outputRoot });
const { output, providers, speechOutput } = createProviderRuntime({ credentials,
  readiness: setup.readiness, localOutput, localCatalogue: await getLocalOutputCatalogue(options.outputRoot),
  createSocket: (url, apiKey) => new WebSocket(url, { headers: { Authorization: `Bearer ${apiKey}` } }),
});
const host = await startBrowserVoiceServer({ ...options,
  createRuntime: (config, context) => createConversationRuntime(config, process.env, undefined,
    { speechOutputFactory: output, speechOutputContext: context?.speechOutputContext }),
  setupExecutor: setup.executor, setupReadiness: setup.readiness, providers,
  speechOutput,
});
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => { void host.close().catch(() => { process.exitCode = 1; }); });
}
console.log(`speak-easy browser voice: ${host.url}`);
console.log("Open the URL, allow microphone access, then start a room.");
if (options.promptReader) {
  console.log(`prompt reader: ${host.url}/prompt-reader (takes saved to ${options.promptReader.outputDir})`);
}

/** Load the gitignored local environment without printing or serializing it. */
function loadLocalEnv(): void {
  try {
    loadEnvFile(".env");
  } catch (error) {
    if (!isMissingFile(error)) throw error;
  }
}

function isMissingFile(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT"
  );
}
