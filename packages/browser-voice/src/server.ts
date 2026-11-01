import { loadEnvFile } from "node:process";
import { createConversationRuntime } from "@speakeasy/convo-engine";
import { SpeechOutputFactory } from "@speakeasy/speech-io";
import { startBrowserVoiceServer } from "./host.ts";
import { createSetupRuntime, readBrowserVoiceOptions } from "./server-options.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { createFileInputAdapters, createHostProviderConfig } from "./setup-routes.ts";
import { LocalSpeechOutputFactory } from "../../speech-io/src/tts/factory.ts";
import { getLocalOutputCatalogue, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";
import { buildSpeechOutputCatalogue } from "./speech-output-catalog.ts";

loadLocalEnv();
const options = readBrowserVoiceOptions(process.argv.slice(2), process.env);
const setup = await createSetupRuntime(options.setupRuntimeFile);
const credentials = createProviderCredentials();
const localOutput = new LocalSpeechOutputFactory({ root: options.outputRoot });
const output = new SpeechOutputFactory({ local: localOutput });
const outputCatalogue = buildSpeechOutputCatalogue({ local: await getLocalOutputCatalogue(options.outputRoot),
  localRegistry: LOCAL_OUTPUT_REGISTRY });
const inputAdapters = createFileInputAdapters();
const providers = createHostProviderConfig({ credentials, readiness: setup.readiness, inputAdapters,
  outputCatalogue: () => outputCatalogue });
const host = await startBrowserVoiceServer({ ...options,
  createRuntime: (config) => createConversationRuntime(config, process.env, undefined, { speechOutputFactory: output }),
  setupExecutor: setup.executor, setupReadiness: setup.readiness, providers,
  speechOutput: {
    catalogue: () => outputCatalogue,
    createOutput: (config, context) => {
      context.recheck();
      return output.create(config, { beforeRequest: context.recheck, getKey: context.apiKey,
        signal: context.signal, observe: context.observe });
    },
    quiesce: (config) => output.quiesce(config),
    close: () => output.close(),
  },
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
