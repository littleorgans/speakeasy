import { loadEnvFile } from "node:process";
import type { ResponderKind, RuntimeConfig, TtsEngine } from "@speakeasy/convo-engine";
import { startBrowserVoiceServer } from "./host.ts";

loadLocalEnv();
const options = readOptions(process.argv.slice(2), process.env);
const host = await startBrowserVoiceServer(options);
console.log(`speak-easy browser voice: ${host.url}`);
console.log("Open the URL, allow microphone access, then start a room.");

function readOptions(argv: string[], env: NodeJS.ProcessEnv): {
  port: number;
  runtimeConfig: RuntimeConfig;
} {
  const portFlag = argv.indexOf("--port");
  const port = portFlag === -1 ? 4317 : Number(argv[portFlag + 1]);
  if (!Number.isInteger(port) || port < 0 || port > 65_535) {
    throw new Error("--port must be an integer from 0 to 65535");
  }
  return {
    port,
    runtimeConfig: {
      responder:
        readChoice(env.SPEAKEASY_RESPONDER, ["cascade", "realtime"]) ??
        "realtime",
      ttsEngine: readChoice(env.SPEAKEASY_TTS, ["sherpa", "cartesia"]),
      llmModel: env.SPEAKEASY_MODEL,
      ttsModel: env.SPEAKEASY_TTS_MODEL,
      voice: env.SPEAKEASY_VOICE,
    },
  };
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

function readChoice<T extends ResponderKind | TtsEngine>(
  value: string | undefined,
  choices: readonly T[],
): T | undefined {
  if (value === undefined) return undefined;
  if (choices.includes(value as T)) return value as T;
  throw new Error(`${value} must be one of: ${choices.join(", ")}`);
}
