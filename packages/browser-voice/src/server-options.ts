import type { RuntimeConfig, TtsEngine } from "@speakeasy/convo-engine";

export function readBrowserVoiceOptions(
  argv: string[],
  env: NodeJS.ProcessEnv,
): { port: number; runtimeConfig: RuntimeConfig } {
  const portFlag = argv.indexOf("--port");
  const port = portFlag === -1 ? 4317 : Number(argv[portFlag + 1]);
  if (!Number.isInteger(port) || port < 0 || port > 65_535) {
    throw new Error("--port must be an integer from 0 to 65535");
  }
  return {
    port,
    runtimeConfig: {
      ttsEngine: readChoice(env.SPEAKEASY_TTS, ["sherpa", "cartesia"]),
      llmModel: env.SPEAKEASY_MODEL,
      ttsModel: env.SPEAKEASY_TTS_MODEL,
      voice: env.SPEAKEASY_TTS_VOICE,
    },
  };
}

function readChoice<T extends TtsEngine>(
  value: string | undefined,
  choices: readonly T[],
): T | undefined {
  if (value === undefined) return undefined;
  const choice = choices.find((candidate) => candidate === value);
  if (choice !== undefined) return choice;
  throw new Error(`${value} must be one of: ${choices.join(", ")}`);
}
