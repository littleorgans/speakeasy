import { parseTtsModelId, TTS_MODELS, type TtsModelId } from "./models.ts";
import { runTtsSweep, type SweepOptions } from "./sweep.ts";

/**
 * TTS sweep entry point. Zero-arg runs the full sweep over every registry
 * model; --model narrows to one, --speed adjusts the speaking rate.
 */

const USAGE = [
  "usage: node src/tts/run.ts [--model <id>] [--speed <rate>]",
  `       --model ids: ${Object.keys(TTS_MODELS).join(", ")}`,
].join("\n");

await runTtsSweep(parseArgs(process.argv.slice(2)));

function parseArgs(args: string[]): SweepOptions {
  const options: SweepOptions = {
    models: Object.keys(TTS_MODELS) as TtsModelId[],
    speed: 1,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--model") {
      options.models = [parseTtsModelId(requireValue(args, index))];
      index += 1;
    } else if (arg === "--speed") {
      options.speed = parseSpeed(requireValue(args, index));
      index += 1;
    } else {
      throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }

  return options;
}

function requireValue(args: string[], index: number): string {
  const value = args[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${args[index]}`);
  }
  return value;
}

function parseSpeed(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--speed must be a positive number, received ${value}`);
  }
  return parsed;
}
