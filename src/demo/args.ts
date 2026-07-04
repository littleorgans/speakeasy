import type { NumbersMode } from "../contract.ts";
import { DEFAULT_CORPUS_DIR } from "../corpus/store.ts";
import {
  DEFAULT_SHERPA_MODEL,
  parseSherpaModelId,
  SHERPA_MODELS,
  type SherpaModelId,
} from "../engines/sherpa-models.ts";
import { parseNumbersMode } from "../rewrite/numbers.ts";

/** CLI parsing and option shape for the push-to-talk demo. */

const SCRIPT_EVENT_PATTERN = /^(start|release)@(\d+(?:\.\d+)?)ms?$/;

const USAGE =
  'usage: pnpm demo [--model <id>] [--no-rewrite] [--numbers digits|words|off] [--wav <path>] [--script "start@0ms,release@2200ms,..."] [--device <index>] [--list-devices] [--save <dir>] [--no-save] [--save-all]\n' +
  `  --model <id>  sherpa model to load (default ${DEFAULT_SHERPA_MODEL}): ${Object.keys(SHERPA_MODELS).join(", ")}\n` +
  "  --no-rewrite  raw engine output (default wraps with domain-rule rewrite)\n" +
  "  --numbers     number rendering: digits (ten->10), words (10->ten), off (default)\n" +
  `  corpus collection is ON by default (dir: ${DEFAULT_CORPUS_DIR}/): after each final, s = save wav+json pair and label it, any other key = discard\n` +
  "  --save <dir>  override the corpus directory\n" +
  "  --no-save     disarm corpus collection\n" +
  "  --save-all    save every utterance without prompting (expected=null; label the sidecars by hand)";

export type ScriptAction = "start" | "release";
export type ScriptEvent = { action: ScriptAction; atMs: number };

export type DemoOptions = {
  model?: SherpaModelId;
  /** Skip the rewrite decorator entirely (raw engine output). */
  noRewrite?: boolean;
  /** Number rendering for the rewrite pipeline. Default "off". */
  numbers?: NumbersMode;
  wavPath?: string;
  script?: ScriptEvent[];
  device?: string;
  listDevices?: boolean;
  saveDir?: string;
  noSave?: boolean;
  saveAll?: boolean;
};

export function parseArgs(argv: string[]): DemoOptions {
  const options: DemoOptions = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    switch (arg) {
      case "--model":
        options.model = parseSherpaModelId(expectValue(argv, (index += 1), arg));
        break;
      case "--no-rewrite":
        options.noRewrite = true;
        break;
      case "--numbers":
        options.numbers = parseNumbersMode(expectValue(argv, (index += 1), arg));
        break;
      case "--wav":
        options.wavPath = expectValue(argv, (index += 1), arg);
        break;
      case "--script":
        options.script = parseScript(expectValue(argv, (index += 1), arg));
        break;
      case "--device": {
        const value = expectValue(argv, (index += 1), arg);
        options.device = /^\d+$/.test(value) ? `:${value}` : value;
        break;
      }
      case "--list-devices":
        options.listDevices = true;
        break;
      case "--save":
        options.saveDir = expectValue(argv, (index += 1), arg);
        break;
      case "--no-save":
        options.noSave = true;
        break;
      case "--save-all":
        options.saveAll = true;
        break;
      default:
        throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }
  return options;
}

/**
 * Corpus collection is armed by default. The keep/label prompts need an
 * interactive TTY, so unattended runs (--script or piped stdin) quietly
 * disarm the default; an EXPLICIT --save there is an error unless --save-all
 * removes the prompting.
 */
export function resolveSavePlan(options: DemoOptions): void {
  if (options.noSave) {
    if (options.saveDir !== undefined || options.saveAll) {
      throw new Error(`--no-save conflicts with --save/--save-all\n${USAGE}`);
    }
    return;
  }
  if (options.saveAll) {
    options.saveDir ??= DEFAULT_CORPUS_DIR;
    return;
  }
  const unattended = Boolean(options.script) || !process.stdin.isTTY;
  if (unattended) {
    if (options.saveDir !== undefined) {
      throw new Error(
        "--save prompts for keep/label after each utterance; unattended runs (--script or piped stdin) need --save-all",
      );
    }
    console.log(
      "corpus saving disarmed (unattended run); pass --save-all to keep every utterance",
    );
    return;
  }
  options.saveDir ??= DEFAULT_CORPUS_DIR;
}

function expectValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (value === undefined) {
    throw new Error(`${flag} requires a value\n${USAGE}`);
  }
  return value;
}

function parseScript(spec: string): ScriptEvent[] {
  const events = spec.split(",").map((part): ScriptEvent => {
    const match = SCRIPT_EVENT_PATTERN.exec(part.trim());
    if (!match) {
      throw new Error(
        `Invalid script event ${JSON.stringify(part)}; expected <start|release>@<ms>, e.g. start@0ms`,
      );
    }
    return { action: match[1] as ScriptAction, atMs: Number(match[2]) };
  });
  return events.sort((left, right) => left.atMs - right.atMs);
}
