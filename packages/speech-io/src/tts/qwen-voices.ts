import type { VoiceSetting } from "../settings.ts";

/**
 * Qwen3 TTS speakers, voice copy references and the controls each model
 * declares. Offered values are the ones the taming runs tried plus stock; the
 * first default is today's request, so a reading with nothing touched is
 * byte identical to before. Evidence: the thread's taming and voice copy
 * reports, and `speech info --json` on both CustomVoice GGUFs.
 */

/** Ryan and Aiden keep speaker ids 0 and 1; `languages` is each speaker's native one. */
export const QWEN_SPEAKERS = [
  { id: "ryan", label: "Ryan", languages: ["en"], speakerId: 0 },
  { id: "aiden", label: "Aiden", languages: ["en"], speakerId: 1 },
  { id: "dylan", label: "Dylan", languages: ["zh"], speakerId: 2 },
  { id: "eric", label: "Eric", languages: ["zh"], speakerId: 3 },
  { id: "ono_anna", label: "Ono Anna", languages: ["ja"], speakerId: 4 },
  { id: "serena", label: "Serena", languages: ["zh"], speakerId: 5 },
  { id: "sohee", label: "Sohee", languages: ["ko"], speakerId: 6 },
  { id: "uncle_fu", label: "Uncle Fu", languages: ["zh"], speakerId: 7 },
  { id: "vivian", label: "Vivian", languages: ["zh"], speakerId: 8 },
] as const;

/** The voice whose reference audio the owner records on the page; the host holds it in memory only. */
export const RECORDING_VOICE = "recording";

/**
 * Synthetic Aiden references, pinned in each voice copy payload with their
 * exact transcripts, then the owner's recording.
 */
export const QWEN_COPY_VOICES = [
  { id: "ref-3s", label: "Synthetic Aiden, 4 s", languages: ["en"], speakerId: 0, origin: "synthetic",
    file: "references/ref-3s.wav", transcript: "This is a calm and steady synthetic reference voice." },
  { id: "ref-10s", label: "Synthetic Aiden, 12 s", languages: ["en"], speakerId: 1, origin: "synthetic",
    file: "references/ref-10s.wav",
    transcript: "This is a calm and steady synthetic reference voice. The window is open, and the room is quiet. " +
      "We can take our time today, and read each sentence slowly and clearly." },
  { id: "ref-20s", label: "Synthetic Aiden, 25 s", languages: ["en"], speakerId: 2, origin: "synthetic",
    file: "references/ref-20s.wav",
    transcript: "This is a calm and steady synthetic reference voice. The window is open, and the room is quiet. " +
      "We can take our time today, and read each sentence slowly and clearly. I will read these sentences clearly " +
      "and steadily. Every word is part of a synthetic recording made for a local speech experiment. The next part " +
      "will use this same voice to say something new." },
  { id: RECORDING_VOICE, label: "My recording", languages: ["en"], speakerId: 3, origin: "recording" },
] as const;

const LANGUAGES = [
  ["en", "English"], ["auto", "Auto"], ["zh", "Chinese"], ["ja", "Japanese"], ["ko", "Korean"], ["de", "German"],
  ["fr", "French"], ["ru", "Russian"], ["pt", "Portuguese"], ["es", "Spanish"], ["it", "Italian"],
] as const;

const stock = (value: number) => ({ value, label: `${value} (stock)` });
const values = (...list: number[]) => list.map((value) => ({ value, label: String(value) }));
const sorted = <T extends { value: number }>(options: T[]) => options.sort((a, b) => a.value - b.value);
const onOff = (stockValue?: boolean) => [false, true].map((value) =>
  ({ value, label: `${value ? "On" : "Off"}${value === stockValue ? " (stock)" : ""}` }));

/** Copy runs only tried stock sampling on official PyTorch. */
const COPY_TRIED = " Voice copy: only stock was tried here.";

function qwenSettings({ instructions, copy }: { instructions: boolean; copy: boolean }): readonly VoiceSetting[] {
  const tried = copy ? COPY_TRIED : "";
  return [
    ...(instructions ? [{ id: "instructions", label: "Instruction", kind: "text", default: "", maxLength: 500,
      note: "Tells the voice how to speak, in words. Every instruction tried changed the audio; whether it obeys is for your ear." }] as const : []),
    ...(copy ? [{ id: "copyMode", label: "Copy", kind: "choice", default: "words-and-voice",
      options: [{ value: "words-and-voice", label: "Words and voice" }, { value: "voice-only", label: "Voice only" }],
      note: "Words and voice uses your transcript and copies more of the voice; voice only needs no transcript and copies less." }] as const : []),
    { id: "language", label: "Language", kind: "choice", default: "en",
      options: LANGUAGES.map(([value, label]) => ({ value, label })),
      note: "The language of the text. Each speaker sounds best in its own; only English was tried here." },
    { id: "seed", label: "Seed", kind: "integer", default: 1, min: 0, max: Number.MAX_SAFE_INTEGER,
      note: "Same seed and settings give the same audio; another seed gives another take." },
    // D1, owner 2026-11-03: Off for Qwen, a main row control so Read again can compare both ways.
    { id: "firstClause", label: "First clause split", kind: "choice", default: false, pipeline: true, options: onOff(),
      note: "On sends the opening words alone so sound starts sooner; Off keeps the first sentence whole." },
    // D8: voice copy caps each part at 20 s; 0 means no cap.
    { id: "maxSeconds", label: "Longest part", kind: "choice", default: copy ? 20 : 0,
      options: [{ value: 0, label: copy ? "Off" : "Off (stock)" }, ...[5, 10, 20].map((value) =>
        ({ value, label: `${value} s${copy && value === 20 ? " (stock)" : ""}` }))],
      note: "Stops each part at this length. It can cut words; it does not stop a laugh." },
    { id: "temperature", label: "Temperature", kind: "choice", default: 0.9, group: "sampling", disabledBy: "greedy",
      options: sorted([stock(0.9), ...values(0.5, 0.7, 1.1)]),
      note: `Lower picks likelier sounds. Every value tried changed the audio; none was always shorter.${tried}` },
    { id: "topP", label: "Top p", kind: "choice", default: 1, group: "sampling", disabledBy: "greedy",
      options: sorted([stock(1), ...values(0.8, 0.9)]),
      note: `Below 1 drops unlikely sounds. At 0.8 one Aiden sentence ran twice as long.${tried}` },
    { id: "topK", label: "Top k", kind: "choice", default: 50, group: "sampling", disabledBy: "greedy",
      options: sorted([stock(50), ...values(1, 10)]),
      note: `Fewer candidates per step. 10 changed one of four test takes; 1 changed all four.${tried}` },
    { id: "repetitionPenalty", label: "Repetition penalty", kind: "choice", default: 1.05, group: "sampling",
      options: sorted([stock(1.05), ...values(1, 1.15, 1.3)]),
      note: `Higher discourages repeats; 1.15 is part of the soft setting.${tried}` },
    { id: "greedy", label: "Greedy", kind: "choice", default: false, group: "sampling", options: onOff(false),
      note: "Always takes the likeliest sound: no randomness, and temperature, top p and top k step aside." +
        (copy ? " Both stacks greedy went degenerate in one outside voice copy report." : "") },
    { id: "codeTemperature", label: "Fine detail temperature", kind: "choice", default: 0.9, group: "sampling",
      disabledBy: "codeGreedy", options: sorted([stock(0.9), ...values(0.5)]),
      note: "The code predictor fills in 15 of the 16 sound codes per frame: texture more than words." },
    { id: "codeTopP", label: "Fine detail top p", kind: "choice", default: 1, group: "sampling", disabledBy: "codeGreedy",
      options: sorted([stock(1), ...values(0.8)]), note: "Below 1 drops unlikely fine detail codes." },
    { id: "codeTopK", label: "Fine detail top k", kind: "choice", default: 50, group: "sampling", disabledBy: "codeGreedy",
      options: sorted([stock(50), ...values(10)]), note: "Fewer fine detail candidates per step." },
    { id: "codeGreedy", label: "Fine detail greedy", kind: "choice", default: false, group: "sampling", options: onOff(false),
      note: "The likeliest fine detail code every time; its temperature, top p and top k step aside." },
  ];
}

export const QWEN_CUSTOM_SETTINGS = qwenSettings({ instructions: false, copy: false });
/** 1.7B lists the same defaults plus instructions (taming loads, `speech info`). */
export const QWEN_CUSTOM_17B_SETTINGS = qwenSettings({ instructions: true, copy: false });
export const QWEN_COPY_SETTINGS = qwenSettings({ instructions: false, copy: true });
