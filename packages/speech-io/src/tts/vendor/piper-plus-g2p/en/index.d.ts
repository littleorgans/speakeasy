/** The EnglishG2P symbols speak-easy calls; see ../NOTICE. */
export class EnglishG2P {
  /** `cmuDict`: lowercase word -> ARPAbet string, used instead of the bundled file. */
  constructor(options?: { cmuDict?: Record<string, string> | Map<string, string> });
  /** IPA tokens for `text`; word boundaries are " " tokens. */
  phonemize(text: string): { tokens: string[]; prosody: null[] };
}

/** Spell numbers out the way g2p_en does ("1984" -> "nineteen eighty-four"). */
export function normalizeNumbers(text: string): string;
