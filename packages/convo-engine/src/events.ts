import type { TurnMetrics } from "./metrics.ts";
import type { ConvoState } from "./state.ts";

/**
 * Observable conversation output for every host, from the terminal demo to a
 * browser or desktop shell. Transcript events state whether their text replaces
 * the current draft or appends a streamed assistant delta.
 */
export type ConversationEvent =
  | { type: "state"; state: ConvoState }
  | {
      type: "transcript";
      role: "user" | "assistant";
      text: string;
      final: boolean;
      mode: "replace" | "append";
    }
  | { type: "metrics"; metrics: TurnMetrics }
  | { type: "interrupted" }
  | { type: "notice"; level: "info" | "error"; message: string };

export type ConversationObserver = (event: ConversationEvent) => void;
