export type Segment = { kind: "same" | "inserted" | "removed" | "replaced"; was: string[]; words: string[] };
export type LiveUpdate = { atMs: number; words: string };
export function diffWords(before: string, after: string): Segment[];
export function isCorrection(segments: Segment[]): boolean;
export function revisions(updates: LiveUpdate[]): (LiveUpdate & { segments: Segment[]; correction: boolean })[];
export function trailSummary(updates: LiveUpdate[]): string;
export function trailItems(document: { createElement(tag: string): any }, updates: LiveUpdate[], final: string | null,
  formatDuration: (ms: unknown) => string | null): any[];
