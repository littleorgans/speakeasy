export function formatDuration(ms: unknown): string | null;
export function formatBytes(bytes: unknown): string | null;
export function inputCard(setup: Record<string, any>): string;
export function outputCard(model: Record<string, any>): string;
export function outputLatency(context: { outputLatency?: number; baseLatency?: number } | null | undefined):
  { outputMs: number | null; baseMs: number | null } | null;
export function readEvents(response: { body: ReadableStream<Uint8Array> }, onEvent: (event: any) => void): Promise<void>;
export function liveLine(live: { setup: Record<string, any>; stream: Record<string, any>; settings: Record<string, number> } | null,
  final?: Record<string, any>): string;
