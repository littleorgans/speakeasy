import type { EndpointConfig } from "../contract.ts";

/** Output formatting helpers shared by the bench harness. */

export function formatEndpoint(endpoint: EndpointConfig | undefined): string {
  if (!endpoint) {
    return "default";
  }
  return `${endpoint.mode ?? "eager"}:trail=${endpoint.minTrailingSilenceMs ?? "default"}ms:minutt=${endpoint.minUtteranceMs ?? "default"}ms`;
}

export function formatMs(value: number): string {
  return `${value.toFixed(1)}ms`;
}

export function formatOptionalMs(value: number | undefined): string {
  return value === undefined ? "n/a" : formatMs(value);
}

export function formatBoolean(value: boolean): "y" | "n" {
  return value ? "y" : "n";
}
