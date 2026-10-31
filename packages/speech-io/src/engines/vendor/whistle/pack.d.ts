/** speak-easy declarations for the vendored pack.js; see NOTICE. */

export type PackManifest = {
  melfb: { src: number };
  tok: unknown;
};

export type Pack = {
  man: PackManifest;
  buf: ArrayBuffer;
  base: number;
};

export function parsePack(buf: ArrayBuffer): Pack;
