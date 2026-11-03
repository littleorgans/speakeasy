import { join } from "node:path";
import type { AssetFile, AssetSpec, FileSetSpec } from "./assets.ts";
import type { SilenceGatePolicy } from "./offline-session.ts";

/**
 * The vocabulary every STT model registry shares (sherpa, Granite CTC,
 * Parakeet TDT, Whistle): descriptor commons, licence bases, pinned files,
 * asset resolution, download size and id lookup. Native free and working
 * directory free, so it sits in the vendored `./stt` closure.
 */

export type ModelArchive = {
  url: string;
  sha256: string;
  bytes: number;
};

export type ModelFile = AssetFile & { bytes: number };

/** One release tarball, extracted under the model root. */
export type ArchiveSource = { archive: ModelArchive };

/** Loose files, each pinned, under the model directory. */
export type FileSetSource = { files: readonly ModelFile[] };

/** Where a model downloads from: one release tarball, or pinned loose files. */
export type ModelSource = ArchiveSource | FileSetSource;

export type ModelLicence = {
  /** SPDX id, or a LicenseRef for unlisted terms. */
  id: string;
  terms: string;
  commercialUse: boolean;
  redistribution: boolean;
  /** Credit line a consumer must show, or null when none is required. */
  attribution: string | null;
  source: string;
};

export type ModelCommon<S extends ModelSource = ModelSource> = S & {
  /** Directory name under the model root; an archive's basename too. */
  name: string;
  /** Human note: architecture / training data / param scale. */
  note: string;
  licence: ModelLicence;
  /** Longest utterance a host should hold open for this model. */
  maxAudioMs: number;
};

export type OfflineCommon<S extends ModelSource = ModelSource> = ModelCommon<S> & {
  kind: "offline";
  /** Opt in to the silence gate (`offline-session.ts::SilenceGate`). */
  silenceGate?: SilenceGatePolicy;
};

export const APACHE_2_0 = {
  id: "Apache-2.0",
  terms:
    "Apache License 2.0: use, modify and redistribute; keep the licence and notices when redistributing.",
  commercialUse: true,
  redistribution: true,
  attribution: null,
} as const;

export const CC_BY_4_0 = {
  id: "CC-BY-4.0",
  terms:
    "Creative Commons Attribution 4.0: use, adapt and redistribute, including commercially, with attribution.",
  commercialUse: true,
  redistribution: true,
} as const;

/**
 * A file pinned at a revision `base`. `path` is its URL path under `base`
 * where that differs from its `name` in the model directory.
 */
export function pinned(
  base: string,
  name: string,
  bytes: number,
  sha256: string,
  path: string = name,
): ModelFile {
  return { name, url: `${base}/${path}`, sha256, bytes };
}

/**
 * Resolve a descriptor to its pinned asset under `root`; descriptor file names
 * join onto the returned `dir`.
 */
export function resolveModelPaths(model: ModelCommon<FileSetSource>, root: string): FileSetSpec;
export function resolveModelPaths(model: ModelCommon, root: string): AssetSpec | FileSetSpec;
export function resolveModelPaths(model: ModelCommon, root: string): AssetSpec | FileSetSpec {
  const dir = join(root, model.name);
  if ("files" in model) {
    return { dir, files: model.files };
  }
  return {
    url: model.archive.url,
    sha256: model.archive.sha256,
    archive: join(root, `${model.name}.tar.bz2`),
    dir,
  };
}

/** Total download size, the figure a consent surface shows. */
export function downloadBytes(model: ModelSource): number {
  return "files" in model
    ? model.files.reduce((total, file) => total + file.bytes, 0)
    : model.archive.bytes;
}

/** Whether `id` names an entry of `registry`; prototype keys never do. */
export function hasModelId<R extends Record<string, unknown>>(
  registry: R,
  id: string,
): id is Extract<keyof R, string> {
  return Object.hasOwn(registry, id);
}

/** The entry `id` names, or an error listing every id of the `kind` registry. */
export function lookupModel<R extends Record<string, unknown>>(
  registry: R,
  id: string,
  kind: string,
): R[Extract<keyof R, string>] {
  if (hasModelId(registry, id)) {
    return registry[id];
  }
  throw new Error(
    `Unknown ${kind} model "${id}"; available: ${Object.keys(registry).join(", ")}`,
  );
}
