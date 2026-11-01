import {
  ProviderError,
  parseSpeechInputConfig,
  parseSpeechOutputConfig,
  safeProviderError,
  sameSpeechInputConfig,
  sameSpeechOutputConfig,
  sameSpeechOutputModel,
} from "@speakeasy/speech-io/providers";
import type {
  SpeechCatalogue,
  SpeechConfigRegistries,
  SpeechDirection,
  SpeechInputConfig,
  SpeechOutputConfig,
} from "@speakeasy/speech-io/providers";
import type { ProviderCredentials } from "./provider-availability.ts";

/**
 * Host supplied workflow metadata, never decoded from a browser source flag.
 * Live routes assign their operation; saved routes first use PromptReader.takePath.
 * The session owns live operation lifetime.
 * This describes the application path, without attesting physical microphone origin.
 */
export type ProviderSource =
  | { kind: "live-capture"; sessionId: string; operationId: string }
  | { kind: "live-response-text"; sessionId: string; operationId: string }
  | { kind: "saved-take"; corpusId: string; takeId: string }
  | { kind: "typed-text"; sessionId: string }
  | { kind: "catalogue-sample"; sampleId: string }
  | { kind: "unclassified-upload" };

/** Only trusted startup injection can supply this; there is no client grant parser. */
export type VerifiedOwnerReplayGrant = {
  grantId: string;
  ownerDecisionRef: string;
  serverRunId: string;
  scope: {
    corpusId: string;
    takeIds: string[];
    inputConfigs: SpeechInputConfig[];
    outputConfigs: SpeechOutputConfig[];
  };
};

type DirectedConfig =
  | { direction: "input"; config: SpeechInputConfig }
  | { direction: "output"; config: SpeechOutputConfig };

export type ProviderSelection = DirectedConfig & {
  sessionId: string;
  catalogueRevision: string;
  selectionRevision: number;
};

export type ProviderSelectionRequest = {
  sessionId: string;
  direction: SpeechDirection;
  catalogueRevision: string;
  config: unknown;
};

export type ProviderDispatchRequest = ProviderSelectionRequest & {
  selectionRevision?: number;
  source: ProviderSource;
};

export type ProviderAuthorization = DirectedConfig & {
  sessionId: string;
  catalogueRevision: string;
  selectionRevision: number | null;
  source: ProviderSource;
  grantId: string | null;
};

export type ProviderDispatchContext = {
  authorization: ProviderAuthorization;
  /** Call before deferred work and each later transmission. Throws after revocation. */
  recheck: () => ProviderAuthorization;
  /** API only; also rechecks policy and current effective environment at call time. */
  apiKey: (() => string) | undefined;
};

export type ProviderPolicyOptions = {
  serverRunId: string;
  catalogue: () => SpeechCatalogue;
  registries: () => SpeechConfigRegistries;
  credentials: ProviderCredentials;
  replayGrants?: readonly VerifiedOwnerReplayGrant[];
};

type ResolvedConfig = DirectedConfig & { catalogueRevision: string };

/** One policy per server run. Protected routes alone call select with a server session. */
export class ProviderPolicy {
  private readonly options: ProviderPolicyOptions;
  private readonly grants: readonly VerifiedOwnerReplayGrant[];
  private readonly selections = new Map<string, Map<SpeechDirection, ProviderSelection>>();
  private nextRevision = 1;

  constructor(options: ProviderPolicyOptions) {
    this.options = { ...options };
    this.grants = structuredClone(options.replayGrants ?? []);
  }

  /** Selection and metadata never execute an adapter or read speech bytes. */
  select(request: ProviderSelectionRequest): ProviderSelection {
    if (!nonempty(request.sessionId)) throw new ProviderError("SELECTION_REQUIRED");
    const resolved = this.resolve(request);
    const selection: ProviderSelection = {
      ...resolved,
      sessionId: request.sessionId,
      selectionRevision: this.nextRevision++,
    };
    let session = this.selections.get(request.sessionId);
    if (!session) {
      session = new Map();
      this.selections.set(request.sessionId, session);
    }
    session.set(request.direction, selection);
    return structuredClone(selection);
  }

  revoke(sessionId: string, direction?: SpeechDirection): void {
    if (direction === undefined) this.selections.delete(sessionId);
    else {
      const session = this.selections.get(sessionId);
      session?.delete(direction);
      if (session?.size === 0) this.selections.delete(sessionId);
    }
  }

  /** Run before loading a saved WAV or constructing a request body. */
  authorize(request: ProviderDispatchRequest): ProviderAuthorization {
    const resolved = this.resolve(request);
    const source = structuredClone(request.source);
    if (resolved.config.provider.kind === "local") {
      return { ...resolved, sessionId: request.sessionId, selectionRevision: null,
        source, grantId: null };
    }
    this.checkSource(request.sessionId, resolved.direction, source);
    const selection = this.selections.get(request.sessionId)?.get(request.direction);
    if (!selection) throw new ProviderError("SELECTION_REQUIRED");
    if (selection.catalogueRevision !== resolved.catalogueRevision ||
        selection.selectionRevision !== request.selectionRevision ||
        !sameDirectedConfig(selection, resolved)) {
      throw new ProviderError("SELECTION_STALE");
    }
    const grantId = source.kind === "saved-take" ? this.savedGrant(source, resolved) : null;
    return { ...resolved, sessionId: request.sessionId,
      selectionRevision: selection.selectionRevision, source, grantId };
  }

  /**
   * The only execution helper: one invocation, no fanout, retry or fallback.
   * The callback must recheck before each deferred transmission, including streams.
   */
  async dispatch<T>(
    request: ProviderDispatchRequest,
    invoke: (context: ProviderDispatchContext) => T | Promise<T>,
  ): Promise<T> {
    try {
      const snapshot = structuredClone(request);
      const authorization = this.authorize(snapshot);
      const recheck = () => this.authorize(snapshot);
      const kind = authorization.config.provider.kind;
      const apiKey = kind === "local" ? undefined : () => {
        recheck();
        return this.options.credentials.getter(kind)();
      };
      return await invoke({ authorization, recheck, apiKey });
    } catch (error) {
      throw safeProviderError(error);
    }
  }

  private resolve(request: ProviderSelectionRequest): ResolvedConfig {
    try {
      const catalogue = this.options.catalogue();
      if (catalogue.revision !== request.catalogueRevision) {
        throw new ProviderError("SELECTION_STALE");
      }
      const registries = this.options.registries();
      let resolved: ResolvedConfig;
      let selectable: boolean | undefined;
      if (request.direction === "input") {
        const config = parseSpeechInputConfig(request.config, registries.input);
        resolved = { direction: "input", config, catalogueRevision: catalogue.revision };
        selectable = catalogue.input.find((entry) => sameSpeechInputConfig(entry.config, config))?.selectable;
      } else if (request.direction === "output") {
        const config = parseSpeechOutputConfig(request.config, registries.output);
        resolved = { direction: "output", config, catalogueRevision: catalogue.revision };
        // Output entries describe a model. Its registry validates voices and speed.
        selectable = catalogue.output.find((entry) => sameSpeechOutputModel(entry.config, config))?.selectable;
      } else {
        throw new ProviderError("CONFIG_UNSUPPORTED");
      }
      if (selectable !== true) throw new ProviderError("PROVIDER_UNAVAILABLE");
      const kind = resolved.config.provider.kind;
      if (kind !== "local" && !this.options.credentials.configured(kind)) {
        throw new ProviderError("PROVIDER_UNAVAILABLE");
      }
      return resolved;
    } catch (error) {
      throw safeProviderError(error);
    }
  }

  private checkSource(sessionId: string, direction: SpeechDirection, source: ProviderSource): void {
    if (!source || typeof source !== "object") throw new ProviderError("SOURCE_UNVERIFIED");
    switch (source.kind) {
      case "live-capture":
        if (direction === "input" && source.sessionId === sessionId && nonempty(source.operationId)) return;
        break;
      case "live-response-text":
        if (direction === "output" && source.sessionId === sessionId && nonempty(source.operationId)) return;
        break;
      case "typed-text":
        if (direction === "output" && source.sessionId === sessionId) return;
        break;
      case "saved-take":
        if (nonempty(source.corpusId) && nonempty(source.takeId)) return;
        break;
    }
    throw new ProviderError("SOURCE_UNVERIFIED");
  }

  private savedGrant(source: Extract<ProviderSource, { kind: "saved-take" }>, resolved: ResolvedConfig): string {
    const grant = this.grants.find((candidate) =>
      nonempty(candidate.grantId) && nonempty(candidate.ownerDecisionRef) &&
      nonempty(this.options.serverRunId) && candidate.serverRunId === this.options.serverRunId &&
      candidate.scope.corpusId === source.corpusId && candidate.scope.takeIds.includes(source.takeId) &&
      (resolved.direction === "input"
        ? candidate.scope.inputConfigs.some((config) => sameSpeechInputConfig(config, resolved.config))
        : candidate.scope.outputConfigs.some((config) => sameSpeechOutputConfig(config, resolved.config))));
    if (!grant) throw new ProviderError("SAVED_REPLAY_NOT_AUTHORIZED");
    return grant.grantId;
  }
}

function sameDirectedConfig(a: DirectedConfig, b: DirectedConfig): boolean {
  if (a.direction === "input" && b.direction === "input") return sameSpeechInputConfig(a.config, b.config);
  if (a.direction === "output" && b.direction === "output") return sameSpeechOutputConfig(a.config, b.config);
  return false;
}

function nonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
