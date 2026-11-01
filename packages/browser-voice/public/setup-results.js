/** Display jobs own no capture or saved audio. The page supplies the one take. */
export class SetupResults {
  constructor(page) {
    this.page = page;
    this.mode = "compare";
    this.catalog = null;
    this.rows = [];
    this.job = null;
    this.serial = 0;
    this.selected = new Set(["mlx-parakeet/v3-greedy"]);
    this.elements = Object.fromEntries([
      "input-provider", "input-model", "input-api", "input-transfer", "input-local",
      "input-presets", "input-selection", "compare-take", "cancel-comparison",
      "comparison-source", "comparison-status", "comparison-results", "input-modes",
    ].map((id) => [id, document.getElementById(id)]));
  }

  bind() {
    this.elements["input-provider"].addEventListener("change", () => {
      this.cancel();
      this.page.cancelTranscription();
      this.renderModels();
      this.page.render();
    });
    this.elements["input-model"].addEventListener("change", () => this.cancel());
    this.elements["input-modes"].addEventListener("change", (event) => {
      this.mode = event.target.value;
      this.cancel();
      this.page.cancelTranscription();
      this.page.render();
    });
    this.elements["compare-take"].addEventListener("click", () => void this.run());
    this.elements["cancel-comparison"].addEventListener("click", () => this.cancel());
  }

  async load() {
    this.loading = true;
    try {
      const response = await fetch("/prompt-reader/setups");
      const catalog = await response.json();
      if (!response.ok) throw new Error(catalog.message ?? catalog.error ?? "Setups unavailable");
      this.catalog = catalog;
      this.message = "";
      this.renderPresets();
    } catch (error) {
      this.message = error.message;
    }
    this.loading = false;
    try {
      const response = await fetch("/speech-providers/catalogue");
      if (response.ok) { this.providers = await response.json(); this.renderProviders(); }
    } catch { /* Local comparison is independent of optional provider metadata. */ }
    this.page.render();
  }

  renderPresets() {
    this.checks = this.catalog.setups.map((setup) => {
      const label = document.createElement("label");
      label.className = "preset-choice";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = setup.id;
      input.checked = this.selected.has(setup.id);
      input.addEventListener("change", () => {
        this.message = "";
        if (input.checked) this.selected.add(setup.id);
        else this.selected.delete(setup.id);
        this.page.render();
      });
      const content = document.createElement("span");
      const title = document.createElement("span");
      title.className = "preset-title";
      const name = document.createElement("strong");
      name.textContent = setup.label;
      const tags = document.createElement("span");
      tags.className = "preset-tags";
      tags.textContent = [setup.id === this.catalog.defaultSetupId ? "Recommended" : "", setup.operation === "transcribe" && (setup.controls?.prompt || setup.controls?.list?.content) ? "Uses fixed vocabulary" : ""].filter(Boolean).join(" · ");
      title.append(name, tags);
      const fact = document.createElement("small");
      fact.className = "preset-fact";
      const quantization = typeof setup.quantization === "string" ? setup.quantization : setup.quantization?.loaded;
      fact.textContent = [setup.operation === "cleanup" ? "Completed take: Ultra transcription, then text cleanup" : "Transcribes the completed recording", setup.controls?.provider ?? setup.runtime?.device ?? setup.runtime?.loadDeviceRequest, quantization].filter(Boolean).join(" · ");
      const limits = document.createElement("small");
      limits.className = "preset-fact";
      const bytes = setup.fileBytes ?? setup.files?.reduce((total, file) => total + file.bytes, 0);
      limits.textContent = [setup.maxSamples ? `Up to ${setup.maxSamples / 16_000} s` : "", Number.isFinite(bytes) ? `Model files: ${(bytes / 1_048_576).toFixed(0)} MiB` : ""].filter(Boolean).join(" · ");
      const status = document.createElement("small");
      status.className = "preset-readiness";
      status.textContent = setup.readiness?.ready === true ? "Ready to load" : `Needs setup: ${setup.readiness?.reason ?? "Readiness unavailable"}`;
      content.append(title, fact, limits, status);
      label.append(input, content);
      return { setup, input, label, tags, fact, status };
    });
    this.elements["input-presets"].replaceChildren(...this.checks.map(({ label }) => label));
  }

  renderProviders() {
    this.apiEntries = (this.providers.input ?? []).filter((entry) => entry.selectable && entry.config.provider.kind !== "local");
    const kinds = [...new Set(this.apiEntries.map((entry) => entry.config.provider.kind))];
    this.elements["input-provider"].replaceChildren(new Option("Local", "local"), ...kinds.map((kind) => new Option(kind, kind)));
    this.elements["input-provider"].value = "local";
    this.renderModels();
  }

  provider() { return this.elements["input-provider"].value || "local"; }

  renderModels() {
    const entries = this.apiEntries?.filter((entry) => entry.config.provider.kind === this.provider()) ?? [];
    this.elements["input-model"].replaceChildren(...entries.map((entry) => new Option(entry.label, entry.id)));
    this.elements["input-model"].value = entries[0]?.id ?? "";
  }

  source() {
    if (this.page.take) return { kind: "live", take: this.page.take };
    const saved = this.page.savedTake();
    return saved ? { kind: "saved", takeName: saved.name, saved } : null;
  }

  /** Completed live captures auto compare locally. APIs always need the button. */
  completeTake(take) {
    if (this.provider() !== "local") return;
    if (this.mode === "current-list") {
      take.transcript = { state: "pending", model: this.page.transcriptionConfig?.model };
      void this.page.transcribe(take);
    } else void this.run();
  }

  cancel() {
    const job = this.job;
    this.job = null;
    job?.controller.abort();
    if (job) for (const row of this.rows) if (!row.result) row.phase = "cancelled";
    if (job) this.message = "Comparison cancelled";
    this.renderRows();
    this.page.render();
  }

  invalidate() {
    this.cancel();
    this.rows = [];
    this.message = "";
    this.resultSource = "";
    this.resultIdentity = null;
    this.renderRows();
  }

  async run() {
    const source = this.source();
    if (!source || this.page.gate.capturing || this.page.draining) return;
    if (this.provider() !== "local") {
      await this.runApi(source);
      return;
    }
    if (!this.catalog) {
      this.message ||= this.loading ? "Setups are loading. Compare this take when ready." : "Setups unavailable. Reload the page to try again.";
      this.page.render();
      return;
    }
    const setups = this.catalog?.setups.filter((setup) => this.selected.has(setup.id)) ?? [];
    if (!setups.length) { this.message = "Select a setup to compare."; this.page.render(); return; }
    this.cancel();
    this.message = "";
    const job = { requestId: `reader-${Date.now()}-${++this.serial}`, sourceKind: source.kind, controller: new AbortController() };
    this.job = job;
    this.resultSource = this.sourceLabel();
    this.rows = setups.map((setup) => ({ setup, phase: "queued", result: null }));
    this.renderRows();
    this.page.render();
    try {
      const setupIds = setups.map((setup) => setup.id);
      const metadata = { requestId: job.requestId, setupIds };
      const query = new URLSearchParams({ requestId: job.requestId, setupIds: JSON.stringify(setupIds) });
      const response = await fetch(source.kind === "live" ? `/prompt-reader/comparisons?${query}` : "/prompt-reader/comparisons", {
        method: "POST",
        headers: { "content-type": source.kind === "live" ? "application/octet-stream" : "application/json" },
        body: source.kind === "live" ? source.take.samples.buffer : JSON.stringify({ ...metadata, source: { kind: "saved", takeName: source.takeName } }),
        signal: job.controller.signal,
      });
      if (this.job !== job) return;
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message ?? error.error ?? `Comparison unavailable (${response.status})`);
      }
      await this.readStream(response, job);
      if (this.job === job && !job.complete) throw new Error("Comparison incomplete. Compare the take again.");
    } catch (error) {
      if (this.job !== job) return;
      this.message = error.message;
      for (const row of this.rows) if (!row.result) row.phase = "error";
    } finally {
      if (this.job === job) { this.job = null; this.renderRows(); this.page.render(); }
    }
  }

  async runApi(source) {
    if (source.kind === "saved") {
      this.message = "Saved takes stay local. API transcription is not authorized.";
      this.page.render();
      return;
    }
    const entry = this.apiEntries?.find((candidate) => candidate.id === this.elements["input-model"].value);
    if (!entry) return;
    this.cancel();
    const job = { requestId: `reader-api-${Date.now()}-${++this.serial}`, controller: new AbortController() };
    this.job = job;
    this.message = "Transcribing with selected API";
    this.resultSource = this.sourceLabel();
    this.rows = [{ setup: { id: entry.id, label: entry.label, controls: entry.config }, phase: "processing", result: null }];
    this.renderRows();
    this.page.render();
    const post = async (url, init) => {
      const response = await fetch(url, { ...init, signal: job.controller.signal });
      const body = await response.json();
      if (!response.ok) throw new Error(body.message ?? body.error ?? "API transcription unavailable");
      return body;
    };
    try {
      await post("/speech-providers/session", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      if (this.job !== job) return;
      const selected = await post("/speech-providers/selection", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ direction: "input", catalogueRevision: this.providers.revision, config: entry.config }) });
      if (this.job !== job) return;
      const query = new URLSearchParams({ requestId: job.requestId, config: JSON.stringify(entry.config), catalogueRevision: selected.catalogueRevision, selectionRevision: String(selected.selectionRevision) });
      const result = await post(`/prompt-reader/api-transcriptions?${query}`, { method: "POST", headers: { "content-type": "application/octet-stream" }, body: source.take.samples.buffer });
      if (this.job !== job) return;
      if (typeof result.result?.text !== "string") throw new Error("API transcription response is incomplete");
      this.rows[0].result = { ...result.result, status: "complete", providerTiming: result.result.timings, appliedConfig: result.config };
      this.rows[0].phase = "complete";
      this.message = "Transcription finished";
    } catch (error) {
      if (this.job !== job) return;
      this.rows[0].phase = "error";
      this.message = error.message;
    } finally {
      if (this.job === job) { this.job = null; this.renderRows(); this.page.render(); }
    }
  }

  async readStream(response, job) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    try {
      while (this.job === job) {
        const { done, value } = await reader.read();
        if (this.job !== job) break;
        pending += decoder.decode(value, { stream: !done });
        if (pending.length > 1_000_000) throw new Error("Comparison response too large");
        let newline;
        while ((newline = pending.indexOf("\n")) !== -1) {
          const line = pending.slice(0, newline);
          pending = pending.slice(newline + 1);
          if (line.trim()) this.accept(JSON.parse(line), job);
        }
        if (done) {
          if (pending.trim()) this.accept(JSON.parse(pending), job);
          break;
        }
      }
    } finally { reader.releaseLock(); }
  }

  accept(event, job) {
    if (this.job !== job || event.requestId !== job.requestId) return;
    if (event.type === "accepted") {
      if (job.comparisonId != null) return;
      job.comparisonId = event.comparisonId;
      this.resultIdentity = event.pcm;
      this.resultManifest = event.manifestSha256;
      if (job.sourceKind === "saved") this.resultSource += ` · ${(event.pcm.sampleCount / event.pcm.sampleRate).toFixed(1)} s`;
    }
    if (event.comparisonId !== job.comparisonId) return;
    const row = this.rows.find((candidate) => candidate.setup.id === event.row?.setupId);
    if (event.type === "row" && row && !row.result) {
      row.phase = event.row.status;
      if (["complete", "error", "skipped", "cancelled"].includes(row.phase)) row.result = event.row;
    }
    if (event.type === "done") {
      job.complete = this.rows.every((candidate) => candidate.result);
      this.message = job.complete ? "Comparison finished" : "Comparison incomplete. Compare the take again.";
    }
    this.renderRows();
    this.page.render();
  }

  sourceLabel() {
    const source = this.source();
    if (!source) return "";
    const prompt = this.page.current()?.id ?? "";
    return source.kind === "live"
      ? `${prompt} · Unsaved take · ${(source.take.samples.length / 16_000).toFixed(1)} s`
      : `${prompt} · Take ${source.saved.number} of ${source.saved.count} · ${source.takeName}`;
  }

  render() {
    const source = this.source();
    const local = this.provider() === "local";
    const frozen = this.page.gate.capturing || this.page.draining;
    this.elements["input-selection"].disabled = frozen;
    this.elements["input-local"].hidden = !local;
    this.elements["input-api"].hidden = local;
    this.elements["input-presets"].hidden = this.mode !== "compare";
    this.elements["input-transfer"].textContent = local ? "Fixed presets. The editable known word list applies to Current list only." : `Audio is sent to ${this.provider()} for transcription. Saved takes stay local.`;
    this.elements["comparison-source"].textContent = this.resultSource || this.sourceLabel();
    this.elements["comparison-status"].textContent = this.message || (local && this.mode === "compare" && !this.selected.size ? "Select a setup to compare." : this.job ? `${this.rows.filter((row) => row.result).length} of ${this.rows.length} complete` : "Record once to compare selected setups.");
    const button = this.elements["compare-take"];
    button.textContent = this.job ? "Comparing" : local ? source?.kind === "saved" ? "Compare saved take" : "Compare take" : "Transcribe take";
    button.hidden = local && this.mode === "current-list";
    button.disabled = Boolean(this.job) || frozen || !source || (local ? !this.selected.size || !this.catalog : source.kind === "saved");
    this.elements["cancel-comparison"].hidden = !this.job;
  }

  renderRows() {
    for (const row of this.rows) {
      if (!row.view) {
        const section = document.createElement("section");
        section.className = "setup-result";
        const heading = document.createElement("h3");
        heading.textContent = row.setup.label;
        const phase = document.createElement("p");
        phase.className = "heard-note";
        const text = document.createElement("p");
        text.className = "setup-transcript";
        const error = document.createElement("p");
        error.className = "heard-error";
        const timing = document.createElement("p");
        timing.className = "heard-note";
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Details";
        const identity = document.createElement("p");
        details.append(summary, identity);
        section.append(heading, phase, text, error, timing, details);
        row.view = { section, phase, text, error, timing, identity };
      }
      const { phase, text, error, timing, identity } = row.view;
      const result = row.result;
      phase.textContent = row.phase;
      text.textContent = result?.text ?? result?.error?.rawText ?? "";
      if (result?.status === "complete" && result.text === "") text.textContent = row.setup.operation === "cleanup" ? "Empty output" : "No speech detected";
      if (result?.error?.rawText != null && result.status !== "complete") phase.textContent = "Incomplete output";
      if (result?.status === "complete" && result.text) {
        const words = diffWords(this.page.current()?.text ?? "", result.text);
        let index = 0;
        text.replaceChildren(...result.text.split(/(\s+)/).filter(Boolean).map((token) => {
          const span = document.createElement("span");
          span.textContent = token;
          if (!/^\s+$/.test(token) && words[index++]?.differs) span.className = "differs";
          return span;
        }));
      }
      error.textContent = result?.error?.message ?? "";
      timing.textContent = result?.providerTiming
        ? [["Provider request wall", "providerRequestWallMs"], ["Provider processing", "providerProcessingMs"], ["Provider network", "providerNetworkMs"]].map(([label, key]) => `${label}: ${formatObservation(result.providerTiming[key])}`).join(" · ")
        : [["Queue", "queueMs"], ["Startup", "startupMs"], ["Load", "loadMs"], ["Processing", "processingMs"], ["Worker wall", "workerWallMs"], ["Request wall", "requestWallMs"], ["Service wall", "serviceWallMs"], ["Ultra dependency processing", "dependencyProcessingMs"]].map(([label, key]) => `${label}: ${formatMs(result?.timings?.[key])}`).join(" · ");
      if (!result?.providerTiming) timing.textContent += ` · First use: ${result?.timings?.firstUse == null ? "Unavailable" : result.timings.firstUse ? "yes" : "no"}`;
      identity.textContent = JSON.stringify({ manifestSha256: this.resultManifest, setupId: row.setup.id, revision: row.setup.revision, pcm: result?.pcm ?? this.resultIdentity, controls: row.setup.controls, runtime: row.setup.runtime, rawText: result?.text ?? result?.error?.rawText, backendRawText: result?.rawText, applied: result?.appliedConfig });
    }
    const root = this.elements["comparison-results"];
    // Preserve native disclosure focus/open state during progressive updates.
    if (root.children.length !== this.rows.length || this.rows.some((row, index) => root.children[index] !== row.view.section)) root.replaceChildren(...this.rows.map((row) => row.view.section));
  }

}

function formatMs(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)} ms` : "Unavailable";
}

function formatObservation(value) {
  if (!value || value.status === "unavailable") return `Unavailable${value?.reason ? `: ${value.reason}` : ""}`;
  return `${formatMs(value.ms)} (${value.status === "provider-reported" ? "provider reported" : "measured"}; ${value.boundary})`;
}

/** Case and punctuation insensitive word key; empty for pure punctuation. */
function wordKey(word) {
  return word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
}

/**
 * The heard words in order, each marked when it is outside the longest
 * common word sequence shared with the prompt.
 */
export function diffWords(expected, heard) {
  const want = expected.split(/\s+/).map(wordKey).filter(Boolean);
  const words = heard.split(/\s+/).filter(Boolean);
  const keys = words.map(wordKey);
  const lengths = Array.from({ length: keys.length + 1 }, () => new Array(want.length + 1).fill(0));
  for (let i = keys.length - 1; i >= 0; i--) {
    for (let j = want.length - 1; j >= 0; j--) {
      lengths[i][j] = keys[i] && keys[i] === want[j]
        ? lengths[i + 1][j + 1] + 1
        : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }
  const differs = keys.map((key) => key !== "");
  for (let i = 0, j = 0; i < keys.length && j < want.length; ) {
    if (keys[i] && keys[i] === want[j]) {
      differs[i] = false;
      i++;
      j++;
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  return words.map((word, index) => ({ word, differs: differs[index] }));
}

