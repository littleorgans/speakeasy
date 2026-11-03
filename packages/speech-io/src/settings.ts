/**
 * Per model, per reading choices a model declares: id, label, one line note,
 * kind, offered values and default. They travel beside the portable
 * selection, never inside it, and are never saved with preferences. The host
 * resolves a request against the declarations, fills defaults and refuses
 * anything not offered; what it resolved is echoed as the applied settings.
 */
export type SettingValue = number | string | boolean;
export type SettingOption = Readonly<{ value: SettingValue; label: string }>;

type SettingBase = Readonly<{
  id: string;
  label: string;
  note: string;
  /** Shown under a closed Sampling disclosure rather than in the main row. */
  group?: "sampling";
  /** A boolean setting that, when true, makes this one step aside: only its default is then accepted. */
  disabledBy?: string;
  /** Consumed by the host's sentence pipeline before the voice runtime sees the request. */
  pipeline?: true;
}>;

export type VoiceSetting =
  | SettingBase & Readonly<{ kind: "choice"; default: SettingValue; options: readonly SettingOption[] }>
  | SettingBase & Readonly<{ kind: "text"; default: string; maxLength: number }>
  | SettingBase & Readonly<{ kind: "integer"; default: number; min: number; max: number }>;

export type Settings = Readonly<Record<string, SettingValue>>;

export class SettingsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SettingsError";
  }
}

/** Every declared setting, defaults filled in; unknown ids, unoffered values and stepped aside changes are refused. */
export function resolveSettings(declared: readonly VoiceSetting[], value: unknown = {}): Settings {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new SettingsError("settings must be an object");
  const given = value as Record<string, unknown>;
  for (const id of Object.keys(given)) {
    if (!declared.some((setting) => setting.id === id)) throw new SettingsError(`unknown setting ${id}`);
  }
  const resolved = Object.fromEntries(declared.map((setting) => {
    const chosen = Object.hasOwn(given, setting.id) ? given[setting.id] : setting.default;
    if (!accepts(setting, chosen)) throw new SettingsError(`unsupported ${setting.id}`);
    return [setting.id, chosen as SettingValue];
  }));
  for (const setting of declared) {
    if (setting.disabledBy && resolved[setting.disabledBy] === true && resolved[setting.id] !== setting.default) {
      throw new SettingsError(`${setting.id} steps aside under ${setting.disabledBy}`);
    }
  }
  return Object.freeze(resolved);
}

function accepts(setting: VoiceSetting, value: unknown): boolean {
  switch (setting.kind) {
    case "choice": return setting.options.some((option) => option.value === value);
    case "text": return typeof value === "string" && value.length <= setting.maxLength;
    case "integer": return Number.isSafeInteger(value) && (value as number) >= setting.min && (value as number) <= setting.max;
  }
}

