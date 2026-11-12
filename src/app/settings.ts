import type { ModelKind } from "./models";

export type OutputFormat = "png" | "jpeg" | "webp";

export interface Settings {
  model: ModelKind;
  /** output / input, 1..16 */
  scale: number;
  /** 0..100 */
  denoise: number;
  sharpen: number;
  fidelity: number;
  grain: number;
  format: OutputFormat;
  /** 50..100, JPEG / WebP only */
  quality: number;
}

export const ADJUST_DEFAULTS = { denoise: 50, sharpen: 15, fidelity: 0, grain: 0 } as const;

export const DEFAULTS: Settings = {
  model: "standard",
  scale: 2,
  ...ADJUST_DEFAULTS,
  format: "png",
  quality: 92,
};

const KEY = "lucent:settings:v1";

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const saved = JSON.parse(raw) as Partial<Settings>;
    const s = { ...DEFAULTS, ...saved };
    // guard against stale or hand-edited values
    if (!["standard", "illustration"].includes(s.model)) s.model = DEFAULTS.model;
    if (!["png", "jpeg", "webp"].includes(s.format)) s.format = DEFAULTS.format;
    s.scale = Math.min(16, Math.max(1, Number(s.scale) || DEFAULTS.scale));
    for (const k of ["denoise", "sharpen", "fidelity", "grain"] as const) s[k] = Math.min(100, Math.max(0, Number(s[k]) || 0));
    s.quality = Math.min(100, Math.max(50, Number(s.quality) || DEFAULTS.quality));
    return s;
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // private mode / storage disabled: settings just won't persist
  }
}
