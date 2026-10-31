/** Model registry: which weights back each mode, and how denoise maps onto them. */
import { parseSafetensors } from "../model/safetensors";
import { type SrvggModel, blendModels, srvggFromTensors } from "../model/srvgg";

export type ModelKind = "standard" | "illustration";

export interface ModelInfo {
  label: string;
  hint: string;
  /** weights at full / zero denoise (DNI endpoints), or a single file */
  files: [string] | [string, string];
}

export const MODELS: Record<ModelKind, ModelInfo> = {
  standard: {
    label: "Standard",
    hint: "Photos and real-world images",
    files: ["realesr-general-x4v3", "realesr-general-wdn-x4v3"],
  },
  illustration: {
    label: "Illustration",
    hint: "Anime, art, line work, flat colour",
    files: ["realesr-animevideov3"],
  },
};

const loaded = new Map<string, Promise<SrvggModel>>();

async function fetchWithProgress(url: string, onProgress?: (frac: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!total || !onProgress) return res.arrayBuffer();
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out.buffer;
}

export function loadWeights(name: string, onProgress?: (frac: number) => void): Promise<SrvggModel> {
  let p = loaded.get(name);
  if (!p) {
    const url = `${import.meta.env.BASE_URL}models/${name}.safetensors`;
    p = fetchWithProgress(url, onProgress).then((buf) => srvggFromTensors(parseSafetensors(buf).tensors));
    p.catch(() => loaded.delete(name));
    loaded.set(name, p);
  }
  return p;
}

/**
 * Resolve a mode + denoise strength (0..1) to concrete weights.
 * For the standard model this is Real-ESRGAN's DNI blend between the
 * strong-denoise and weak-denoise checkpoints.
 */
export async function resolveModel(kind: ModelKind, denoise: number, onProgress?: (frac: number) => void): Promise<SrvggModel> {
  const { files } = MODELS[kind];
  if (files.length === 1) return loadWeights(files[0], onProgress);
  const progress = [0, 0];
  const report = (i: number) => (f: number) => {
    progress[i] = f;
    onProgress?.((progress[0] + progress[1]) / 2);
  };
  const [strong, weak] = await Promise.all([loadWeights(files[0], report(0)), loadWeights(files[1], report(1))]);
  if (denoise >= 1) return strong;
  if (denoise <= 0) return weak;
  return blendModels(strong, weak, denoise);
}
