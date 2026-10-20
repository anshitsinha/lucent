/** Minimal safetensors reader (F16 / F32 only), decoding everything to f32. */

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export type Metadata = Record<string, string>;

export function parseSafetensors(buf: ArrayBuffer): { tensors: Map<string, Tensor>; metadata: Metadata } {
  const dv = new DataView(buf);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] } | Metadata
  >;
  const base = 8 + headerLen;
  const tensors = new Map<string, Tensor>();
  let metadata: Metadata = {};
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") {
      metadata = entry as Metadata;
      continue;
    }
    const { dtype, shape, data_offsets } = entry as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const [start, end] = data_offsets;
    let data: Float32Array;
    if (dtype === "F32") {
      data = new Float32Array(buf.slice(base + start, base + end));
    } else if (dtype === "F16") {
      data = f16ToF32(new Uint16Array(buf.slice(base + start, base + end)));
    } else {
      throw new Error(`safetensors: unsupported dtype ${dtype} for ${name}`);
    }
    tensors.set(name, { shape, data });
  }
  return { tensors, metadata };
}

const F16_TABLE = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) t[h] = sign * mant * 2 ** -24;
    else if (exp === 31) t[h] = mant ? NaN : sign * Infinity;
    else t[h] = sign * (1 + mant / 1024) * 2 ** (exp - 15);
  }
  return t;
})();

export function f16ToF32(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = F16_TABLE[h[i]];
  return out;
}
