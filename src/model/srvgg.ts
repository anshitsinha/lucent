/**
 * SRVGGNetCompact (Real-ESRGAN's compact model), as plain arrays.
 *
 *   x -> conv3x3(3->64) -> PReLU -> [conv3x3(64->64) -> PReLU] x num_conv
 *     -> conv3x3(64 -> 3*r*r) -> pixel_shuffle(r) -> + nearest_upsample(x, r)
 *
 * Weights are kept in PyTorch layout: [out][in][ky][kx].
 */
import type { Tensor } from "./safetensors";

export interface ConvLayer {
  cin: number;
  cout: number;
  weight: Float32Array; // [cout][cin][3][3]
  bias: Float32Array; // [cout]
  /** PReLU slopes [cout]; absent for the final conv */
  alpha?: Float32Array;
}

export interface SrvggModel {
  scale: number;
  layers: ConvLayer[];
}

export function srvggFromTensors(tensors: Map<string, Tensor>, scale = 4): SrvggModel {
  const layers: ConvLayer[] = [];
  for (let i = 0; ; i += 2) {
    const w = tensors.get(`body.${i}.weight`);
    if (!w) break;
    const b = tensors.get(`body.${i}.bias`);
    if (!b || w.shape.length !== 4 || w.shape[2] !== 3 || w.shape[3] !== 3) {
      throw new Error(`srvgg: unexpected tensor layout at body.${i}`);
    }
    const act = tensors.get(`body.${i + 1}.weight`);
    layers.push({ cout: w.shape[0], cin: w.shape[1], weight: w.data, bias: b.data, alpha: act?.data });
  }
  if (layers.length < 2) throw new Error("srvgg: no conv layers found");
  const last = layers[layers.length - 1];
  if (last.alpha || last.cout !== 3 * scale * scale) throw new Error("srvgg: unexpected final layer");
  return { scale, layers };
}

/**
 * Deep network interpolation: theta = t * a + (1 - t) * b.
 * Real-ESRGAN's "denoise strength" is exactly this between
 * realesr-general-x4v3 (t = 1) and realesr-general-wdn-x4v3 (t = 0).
 */
export function blendModels(a: SrvggModel, b: SrvggModel, t: number): SrvggModel {
  if (a.layers.length !== b.layers.length) throw new Error("blend: architectures differ");
  const mix = (x: Float32Array, y: Float32Array) => {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = t * x[i] + (1 - t) * y[i];
    return out;
  };
  return {
    scale: a.scale,
    layers: a.layers.map((la, i) => {
      const lb = b.layers[i];
      return {
        cin: la.cin,
        cout: la.cout,
        weight: mix(la.weight, lb.weight),
        bias: mix(la.bias, lb.bias),
        alpha: la.alpha && lb.alpha ? mix(la.alpha, lb.alpha) : undefined,
      };
    }),
  };
}
