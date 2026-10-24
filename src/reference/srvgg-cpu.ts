/**
 * Straightforward CPU forward pass of SRVGGNetCompact - the ground truth the
 * WebGPU kernels are tested against. Follows PyTorch semantics exactly
 * (zero padding, PReLU per channel, pixel_shuffle channel order).
 * Slow on purpose: no tricks to get wrong.
 */
import type { SrvggModel } from "../model/srvgg";

/** Output of the final conv, planar [3*r*r][h][w], before pixel shuffle. */
export function srvggFeaturesCpu(model: SrvggModel, rgb: Float32Array, w: number, h: number): Float32Array {
  const hw = w * h;
  let x = rgb;
  for (const layer of model.layers) {
    const { cin, cout, weight, bias, alpha } = layer;
    const y = new Float32Array(cout * hw);
    for (let co = 0; co < cout; co++) {
      const out = y.subarray(co * hw, (co + 1) * hw);
      out.fill(bias[co]);
      for (let ci = 0; ci < cin; ci++) {
        const inp = x.subarray(ci * hw, (ci + 1) * hw);
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const k = weight[((co * cin + ci) * 3 + ky) * 3 + kx];
            for (let py = 0; py < h; py++) {
              const sy = py + ky - 1;
              if (sy < 0 || sy >= h) continue;
              for (let px = 0; px < w; px++) {
                const sx = px + kx - 1;
                if (sx < 0 || sx >= w) continue;
                out[py * w + px] += k * inp[sy * w + sx];
              }
            }
          }
        }
      }
      if (alpha) {
        const a = alpha[co];
        for (let i = 0; i < hw; i++) if (out[i] < 0) out[i] *= a;
      }
    }
    x = y;
  }
  return x;
}

/** rgb: planar [3][h][w] in [0,1]. Returns planar [3][h*r][w*r], unclamped. */
export function srvggForwardCpu(model: SrvggModel, rgb: Float32Array, w: number, h: number): Float32Array {
  const hw = w * h;
  const x = srvggFeaturesCpu(model, rgb, w, h);
  // pixel shuffle + nearest-upsampled input
  const r = model.scale;
  const W = w * r;
  const H = h * r;
  const out = new Float32Array(3 * W * H);
  for (let c = 0; c < 3; c++) {
    for (let Y = 0; Y < H; Y++) {
      for (let X = 0; X < W; X++) {
        const sy = Math.floor(Y / r);
        const sx = Math.floor(X / r);
        const ch = c * r * r + (Y % r) * r + (X % r);
        out[(c * H + Y) * W + X] = x[ch * hw + sy * w + sx] + rgb[c * hw + sy * w + sx];
      }
    }
  }
  return out;
}
