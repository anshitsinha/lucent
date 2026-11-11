/**
 * Upscaling beyond the network's native 4x by chaining passes.
 *
 * Scales up to 4x run in one pass. Above that, two passes split the scale
 * evenly (sqrt each, so 8x = 2.83x * 2.83x and 16x = 4x * 4x), which costs
 * about as much as one 4x pass on the intermediate and gives the second
 * pass a real AI-upscaled image to work from rather than a bicubic one.
 *
 * The second pass sees an already clean image, so callers should give it a
 * gentler model (e.g. less denoise) to avoid the over-smoothed "plastic" look.
 */
import { type Enhancer, MAX_SCALE, type TileResult } from "./enhancer";
import type { SrvggModel } from "../model/srvgg";

export const MAX_TOTAL_SCALE = MAX_SCALE * MAX_SCALE;

export interface Size {
  width: number;
  height: number;
}

/** Output size of each pass; the last entry is the target. */
export function planPasses(src: Size, target: Size): Size[] {
  const s = Math.max(target.width / src.width, target.height / src.height);
  if (s > MAX_TOTAL_SCALE + 1e-9) throw new Error(`planPasses: ${s.toFixed(2)}x exceeds ${MAX_TOTAL_SCALE}x`);
  if (s <= MAX_SCALE + 1e-9) return [{ ...target }];
  const s1 = Math.min(MAX_SCALE, Math.sqrt(s));
  const mid = {
    width: Math.max(1, Math.round(src.width * s1)),
    height: Math.max(1, Math.round(src.height * s1)),
  };
  return [mid, { ...target }];
}

export interface PassTile extends TileResult {
  pass: number;
  passes: number;
  /** output size of this pass */
  passSize: Size;
}

export interface UpscaleOptions {
  /** weights for each pass (index = pass) */
  models: (pass: number, passes: number) => SrvggModel;
  fidelity: number;
  sharpen: number;
  grain: number;
  signal?: AbortSignal;
  /** process tiles nearest to this point first (target-space px) */
  focus?: { x: number; y: number };
  onTileStart?: (rect: { x: number; y: number; width: number; height: number }, pass: number, passSize: Size) => void;
}

type Source = ImageBitmap | HTMLCanvasElement | OffscreenCanvas;

/**
 * Upscale `src` to `target`, streaming every tile of every pass to `onTile`
 * (intermediate passes too, so callers can show progress). Resolves true if
 * it ran to completion, false if aborted.
 */
export async function upscale(enhancer: Enhancer, src: Source, target: Size, o: UpscaleOptions, onTile: (t: PassTile) => void): Promise<boolean> {
  const plan = planPasses(src, target);
  let input: Source = src;
  for (let pass = 0; pass < plan.length; pass++) {
    if (o.signal?.aborted) return false;
    const size = plan[pass];
    const last = pass === plan.length - 1;
    enhancer.setModel(o.models(pass, plan.length));
    enhancer.setSource(input);
    const inter = last ? null : new OffscreenCanvas(size.width, size.height);
    const ctx = inter?.getContext("2d");
    const focus = o.focus && { x: (o.focus.x * input.width) / target.width, y: (o.focus.y * input.height) / target.height };
    await enhancer.run(
      {
        scale: 1,
        size,
        // "preserve original" blends toward the real original, so only pass 1
        fidelity: pass === 0 ? o.fidelity : 0,
        // finishing touches belong at final resolution
        sharpen: last ? o.sharpen : 0,
        grain: last ? o.grain : 0,
        signal: o.signal,
        focus,
        onTileStart: (r) => o.onTileStart?.(r, pass, size),
      },
      (t) => {
        ctx?.putImageData(new ImageData(t.pixels, t.width, t.height), t.x, t.y);
        onTile({ ...t, pass, passes: plan.length, passSize: size });
      },
    );
    if (o.signal?.aborted) return false;
    if (inter) input = inter;
  }
  return true;
}
