/**
 * Tiled enhancement pipeline. For each tile of the source image:
 *
 *   pack (tile + context apron) -> SRVGGNetCompact (4x) -> compose
 *   (pixel shuffle, resample to target scale, fidelity blend, alpha)
 *   -> finish (unsharp mask, grain, rgba8) -> readback
 *
 * The apron gives every tile real neighbouring pixels to look at, so tiles
 * stitch without seams; only the true image border sees zero padding, which
 * is what the network was trained with.
 */
import { COMPOSE_WGSL, FINISH_WGSL, PACK_WGSL } from "./kernels";
import { GpuSrvgg, cachedPipeline } from "./network";
import type { SrvggModel } from "../model/srvgg";

export interface EnhanceOptions {
  /** output size / input size, 1..4 */
  scale: number;
  /** 0 = pure AI output, 1 = plain bicubic */
  fidelity: number;
  /** unsharp-mask amount at output resolution, 0..2 */
  sharpen: number;
  /** film grain amount, 0..1 */
  grain: number;
  /** source pixels per tile side (before the apron) */
  tile?: number;
  /** context pixels around each tile (the 34-conv net sees ±34; 24 keeps seams <= 2/255) */
  apron?: number;
  signal?: AbortSignal;
  /** tiles are processed nearest-first to this source-space point */
  focus?: { x: number; y: number };
}

export interface TileResult {
  /** output-space rect */
  x: number;
  y: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray<ArrayBuffer>;
  index: number;
  total: number;
}

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export const MAX_SCALE = 4;

export function outputSize(w: number, h: number, scale: number) {
  return { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) };
}

/** Split [0,w)x[0,h) into near-equal tiles of at most `tile` px a side. */
export function planTiles(w: number, h: number, tile: number): Rect[] {
  const nx = Math.ceil(w / tile);
  const ny = Math.ceil(h / tile);
  const rects: Rect[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      rects.push({
        x0: Math.round((i * w) / nx),
        x1: Math.round(((i + 1) * w) / nx),
        y0: Math.round((j * h) / ny),
        y1: Math.round(((j + 1) * h) / ny),
      });
    }
  }
  return rects;
}

function align(n: number, a: number) {
  return Math.ceil(n / a) * a;
}

export class Enhancer {
  private net: GpuSrvgg | null = null;
  private packPipe: GPUComputePipeline;
  private composePipe: GPUComputePipeline;
  private finishPipe: GPUComputePipeline;
  private packParams: GPUBuffer;
  private composeParams: GPUBuffer;
  private finishParams: GPUBuffer;
  private source: GPUTexture | null = null;
  private composed: GPUBuffer | null = null;
  private finished: GPUBuffer | null = null;
  private readbacks: GPUBuffer[] = [];
  private bufferBytes = 0;
  private running: Promise<unknown> = Promise.resolve();

  constructor(readonly device: GPUDevice) {
    this.packPipe = cachedPipeline(device, PACK_WGSL, "pack");
    this.composePipe = cachedPipeline(device, COMPOSE_WGSL, "compose");
    this.finishPipe = cachedPipeline(device, FINISH_WGSL, "finish");
    const uni = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.packParams = uni(16);
    this.composeParams = uni(48);
    this.finishParams = uni(32);
  }

  setModel(model: SrvggModel) {
    this.net?.destroy();
    this.net = new GpuSrvgg(this.device, model);
  }

  /** Upload the image to enhance. Returns its size. */
  setSource(image: ImageBitmap | HTMLCanvasElement | OffscreenCanvas | ImageData) {
    this.source?.destroy();
    const { width, height } = image;
    const max = this.device.limits.maxTextureDimension2D;
    if (width > max || height > max) throw new Error(`Image is ${width}x${height}; this GPU handles at most ${max}px a side.`);
    this.source = this.device.createTexture({
      label: "source",
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (image instanceof ImageData) {
      this.device.queue.writeTexture({ texture: this.source }, image.data, { bytesPerRow: width * 4 }, [width, height]);
    } else {
      this.device.queue.copyExternalImageToTexture({ source: image }, { texture: this.source, premultipliedAlpha: false }, [width, height]);
    }
    return { width, height };
  }

  /**
   * Enhance the current source, streaming finished tiles to `onTile`.
   * Runs are serialised: a new run waits for the previous one to stop
   * (abort it via `signal` to make that quick).
   */
  run(opts: EnhanceOptions, onTile: (t: TileResult) => void): Promise<void> {
    const next = this.running.catch(() => {}).then(() => this.runNow(opts, onTile));
    this.running = next;
    return next;
  }

  private async runNow(opts: EnhanceOptions, onTile: (t: TileResult) => void) {
    const { device } = this;
    const net = this.net;
    const source = this.source;
    if (!net || !source) throw new Error("Enhancer: setModel() and setSource() first");
    const scale = Math.min(MAX_SCALE, Math.max(1, opts.scale));
    const iw = source.width;
    const ih = source.height;
    const tile = opts.tile ?? 256;
    const apron = opts.apron ?? 24;
    const out = outputSize(iw, ih, scale);
    const ox = (x: number) => (x >= iw ? out.width : Math.round(x * scale));
    const oy = (y: number) => (y >= ih ? out.height : Math.round(y * scale));

    let rects = planTiles(iw, ih, tile);
    if (opts.focus) {
      const { x, y } = opts.focus;
      const d = (r: Rect) => Math.hypot((r.x0 + r.x1) / 2 - x, (r.y0 + r.y1) / 2 - y);
      rects = rects.sort((a, b) => d(a) - d(b));
    }

    // size scratch buffers for the largest tile
    const maxRegion = (tile + 2 * apron + 2) ** 2;
    net.reserve(maxRegion);
    const maxOut = Math.ceil(tile * scale) + 2;
    this.reserveScratch((maxOut + 2) ** 2 * 16, maxOut * maxOut * 4);

    const packBG = device.createBindGroup({
      layout: this.packPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        { binding: 1, resource: { buffer: net.input } },
        { binding: 2, resource: { buffer: this.packParams } },
      ],
    });
    const composeBG = device.createBindGroup({
      layout: this.composePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: net.output } },
        { binding: 1, resource: source.createView() },
        { binding: 2, resource: { buffer: this.composed! } },
        { binding: 3, resource: { buffer: this.composeParams } },
      ],
    });
    const finishBG = device.createBindGroup({
      layout: this.finishPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.composed! } },
        { binding: 1, resource: { buffer: this.finished! } },
        { binding: 2, resource: { buffer: this.finishParams } },
      ],
    });

    const seed = (Math.random() * 2 ** 32) >>> 0;
    const runNet = opts.fidelity < 1;
    // Two readback buffers: the GPU works on tile n+1 while tile n is mapped.
    const done: (Promise<void> | null)[] = [null, null];
    let last: Promise<void> = Promise.resolve();

    for (let index = 0; index < rects.length; index++) {
      if (opts.signal?.aborted) break;
      const r = rects[index];
      const rx0 = Math.max(0, r.x0 - apron);
      const ry0 = Math.max(0, r.y0 - apron);
      const rw = Math.min(iw, r.x1 + apron) - rx0;
      const rh = Math.min(ih, r.y1 + apron) - ry0;
      const tx = ox(r.x0);
      const ty = oy(r.y0);
      const tw = ox(r.x1) - tx;
      const th = oy(r.y1) - ty;

      const pi = new Int32Array([rx0, ry0, rw, rh]);
      device.queue.writeBuffer(this.packParams, 0, pi);
      const cp = new ArrayBuffer(48);
      new Int32Array(cp, 0, 10).set([rx0, ry0, rw, rh, tx - 1, ty - 1, tw + 2, th + 2, iw, ih]);
      new Float32Array(cp, 40, 2).set([scale, Math.min(1, Math.max(0, opts.fidelity))]);
      device.queue.writeBuffer(this.composeParams, 0, cp);
      const fp = new ArrayBuffer(32);
      new Uint32Array(fp, 0, 4).set([tw, th, tx, ty]);
      new Float32Array(fp, 16, 2).set([opts.sharpen, opts.grain]);
      new Uint32Array(fp, 24, 1).set([seed]);
      device.queue.writeBuffer(this.finishParams, 0, fp);

      const enc = device.createCommandEncoder({ label: `tile${index}` });
      if (runNet) {
        const pack = enc.beginComputePass({ label: "pack" });
        pack.setPipeline(this.packPipe);
        pack.setBindGroup(0, packBG);
        pack.dispatchWorkgroups(Math.ceil(rw / 16), Math.ceil(rh / 16));
        pack.end();
        net.encode(enc, rw, rh);
      }
      const post = enc.beginComputePass({ label: "post" });
      post.setPipeline(this.composePipe);
      post.setBindGroup(0, composeBG);
      post.dispatchWorkgroups(Math.ceil((tw + 2) / 16), Math.ceil((th + 2) / 16));
      post.setPipeline(this.finishPipe);
      post.setBindGroup(0, finishBG);
      post.dispatchWorkgroups(Math.ceil(tw / 16), Math.ceil(th / 16));
      post.end();
      const bytes = tw * th * 4;
      const rb = this.readbacks[index % 2];
      await done[index % 2]; // tile n-2 has released this buffer
      enc.copyBufferToBuffer(this.finished!, 0, rb, 0, align(bytes, 4));
      device.queue.submit([enc.finish()]);

      const prev = last;
      last = done[index % 2] = (async () => {
        await rb.mapAsync(GPUMapMode.READ, 0, align(bytes, 4));
        const pixels = new Uint8ClampedArray(rb.getMappedRange(0, bytes).slice(0));
        rb.unmap();
        await prev; // keep tile callbacks in order
        if (!opts.signal?.aborted) onTile({ x: tx, y: ty, width: tw, height: th, pixels, index, total: rects.length });
      })();
    }
    await last;
  }

  private reserveScratch(composedBytes: number, finishedBytes: number) {
    if (this.composed && this.composed.size >= composedBytes && this.bufferBytes >= finishedBytes) return;
    this.composed?.destroy();
    this.finished?.destroy();
    for (const b of this.readbacks) b.destroy();
    const { device } = this;
    this.composed = device.createBuffer({ label: "composed", size: composedBytes, usage: GPUBufferUsage.STORAGE });
    this.finished = device.createBuffer({ label: "finished", size: align(finishedBytes, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readbacks = [0, 1].map((i) =>
      device.createBuffer({ label: `readback${i}`, size: align(finishedBytes, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    );
    this.bufferBytes = finishedBytes;
  }

  destroy() {
    this.net?.destroy();
    this.source?.destroy();
    this.composed?.destroy();
    this.finished?.destroy();
    for (const b of this.readbacks) b.destroy();
    this.packParams.destroy();
    this.composeParams.destroy();
    this.finishParams.destroy();
  }
}
