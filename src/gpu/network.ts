/**
 * SRVGGNetCompact on WebGPU: weights packed into mat4x4 blocks, one dispatch
 * per conv (PReLU fused), ping-ponging between two activation buffers sized
 * for the largest region it will see.
 */
import type { ConvLayer, SrvggModel } from "../model/srvgg";
import { f32ToF16 } from "../model/safetensors";
import { type ConvTiling, DEFAULT_TILING, type Precision, convWgsl, tilingFor } from "./kernels";

interface GpuLayer {
  in4: number;
  out4: number;
  tiling: ConvTiling;
  pipeline: GPUComputePipeline;
  weights: GPUBuffer;
  bias: GPUBuffer;
  alpha: GPUBuffer;
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

export function cachedPipeline(device: GPUDevice, code: string, label: string): GPUComputePipeline {
  let cache = pipelineCache.get(device);
  if (!cache) pipelineCache.set(device, (cache = new Map()));
  let p = cache.get(code);
  if (!p) {
    p = device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    cache.set(code, p);
  }
  return p;
}

/** [cout][cin][3][3] -> [og][ig][tap] mat4x4 (column j = input channel j). */
export function packConvWeights(layer: ConvLayer): Float32Array {
  const in4 = Math.ceil(layer.cin / 4);
  const out4 = Math.ceil(layer.cout / 4);
  const out = new Float32Array(out4 * in4 * 9 * 16);
  for (let co = 0; co < layer.cout; co++) {
    for (let ci = 0; ci < layer.cin; ci++) {
      for (let tap = 0; tap < 9; tap++) {
        const m = ((co >> 2) * in4 + (ci >> 2)) * 9 + tap;
        out[m * 16 + (ci & 3) * 4 + (co & 3)] = layer.weight[(co * layer.cin + ci) * 9 + tap];
      }
    }
  }
  return out;
}

function padTo4(v: Float32Array): Float32Array {
  const out = new Float32Array(Math.ceil(v.length / 4) * 4);
  out.set(v);
  return out;
}

export class GpuSrvgg {
  readonly layers: GpuLayer[];
  readonly scale: number;
  private paramBuf: GPUBuffer;
  private bufA: GPUBuffer | null = null;
  private bufB: GPUBuffer | null = null;
  private capacity = 0; // pixels
  private bindGroups: GPUBindGroup[] = [];

  /** bytes per packed group of 4 channels */
  readonly groupBytes: number;

  constructor(
    private device: GPUDevice,
    model: SrvggModel,
    readonly precision: Precision = "f32",
    tiling: ConvTiling = DEFAULT_TILING,
  ) {
    this.scale = model.scale;
    this.groupBytes = precision === "f16" ? 8 : 16;
    const upload = (data: Float32Array, label: string) => {
      const typed = precision === "f16" ? f32ToF16(data) : data;
      const buf = device.createBuffer({ label, size: Math.max(16, typed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, typed);
      return buf;
    };
    this.layers = model.layers.map((l, i) => {
      const shape = { in4: Math.ceil(l.cin / 4), out4: Math.ceil(l.cout / 4), prelu: !!l.alpha };
      const t = tilingFor(shape, tiling);
      return {
        ...shape,
        tiling: t,
        pipeline: cachedPipeline(device, convWgsl(shape, t, precision), `conv${i}`),
        weights: upload(packConvWeights(l), `w${i}`),
        bias: upload(padTo4(l.bias), `b${i}`),
        alpha: upload(padTo4(l.alpha ?? new Float32Array(4)), `a${i}`),
      };
    });
    this.paramBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Activation buffers fit regions of up to `pixels`; grows on demand. */
  reserve(pixels: number) {
    if (pixels <= this.capacity) return;
    this.bufA?.destroy();
    this.bufB?.destroy();
    const groups = Math.max(...this.layers.map((l) => Math.max(l.in4, l.out4)));
    const size = pixels * groups * this.groupBytes;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.bufA = this.device.createBuffer({ label: "actA", size, usage });
    this.bufB = this.device.createBuffer({ label: "actB", size, usage });
    this.capacity = pixels;
    // layer i reads (i even ? B : A) and writes the other; the packed input lives in B
    this.bindGroups = this.layers.map((l, i) => {
      const [src, dst] = i % 2 === 0 ? [this.bufB!, this.bufA!] : [this.bufA!, this.bufB!];
      return this.device.createBindGroup({
        layout: l.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: l.weights } },
          { binding: 2, resource: { buffer: l.bias } },
          { binding: 3, resource: { buffer: l.alpha } },
          { binding: 4, resource: { buffer: dst } },
          { binding: 5, resource: { buffer: this.paramBuf } },
        ],
      });
    });
  }

  /** Buffer the packed (rgb, 0) input must be written to before `encode`. */
  get input(): GPUBuffer {
    if (!this.bufB) throw new Error("GpuSrvgg: reserve() first");
    return this.bufB;
  }

  /** Buffer holding the final conv output (3*r*r channels) after `encode`. */
  get output(): GPUBuffer {
    return this.layers.length % 2 === 1 ? this.bufA! : this.bufB!;
  }

  /** Record the network over a w x h region. Call reserve() first. */
  encode(enc: GPUCommandEncoder, w: number, h: number) {
    if (w * h > this.capacity) throw new Error("GpuSrvgg: region exceeds reserved capacity");
    this.device.queue.writeBuffer(this.paramBuf, 0, new Uint32Array([w, h, 0, 0]));
    const pass = enc.beginComputePass({ label: "srvgg" });
    this.layers.forEach((l, i) => {
      const { tx, ty, py, og } = l.tiling;
      pass.setPipeline(l.pipeline);
      pass.setBindGroup(0, this.bindGroups[i]);
      pass.dispatchWorkgroups(Math.ceil(w / tx), Math.ceil(h / (ty * py)), l.out4 / og);
    });
    pass.end();
  }

  destroy() {
    for (const l of this.layers) {
      l.weights.destroy();
      l.bias.destroy();
      l.alpha.destroy();
    }
    this.bufA?.destroy();
    this.bufB?.destroy();
    this.paramBuf.destroy();
  }
}
