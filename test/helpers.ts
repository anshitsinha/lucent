import { getGpu } from "../src/gpu/device";
import { parseSafetensors } from "../src/model/safetensors";
import { type SrvggModel, srvggFromTensors } from "../src/model/srvgg";

export async function device(): Promise<GPUDevice> {
  return (await getGpu()).device;
}

export async function loadModel(name: string): Promise<SrvggModel> {
  const res = await fetch(`/models/${name}.safetensors`);
  if (!res.ok) throw new Error(`fetch ${name}: ${res.status}`);
  return srvggFromTensors(parseSafetensors(await res.arrayBuffer()).tensors);
}

/** Deterministic test image: gradients, edges and a little noise. */
export function testImage(w: number, h: number, seed = 1): ImageData {
  let s = seed >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const edge = (x + 2 * y) % 11 < 5 ? 0.25 : 0;
      data[i] = 255 * (x / w * 0.7 + edge + rnd() * 0.08);
      data[i + 1] = 255 * (y / h * 0.6 + 0.2 + rnd() * 0.08);
      data[i + 2] = 255 * (0.5 + 0.4 * Math.sin(x * 0.7) * Math.cos(y * 0.5) + rnd() * 0.05);
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, w, h);
}

/** ImageData -> planar float rgb in [0,1] */
export function planar(img: ImageData): Float32Array {
  const { width: w, height: h, data } = img;
  const out = new Float32Array(3 * w * h);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) out[c * w * h + i] = data[i * 4 + c] / 255;
  }
  return out;
}

export async function readBuffer(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array> {
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.destroy();
  return out;
}
