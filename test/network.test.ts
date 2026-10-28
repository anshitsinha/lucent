import { describe, expect, it } from "vitest";
import { Enhancer, type TileResult } from "../src/gpu/enhancer";
import { GpuSrvgg } from "../src/gpu/network";
import { packWgsl } from "../src/gpu/kernels";
import { blendModels } from "../src/model/srvgg";
import { srvggFeaturesCpu, srvggForwardCpu } from "../src/reference/srvgg-cpu";
import { device, loadModel, planar, readBuffer, testImage } from "./helpers";

async function runEnhancer(e: Enhancer, img: ImageData, opts: Parameters<Enhancer["run"]>[0]) {
  e.setSource(img);
  const tiles: TileResult[] = [];
  await e.run(opts, (t) => tiles.push(t));
  const w = Math.round(img.width * opts.scale);
  const h = Math.round(img.height * opts.scale);
  const out = new Uint8ClampedArray(w * h * 4);
  for (const t of tiles) {
    for (let y = 0; y < t.height; y++) {
      out.set(t.pixels.subarray(y * t.width * 4, (y + 1) * t.width * 4), ((t.y + y) * w + t.x) * 4);
    }
  }
  return { out, w, h, tiles };
}

describe("SRVGGNetCompact on WebGPU", () => {
  for (const [name, precision] of [
    ["realesr-animevideov3", "f32"],
    ["realesr-general-x4v3", "f32"],
    ["realesr-general-x4v3", "f16"],
  ] as const) {
    it(`${name} (${precision}): final conv features match the CPU reference`, async () => {
      const dev = await device();
      if (precision === "f16" && !dev.features.has("shader-f16")) return;
      const model = await loadModel(name);
      const img = testImage(21, 13); // deliberately not a multiple of the workgroup tile
      const { width: w, height: h } = img;

      const net = new GpuSrvgg(dev, model, precision);
      net.reserve(w * h);
      const tex = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      dev.queue.writeTexture({ texture: tex }, img.data, { bytesPerRow: w * 4 }, [w, h]);
      const pack = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: packWgsl(precision) }), entryPoint: "main" } });
      const params = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(params, 0, new Int32Array([0, 0, w, h]));
      const enc = dev.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pack);
      p.setBindGroup(0, dev.createBindGroup({ layout: pack.getBindGroupLayout(0), entries: [
        { binding: 0, resource: tex.createView() },
        { binding: 1, resource: { buffer: net.input } },
        { binding: 2, resource: { buffer: params } },
      ] }));
      p.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
      p.end();
      net.encode(enc, w, h);
      dev.queue.submit([enc.finish()]);

      const groups = 12;
      const gpu = await readBuffer(dev, net.output, groups * w * h * net.groupBytes, precision);
      const ref = srvggFeaturesCpu(model, planar(img), w, h);
      let maxErr = 0;
      let maxRef = 0;
      for (let c = 0; c < 48; c++) {
        for (let i = 0; i < w * h; i++) {
          const g = gpu[((c >> 2) * w * h + i) * 4 + (c & 3)];
          const r = ref[c * w * h + i];
          maxErr = Math.max(maxErr, Math.abs(g - r));
          maxRef = Math.max(maxRef, Math.abs(r));
        }
      }
      console.log(`${name} ${precision}: max |gpu - cpu| = ${maxErr.toExponential(2)} (max |ref| ${maxRef.toFixed(3)})`);
      expect(maxErr).toBeLessThan(precision === "f16" ? 2e-2 : 1e-3);
      net.destroy();
    });
  }

  for (const precision of ["f32", "f16"] as const) it(`full pipeline at 4x matches the CPU reference (${precision})`, async () => {
    const dev = await device();
    if (precision === "f16" && !dev.features.has("shader-f16")) return;
    const model = await loadModel("realesr-general-x4v3");
    const img = testImage(19, 14, 7);
    const e = new Enhancer(dev);
    e.setModel(model, precision);
    const { out, w, h } = await runEnhancer(e, img, { scale: 4, fidelity: 0, sharpen: 0, grain: 0 });
    const ref = srvggForwardCpu(model, planar(img), img.width, img.height);
    let maxErr = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < w * h; i++) {
        const r = Math.round(Math.min(1, Math.max(0, ref[c * w * h + i])) * 255);
        maxErr = Math.max(maxErr, Math.abs(out[i * 4 + c] - r));
      }
    }
    console.log(`4x pipeline ${precision}: max ${maxErr}/255`);
    expect(maxErr).toBeLessThanOrEqual(precision === "f16" ? 3 : 1);
    e.destroy();
  });

  it("tiles stitch without visible seams", async () => {
    const dev = await device();
    const model = await loadModel("realesr-general-x4v3");
    const img = testImage(70, 52, 3);
    const e = new Enhancer(dev);
    e.setModel(model);
    const opts = { scale: 2, fidelity: 0, sharpen: 0.5, grain: 0 };
    const whole = await runEnhancer(e, img, { ...opts, tile: 256 });
    const tiled = await runEnhancer(e, img, { ...opts, tile: 16, focus: { x: 40, y: 10 } });
    expect(whole.tiles.length).toBe(1);
    expect(tiled.tiles.length).toBe(20);
    let maxErr = 0;
    let sum = 0;
    for (let i = 0; i < whole.out.length; i++) {
      const d = Math.abs(whole.out[i] - tiled.out[i]);
      maxErr = Math.max(maxErr, d);
      sum += d;
    }
    console.log(`tiled vs whole: max ${maxErr}/255, mean ${(sum / whole.out.length).toFixed(4)}`);
    expect(maxErr).toBeLessThanOrEqual(2);
    e.destroy();
  });

  it("every scale produces finite, correctly sized output", async () => {
    const dev = await device();
    const a = await loadModel("realesr-general-x4v3");
    const b = await loadModel("realesr-general-wdn-x4v3");
    const e = new Enhancer(dev);
    e.setModel(blendModels(a, b, 0.5));
    const img = testImage(33, 17, 11);
    for (const scale of [1, 1.5, 2, 3, 4]) {
      const { out, w, h, tiles } = await runEnhancer(e, img, { scale, fidelity: 0.1, sharpen: 1, grain: 0.3 });
      expect(tiles.reduce((n, t) => n + t.width * t.height, 0)).toBe(w * h);
      // mean brightness should track the input (no blown-out or black tiles)
      let mo = 0;
      for (let i = 0; i < w * h; i++) mo += out[i * 4 + 1];
      let mi = 0;
      for (let i = 0; i < img.width * img.height; i++) mi += img.data[i * 4 + 1];
      expect(Math.abs(mo / (w * h) - mi / (img.width * img.height))).toBeLessThan(6);
    }
    e.destroy();
  });
});
