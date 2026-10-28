/**
 * Sweep conv tilings on the realesr-general-x4v3 network at one 304x304
 * region (a 256px tile + 24px apron) and report GPU throughput.
 *   npm run bench
 */
import { it } from "vitest";
import { type ConvTiling, type Precision } from "../src/gpu/kernels";
import { GpuSrvgg } from "../src/gpu/network";
import { device, loadModel } from "../test/helpers";

const W = 304;
const H = 304;

async function time(dev: GPUDevice, net: GpuSrvgg, reps: number) {
  const run = () => {
    const enc = dev.createCommandEncoder();
    net.encode(enc, W, H);
    dev.queue.submit([enc.finish()]);
  };
  run();
  await dev.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) run();
  await dev.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / reps;
}

it("conv tiling sweep", async () => {
  const dev = await device();
  const model = await loadModel("realesr-general-x4v3");
  const flops = model.layers.reduce((s, l) => s + 2 * 9 * l.cin * l.cout, 0) * W * H;
  const configs: (ConvTiling & { precision: Precision })[] = [];
  const precisions: Precision[] = dev.features.has("shader-f16") ? ["f32", "f16"] : ["f32"];
  for (const precision of precisions) for (const ws of [true]) {
    for (const [tx, ty] of [[16, 8], [32, 4], [8, 16], [16, 16]]) {
      for (const py of [2, 4, 8]) {
        for (const og of [4, 8]) {
          const b = precision === "f16" ? 8 : 16;
          if ((tx + 2) * (ty * py + 2) * b + (ws ? og * 9 * 4 * b : 0) > 16384) continue;
          configs.push({ tx, ty, py, og, ws, precision });
        }
      }
    }
  }
  const info = (await navigator.gpu.requestAdapter())?.info;
  console.log(`GPU: ${info?.vendor} ${info?.architecture} ${info?.description}`);
  const rows: string[] = [];
  for (const t of configs) {
    const net = new GpuSrvgg(dev, model, t.precision, t);
    net.reserve(W * H);
    const ms = await time(dev, net, 6);
    net.destroy();
    rows.push(`${JSON.stringify(t)}  ${ms.toFixed(1)} ms  ${(flops / ms / 1e9).toFixed(2)} TFLOP/s  ${((256 * 256) / ms / 1e3).toFixed(2)} MP/s (net, per 256px tile)`);
  }
  rows.sort((a, b) => parseFloat(a.split("  ")[1]) - parseFloat(b.split("  ")[1]));
  console.log(rows.join("\n"));
}, 600_000);
