#!/usr/bin/env node
/**
 * Render the homepage showcase images with lucent itself, on the GPU, in a
 * real browser: crops of the sample enhanced at 4x and 16x. They're the
 * no-WebGPU fallback for the live hero demo and the source of the
 * "zoom, enhance" scroll sequence.
 *
 *   npm run render-assets      -> public/landing/*.webp
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";
import { createServer } from "vite";

const OUT = "public/landing";
mkdirSync(OUT, { recursive: true });

/** crops of public/samples/apollo11-aldrin.jpg (640 x 624) */
export const CROPS = {
  // helmet and visor, for the hero before/after at 4x
  hero: { x: 256, y: 0, w: 144, h: 144, scale: 4 },
  // the visor reflection, for the 16x zoom
  visor: { x: 292, y: 22, w: 72, h: 72, scale: 16 },
};

const local = process.env.LOCALAPPDATA ?? "";
const executablePath = [
  process.env.LUCENT_BROWSER,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  `${local}\\Google\\Chrome\\Application\\chrome.exe`,
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
].find((p) => p && existsSync(p));

const server = await createServer({ server: { port: 4319 }, logLevel: "warn" });
await server.listen();
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
});
try {
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && console.error(m.text()));
  await page.goto(server.resolvedUrls.local[0] + "app.html");
  const results = await page.evaluate(async (crops) => {
    const { getGpu } = await import("/src/gpu/device.ts");
    const { Enhancer } = await import("/src/gpu/enhancer.ts");
    const { upscale } = await import("/src/gpu/upscale.ts");
    const { loadWeights } = await import("/src/app/models.ts");
    const { blendModels } = await import("/src/model/srvgg.ts");
    const gpu = await getGpu();
    const strong = await loadWeights("realesr-general-x4v3");
    const weak = await loadWeights("realesr-general-wdn-x4v3");
    const e = new Enhancer(gpu.device);
    const img = await createImageBitmap(await (await fetch("/samples/apollo11-aldrin.jpg")).blob());
    const out = {};
    for (const [name, c] of Object.entries(crops)) {
      const crop = new OffscreenCanvas(c.w, c.h);
      crop.getContext("2d").drawImage(img, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
      const src = crop.transferToImageBitmap();
      const W = c.w * c.scale;
      const H = c.h * c.scale;
      const dst = new OffscreenCanvas(W, H);
      const ctx = dst.getContext("2d");
      const t0 = performance.now();
      await upscale(
        e,
        src,
        { width: W, height: H },
        { models: (p) => blendModels(strong, weak, p === 0 ? 0.6 : 0.15), fidelity: 0, sharpen: 0.3, grain: 0 },
        (t) => t.pass === t.passes - 1 && ctx.putImageData(new ImageData(t.pixels, t.width, t.height), t.x, t.y),
      );
      const ms = performance.now() - t0;
      const blob = await dst.convertToBlob({ type: "image/webp", quality: 0.9 });
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let bin = "";
      for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      out[name] = { b64: btoa(bin), ms, W, H };
    }
    return out;
  }, CROPS);
  for (const [name, r] of Object.entries(results)) {
    const file = `${OUT}/${name}-${CROPS[name].scale}x.webp`;
    writeFileSync(file, Buffer.from(r.b64, "base64"));
    console.log(`${file}: ${r.W}x${r.H}, ${(r.b64.length * 0.75 / 1024).toFixed(0)} KB, rendered in ${r.ms.toFixed(0)} ms`);
  }
  writeFileSync(`${OUT}/crops.json`, JSON.stringify(CROPS, null, 2) + "\n");
} finally {
  await browser.close();
  await server.close();
}
