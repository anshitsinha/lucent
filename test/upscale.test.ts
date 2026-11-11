import { describe, expect, it } from "vitest";
import { Enhancer } from "../src/gpu/enhancer";
import { type PassTile, planPasses, upscale } from "../src/gpu/upscale";
import { blendModels } from "../src/model/srvgg";
import { device, loadModel, testImage } from "./helpers";

describe("planPasses", () => {
  it("uses one pass up to 4x", () => {
    expect(planPasses({ width: 100, height: 50 }, { width: 400, height: 200 })).toEqual([{ width: 400, height: 200 }]);
    expect(planPasses({ width: 100, height: 50 }, { width: 250, height: 125 })).toHaveLength(1);
  });
  it("splits 16x into 4x * 4x and 8x evenly", () => {
    expect(planPasses({ width: 30, height: 20 }, { width: 480, height: 320 })).toEqual([
      { width: 120, height: 80 },
      { width: 480, height: 320 },
    ]);
    const [mid] = planPasses({ width: 100, height: 100 }, { width: 800, height: 800 });
    expect(mid.width).toBe(283);
  });
  it("keeps every pass within the network's 4x", () => {
    for (const s of [4.01, 5, 6.5, 9, 12, 15.9, 16]) {
      const src = { width: 37, height: 23 };
      const target = { width: Math.round(37 * s), height: Math.round(23 * s) };
      let prev = src;
      for (const p of planPasses(src, target)) {
        expect(p.width / prev.width).toBeLessThanOrEqual(4.0001);
        expect(p.height / prev.height).toBeLessThanOrEqual(4.0001);
        prev = p;
      }
      expect(prev).toEqual(target);
    }
  });
  it("rejects more than 16x", () => {
    expect(() => planPasses({ width: 10, height: 10 }, { width: 170, height: 170 })).toThrow();
  });
});

describe("multi-pass upscale on the GPU", () => {
  for (const scale of [8, 16]) {
    it(`${scale}x covers the target exactly and keeps the image's tone`, async () => {
      const dev = await device();
      const a = await loadModel("realesr-general-x4v3");
      const b = await loadModel("realesr-general-wdn-x4v3");
      const e = new Enhancer(dev);
      const img = testImage(23, 17, 5);
      const src = await createImageBitmap(img);
      const target = { width: 23 * scale, height: 17 * scale };
      const tiles: PassTile[] = [];
      const ok = await upscale(
        e,
        src,
        target,
        { models: (pass) => (pass === 0 ? blendModels(a, b, 0.5) : blendModels(a, b, 0.1)), fidelity: 0, sharpen: 0.3, grain: 0 },
        (t) => tiles.push(t),
      );
      expect(ok).toBe(true);
      const final = tiles.filter((t) => t.pass === t.passes - 1);
      expect(tiles[0].passes).toBe(2);
      expect(final.reduce((n, t) => n + t.width * t.height, 0)).toBe(target.width * target.height);
      let mean = 0;
      let n = 0;
      for (const t of final) for (let i = 1; i < t.pixels.length; i += 4) (mean += t.pixels[i]), n++;
      let ref = 0;
      for (let i = 1; i < img.data.length; i += 4) ref += img.data[i];
      expect(Math.abs(mean / n - ref / (img.width * img.height))).toBeLessThan(8);
      e.destroy();
    });
  }
});
