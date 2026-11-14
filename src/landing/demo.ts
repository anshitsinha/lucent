/**
 * Hero demo: a before/after of the helmet crop. It starts with a result
 * pre-rendered by lucent, then, if the visitor has WebGPU, re-enhances the
 * crop live on their own GPU with the real engine and says how long it took.
 */
import crops from "../../public/landing/crops.json";

const SAMPLE = "/samples/apollo11-aldrin.jpg";

export async function startDemo() {
  const stage = document.getElementById("demo-stage")!;
  const before = document.getElementById("demo-before-canvas") as HTMLCanvasElement;
  const after = document.getElementById("demo-after") as HTMLImageElement;
  const scan = document.getElementById("demo-scan")!;
  const readout = document.getElementById("demo-readout")!;
  const status = document.getElementById("demo-status")!;
  const c = crops.hero;

  // --- split slider: drifts on its own until you touch it
  let user = false;
  const setSplit = (f: number) => stage.style.setProperty("--split", `${(f * 100).toFixed(2)}%`);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  const drift = () => {
    if (user) return;
    const t = (performance.now() - t0) / 1000;
    setSplit(0.5 + 0.32 * Math.sin(t * 0.9));
    requestAnimationFrame(drift);
  };
  if (!reduced) requestAnimationFrame(drift);
  const fromEvent = (e: PointerEvent) => {
    const r = stage.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  stage.addEventListener("pointerdown", (e) => {
    user = true;
    stage.setPointerCapture(e.pointerId);
    fromEvent(e);
  });
  stage.addEventListener("pointermove", (e) => {
    if (e.buttons || e.pointerType === "mouse") {
      user = true;
      fromEvent(e);
    }
  });

  // --- the original crop, shown with hard pixels
  const sample = await createImageBitmap(await (await fetch(SAMPLE)).blob());
  const crop = new OffscreenCanvas(c.w, c.h);
  crop.getContext("2d")!.drawImage(sample, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
  before.getContext("2d")!.drawImage(crop, 0, 0);

  // --- live run on the visitor's GPU
  if (!("gpu" in navigator)) {
    status.textContent = "No WebGPU in this browser: showing a result pre-rendered by lucent.";
    return;
  }
  await new Promise((r) => setTimeout(r, 900)); // let the page settle first
  try {
    const [{ getGpu }, { Enhancer }, { loadWeights }] = await Promise.all([
      import("../gpu/device"),
      import("../gpu/enhancer"),
      import("../app/models"),
    ]);
    const gpu = await getGpu();
    readout.classList.add("busy");
    status.textContent = "Downloading the model (2.4 MB, once)…";
    const model = await loadWeights("realesr-general-x4v3", (f) => (status.textContent = `Downloading the model… ${Math.round(f * 100)}%`));
    const e = new Enhancer(gpu.device);
    e.setModel(model);
    e.setSource(crop);
    // warm-up compiles the shaders, so the timed run measures the GPU, not the driver
    status.textContent = `Compiling kernels for ${gpu.adapterInfo}…`;
    await e.run({ scale: 4, fidelity: 0, sharpen: 0.3, grain: 0 }, () => {});
    status.textContent = `Enhancing on ${gpu.adapterInfo}…`;
    scan.classList.add("on");
    const out = new OffscreenCanvas(c.w * 4, c.h * 4);
    const ctx = out.getContext("2d")!;
    const start = performance.now();
    await e.run({ scale: 4, fidelity: 0, sharpen: 0.3, grain: 0 }, (t) => ctx.putImageData(new ImageData(t.pixels, t.width, t.height), t.x, t.y));
    const ms = performance.now() - start;
    const blob = await out.convertToBlob({ type: "image/png" });
    after.src = URL.createObjectURL(blob);
    await after.decode().catch(() => {});
    setTimeout(() => scan.classList.remove("on"), 500);
    e.destroy();
    readout.classList.remove("busy");
    readout.classList.add("live");
    status.textContent = `Enhanced live on your GPU (${gpu.adapterInfo}) in ${ms.toFixed(0)} ms. Nothing was uploaded.`;
  } catch (err) {
    readout.classList.remove("busy");
    scan.classList.remove("on");
    status.textContent = "Couldn't run WebGPU here: showing a result pre-rendered by lucent.";
    console.warn("lucent demo:", err);
  }
}
