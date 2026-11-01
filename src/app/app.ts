/**
 * App controller: wires the queue, settings panel, viewer and the GPU
 * enhancer together. Every settings change re-runs enhancement (debounced),
 * starting from the tiles nearest the centre of the view, painting finished
 * tiles over the previous result so the image never blanks.
 */
import { Enhancer, MAX_SCALE, outputSize } from "../gpu/enhancer";
import type { GpuContext } from "../gpu/device";
import { encode, outputName, pickFolder, saveTo } from "./export";
import { MODELS, type ModelKind, resolveModel } from "./models";
import { ADJUST_DEFAULTS, type OutputFormat, type Settings, loadSettings, saveSettings } from "./settings";
import { type CompareMode, Viewer } from "./viewer";

/** Canvas limits that hold across Chromium, Firefox and Safari. */
const MAX_SIDE = 16384;
const MAX_PIXELS = 120e6;

interface QueueItem {
  id: number;
  file: File;
  thumb: string;
  state: "idle" | "done" | "error";
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export class App {
  private settings: Settings = loadSettings();
  private enhancer: Enhancer;
  private viewer: Viewer;
  private queue: QueueItem[] = [];
  private nextId = 1;
  private current: QueueItem | null = null;
  private bitmap: ImageBitmap | null = null;
  private output: HTMLCanvasElement | null = null;
  private runCtl: AbortController | null = null;
  private runPromise: Promise<boolean> = Promise.resolve(false);
  private complete = false;
  private failed = false;
  private debounce = 0;
  private toastTimer = 0;
  private batching = false;

  constructor(gpu: GpuContext) {
    this.enhancer = new Enhancer(gpu.device);
    this.viewer = new Viewer($("viewer"));
    this.viewer.onZoom = (z) => ($("zoom-label").textContent = `${Math.round(z * 100)}%`);
    const precision = Enhancer.defaultPrecision(gpu.device);
    const badge = $("gpu-badge");
    badge.textContent = `${gpu.adapterInfo} · ${precision}`;
    badge.title = `WebGPU on ${gpu.adapterInfo}, ${precision === "f16" ? "half" : "single"} precision`;
    badge.classList.add("ok");
    this.bindPanel();
    this.bindFiles();
    this.bindKeys();
    this.renderPanel();
    // warm the model up while the user picks a file
    resolveModel(this.settings.model, this.settings.denoise / 100).catch(() => {});
  }

  // ------------------------------------------------------------- queue --

  addFiles(files: Iterable<File>) {
    const added: QueueItem[] = [];
    for (const file of files) {
      if (!file.type.startsWith("image/") && !/\.(png|jpe?g|webp|gif|bmp|avif|heic)$/i.test(file.name)) continue;
      const item: QueueItem = { id: this.nextId++, file, thumb: URL.createObjectURL(file), state: "idle" };
      this.queue.push(item);
      added.push(item);
    }
    if (!added.length) {
      this.toast("Those files aren't images lucent can open.", true);
      return;
    }
    this.renderQueue();
    if (!this.current || added.length === this.queue.length) void this.select(added[0]);
  }

  private remove(item: QueueItem) {
    URL.revokeObjectURL(item.thumb);
    const i = this.queue.indexOf(item);
    this.queue.splice(i, 1);
    if (item === this.current) {
      const next = this.queue[Math.min(i, this.queue.length - 1)];
      if (next) void this.select(next);
      else void this.clear();
    }
    this.renderQueue();
  }

  private async clear() {
    this.runCtl?.abort();
    await this.enhancer.idle();
    this.current = null;
    this.bitmap?.close();
    this.bitmap = null;
    this.output = null;
    this.viewer.setSource(null);
    $("empty").hidden = false;
    $("view-tools").hidden = true;
    $<HTMLButtonElement>("export-btn").disabled = true;
    this.setStatus("Ready", "");
    this.renderPanel();
  }

  private async select(item: QueueItem) {
    this.runCtl?.abort();
    this.current = item;
    this.renderQueue();
    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(item.file, { premultiplyAlpha: "none", colorSpaceConversion: "default" });
    } catch {
      item.state = "error";
      this.renderQueue();
      this.toast(`Couldn't decode ${item.file.name}.`, true);
      return;
    }
    if (this.current !== item) {
      bitmap.close();
      return;
    }
    await this.enhancer.idle();
    this.bitmap?.close();
    this.bitmap = bitmap;
    this.output = null;
    try {
      this.enhancer.setSource(bitmap);
    } catch (e) {
      item.state = "error";
      this.renderQueue();
      this.toast((e as Error).message, true);
      return;
    }
    $("empty").hidden = true;
    $("view-tools").hidden = false;
    $<HTMLButtonElement>("export-btn").disabled = false;
    this.renderPanel();
    this.start(true);
  }

  private renderQueue() {
    const q = $("queue");
    q.hidden = this.queue.length < 2;
    $("stage").classList.toggle("has-queue", !q.hidden);
    $("export-all-btn").hidden = this.queue.length < 2;
    q.replaceChildren(
      ...this.queue.map((item) => {
        const b = document.createElement("div");
        b.className = `qitem${item === this.current ? " active" : ""}`;
        b.tabIndex = 0;
        b.title = item.file.name;
        b.setAttribute("role", "button");
        const img = document.createElement("img");
        img.src = item.thumb;
        img.alt = item.file.name;
        img.loading = "lazy";
        b.append(img);
        if (item.state !== "idle") {
          const s = document.createElement("span");
          s.className = `qstate ${item.state}`;
          s.textContent = item.state === "done" ? "✓ saved" : "error";
          b.append(s);
        }
        const x = document.createElement("button");
        x.className = "qremove";
        x.textContent = "×";
        x.title = "Remove";
        x.addEventListener("click", (e) => {
          e.stopPropagation();
          this.remove(item);
        });
        b.append(x);
        b.addEventListener("click", () => item !== this.current && !this.batching && void this.select(item));
        b.addEventListener("keydown", (e) => e.key === "Enter" && b.click());
        return b;
      }),
    );
  }

  // --------------------------------------------------------- enhancing --

  private maxScale(): number {
    if (!this.bitmap) return MAX_SCALE;
    const { width: w, height: h } = this.bitmap;
    return Math.min(MAX_SCALE, Math.sqrt(MAX_PIXELS / (w * h)), MAX_SIDE / Math.max(w, h));
  }

  private effectiveScale() {
    return Math.max(1, Math.min(this.settings.scale, this.maxScale()));
  }

  /** (Re)start enhancement after a short debounce. */
  private start(immediate = false) {
    clearTimeout(this.debounce);
    this.runCtl?.abort();
    this.complete = false;
    this.failed = false;
    $<HTMLButtonElement>("export-btn").disabled = !this.bitmap;
    this.debounce = window.setTimeout(() => (this.runPromise = this.run()), immediate ? 0 : 140);
  }

  /** Resolves true when this run finished every tile. */
  private async run(): Promise<boolean> {
    const bitmap = this.bitmap;
    if (!bitmap) return false;
    const ctl = new AbortController();
    this.runCtl = ctl;
    await this.enhancer.idle();
    if (ctl.signal.aborted) return false;

    const s = this.settings;
    const progress = $("progress");
    let model;
    try {
      this.setStatus("Loading model…", "");
      progress.classList.add("indeterminate");
      model = await resolveModel(s.model, s.denoise / 100, (f) => this.setStatus(`Loading model ${Math.round(f * 100)}%`, ""));
    } catch (e) {
      progress.classList.remove("indeterminate");
      this.failed = true;
      this.setStatus("Couldn't load the model", "");
      this.toast((e as Error).message, true);
      return false;
    }
    progress.classList.remove("indeterminate");
    if (ctl.signal.aborted) return false;
    this.enhancer.setModel(model);

    const scale = this.effectiveScale();
    const { width, height } = outputSize(bitmap.width, bitmap.height, scale);
    if (!this.output || this.output.width !== width || this.output.height !== height) {
      // placeholder: a smooth upscale, replaced tile by tile
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      const ctx = c.getContext("2d")!;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(bitmap, 0, 0, width, height);
      this.output = c;
      this.viewer.setSource({ original: bitmap, enhanced: c }, true);
    }
    const out = this.output;
    const ctx = out.getContext("2d")!;
    const focus = this.viewer.focus;
    const t0 = performance.now();
    let done = 0;
    this.setProgress(0);
    this.setStatus("Enhancing…", `${width} × ${height}`);
    try {
      await this.enhancer.run(
        {
          scale,
          fidelity: s.fidelity / 100,
          sharpen: (s.sharpen / 100) * 2,
          grain: s.grain / 100,
          signal: ctl.signal,
          focus: { x: focus.x / scale, y: focus.y / scale },
          onTileStart: (r) => {
            if (!ctl.signal.aborted) this.viewer.activeTile = r;
          },
        },
        (t) => {
          ctx.putImageData(new ImageData(t.pixels, t.width, t.height), t.x, t.y);
          done++;
          this.setProgress(done / t.total);
          this.setStatus(`Enhancing · ${done}/${t.total}`, `${width} × ${height}`);
          this.viewer.invalidate();
        },
      );
    } catch (e) {
      if (this.runCtl === ctl) this.viewer.activeTile = null;
      this.failed = true;
      this.setStatus("Enhancement failed", "");
      this.toast(`GPU error: ${(e as Error).message}`, true);
      return false;
    }
    if (this.runCtl === ctl) this.viewer.activeTile = null;
    this.viewer.invalidate();
    if (ctl.signal.aborted) return false;
    const secs = (performance.now() - t0) / 1000;
    const mp = (bitmap.width * bitmap.height) / 1e6;
    this.complete = true;
    this.setProgress(1);
    this.setStatus(`Done in ${secs < 10 ? secs.toFixed(1) : Math.round(secs)} s`, `${(mp / secs).toFixed(2)} MP/s`);
    return true;
  }

  /** Wait until the image is fully enhanced with the current settings. */
  private async whenComplete(): Promise<void> {
    while (!this.complete) {
      if (!this.bitmap) throw new Error("No image");
      if (this.failed) throw new Error("Enhancement failed; change a setting to retry.");
      // a pending debounce hasn't started the run yet
      await new Promise((r) => setTimeout(r, 50));
      await this.runPromise;
    }
  }

  private async exportCurrent(dir: Awaited<ReturnType<typeof pickFolder>> = null) {
    if (!this.current || !this.output) return;
    const item = this.current;
    try {
      this.setStatus("Finishing…", "");
      await this.whenComplete();
      this.setStatus("Encoding…", "");
      const blob = await encode(this.output, this.settings.format, this.settings.quality);
      const name = outputName(item.file.name, this.effectiveScale(), this.settings.format);
      await saveTo(dir, blob, name);
      item.state = "done";
      this.renderQueue();
      this.setStatus("Saved", `${name} · ${(blob.size / 1e6).toFixed(1)} MB`);
    } catch (e) {
      this.toast((e as Error).message, true);
      this.setStatus("Export failed", "");
    }
  }

  private async exportAll() {
    if (this.batching || !this.queue.length) return;
    this.batching = true;
    const dir = await pickFolder();
    try {
      for (const item of [...this.queue]) {
        if (!this.queue.includes(item)) continue;
        if (item !== this.current) await this.select(item);
        if (this.current !== item) continue; // failed to open
        await this.exportCurrent(dir);
      }
      this.toast(dir ? "All images saved to the folder." : "All images exported.");
    } finally {
      this.batching = false;
    }
  }

  // ------------------------------------------------------------- panel --

  private set<K extends keyof Settings>(key: K, value: Settings[K]) {
    if (this.settings[key] === value) return;
    this.settings[key] = value;
    saveSettings(this.settings);
    this.renderPanel();
    if (key !== "format" && key !== "quality" && this.bitmap) this.start();
  }

  private bindPanel() {
    for (const b of $("model-seg").querySelectorAll<HTMLButtonElement>("button")) {
      b.addEventListener("click", () => this.set("model", b.dataset.model as ModelKind));
    }
    for (const b of $("scale-seg").querySelectorAll<HTMLButtonElement>("button")) {
      b.addEventListener("click", () => this.set("scale", Number(b.dataset.scale)));
    }
    for (const b of $("format-seg").querySelectorAll<HTMLButtonElement>("button")) {
      b.addEventListener("click", () => this.set("format", b.dataset.format as OutputFormat));
    }
    for (const row of document.querySelectorAll<HTMLElement>(".slider")) {
      const key = row.dataset.key as "denoise" | "sharpen" | "fidelity" | "grain" | "quality";
      const input = row.querySelector("input")!;
      input.addEventListener("input", () => this.set(key, Number(input.value)));
    }
    const dimInput = (id: string, axis: "width" | "height") => {
      const el = $<HTMLInputElement>(id);
      el.addEventListener("change", () => {
        if (!this.bitmap) return;
        const v = Number(el.value);
        if (!(v > 0)) return this.renderPanel();
        const scale = Math.round((v / this.bitmap[axis]) * 1000) / 1000;
        this.set("scale", Math.min(MAX_SCALE, Math.max(1, scale)));
        this.renderPanel();
      });
    };
    dimInput("out-w", "width");
    dimInput("out-h", "height");
    $("reset-btn").addEventListener("click", () => {
      Object.assign(this.settings, ADJUST_DEFAULTS);
      saveSettings(this.settings);
      this.renderPanel();
      if (this.bitmap) this.start();
    });
    $("export-btn").addEventListener("click", () => void this.exportCurrent());
    $("export-all-btn").addEventListener("click", () => void this.exportAll());
    for (const b of $("view-tools").querySelectorAll<HTMLButtonElement>("[data-mode]")) {
      b.addEventListener("click", () => this.setMode(b.dataset.mode as CompareMode));
    }
    $("zoom-in").addEventListener("click", () => this.viewer.zoomTo(this.viewer.zoomLevel * 1.5));
    $("zoom-out").addEventListener("click", () => this.viewer.zoomTo(this.viewer.zoomLevel / 1.5));
    $("zoom-label").addEventListener("click", () => (Math.abs(this.viewer.zoomLevel - 1) < 1e-3 ? this.viewer.fit() : this.viewer.zoomTo(1)));
  }

  private setMode(mode: CompareMode) {
    this.viewer.mode = mode;
    this.viewer.invalidate();
    for (const b of $("view-tools").querySelectorAll<HTMLButtonElement>("[data-mode]")) {
      b.classList.toggle("on", b.dataset.mode === mode);
    }
  }

  private renderPanel() {
    const s = this.settings;
    const seg = (id: string, attr: string, value: string) => {
      for (const b of $(id).querySelectorAll<HTMLButtonElement>("button")) {
        const on = b.dataset[attr] === value;
        b.classList.toggle("on", on);
        b.setAttribute("aria-checked", String(on));
      }
    };
    seg("model-seg", "model", s.model);
    seg("format-seg", "format", s.format);
    seg("scale-seg", "scale", Number.isInteger(s.scale) ? String(s.scale) : "");
    $("model-hint").textContent = MODELS[s.model].hint;

    for (const row of document.querySelectorAll<HTMLElement>(".slider")) {
      const key = row.dataset.key as keyof Settings;
      const input = row.querySelector("input")!;
      const v = Number(s[key]);
      if (document.activeElement !== input) input.value = String(v);
      const min = Number(input.min);
      const max = Number(input.max);
      input.style.setProperty("--fill", `${((v - min) / (max - min)) * 100}%`);
      row.querySelector("output")!.textContent = String(v);
    }
    const denoiseRow = document.querySelector<HTMLElement>('.slider[data-key="denoise"]')!;
    const noDenoise = MODELS[s.model].files.length === 1;
    denoiseRow.classList.toggle("disabled", noDenoise);
    denoiseRow.title = noDenoise ? "The illustration model has a fixed denoise level" : "";
    $("quality-row").hidden = s.format === "png";

    const note = $("scale-note");
    const w = $<HTMLInputElement>("out-w");
    const h = $<HTMLInputElement>("out-h");
    if (this.bitmap) {
      const scale = this.effectiveScale();
      const o = outputSize(this.bitmap.width, this.bitmap.height, scale);
      if (document.activeElement !== w) w.value = String(o.width);
      if (document.activeElement !== h) h.value = String(o.height);
      w.disabled = h.disabled = false;
      const mp = (o.width * o.height) / 1e6;
      if (scale < s.scale - 1e-6) {
        note.textContent = `Capped at ${scale.toFixed(2)}× — browsers can't hold images over ${MAX_PIXELS / 1e6} MP.`;
        note.className = "note warn";
      } else {
        note.textContent = `${this.bitmap.width} × ${this.bitmap.height} → ${scale.toFixed(scale % 1 ? 2 : 0)}× · ${mp.toFixed(1)} MP`;
        note.className = "note";
      }
    } else {
      w.value = h.value = "";
      w.disabled = h.disabled = true;
      note.textContent = "";
    }
  }

  private setProgress(f: number) {
    $("progress-bar").style.width = `${Math.round(f * 100)}%`;
  }

  private setStatus(text: string, stats: string) {
    $("status-text").textContent = text;
    $("status-stats").textContent = stats;
  }

  toast(msg: string, error = false) {
    const t = $("toast");
    t.textContent = msg;
    t.className = `toast${error ? " error" : ""}`;
    t.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => (t.hidden = true), error ? 6000 : 3000);
  }

  // ------------------------------------------------------- files, keys --

  private bindFiles() {
    const input = $<HTMLInputElement>("file-input");
    const open = () => input.click();
    $("open-btn").addEventListener("click", open);
    $("empty-open").addEventListener("click", open);
    $("sample-btn").addEventListener("click", async () => {
      const res = await fetch(`${import.meta.env.BASE_URL}samples/apollo11-aldrin.jpg`);
      const blob = await res.blob();
      this.addFiles([new File([blob], "apollo11-aldrin.jpg", { type: blob.type || "image/jpeg" })]);
    });
    input.addEventListener("change", () => {
      if (input.files?.length) this.addFiles(input.files);
      input.value = "";
    });

    const overlay = $("drop-overlay");
    let depth = 0;
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files");
    window.addEventListener("dragenter", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth++;
      overlay.hidden = false;
    });
    window.addEventListener("dragover", (e) => {
      if (hasFiles(e)) e.preventDefault();
    });
    window.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (!depth) overlay.hidden = true;
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      depth = 0;
      overlay.hidden = true;
      if (e.dataTransfer?.files.length) this.addFiles(e.dataTransfer.files);
    });
    window.addEventListener("paste", (e) => {
      const files = [...(e.clipboardData?.files ?? [])];
      if (files.length) this.addFiles(files);
    });
  }

  private bindKeys() {
    let held: CompareMode | null = null;
    window.addEventListener("keydown", (e) => {
      const target = e.target as HTMLElement;
      if (target.matches("input[type=number], input[type=text]")) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "o") {
        e.preventDefault();
        $("file-input").click();
      } else if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void this.exportCurrent();
      } else if (mod) {
        return;
      } else if (e.key === " " && this.bitmap && !held) {
        e.preventDefault();
        held = this.viewer.mode;
        this.setMode("original");
      } else if (e.key === "f" || e.key === "F") {
        this.viewer.fit();
      } else if (e.key === "1") {
        this.viewer.zoomTo(1);
      } else if (e.key === "+" || e.key === "=") {
        this.viewer.zoomTo(this.viewer.zoomLevel * 1.25);
      } else if (e.key === "-") {
        this.viewer.zoomTo(this.viewer.zoomLevel / 1.25);
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.key === " " && held) {
        this.setMode(held);
        held = null;
      }
    });
  }
}
