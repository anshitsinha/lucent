/**
 * Before/after viewer. Everything is drawn in output-pixel space:
 *   screen = (p - pan) * zoom          (CSS px)
 * The original is drawn scaled up to output size on the left of the split,
 * the enhanced canvas on the right. Wheel zooms at the cursor, drag pans,
 * dragging near the split line moves it, two fingers pinch-zoom.
 */

export type CompareMode = "split" | "original" | "enhanced";

export interface ViewerSource {
  original: CanvasImageSource & { width: number; height: number };
  enhanced: HTMLCanvasElement;
}

const MIN_ZOOM = 0.02;
const MAX_ZOOM = 32;

export class Viewer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private src: ViewerSource | null = null;
  private zoom = 1;
  private panX = 0;
  private panY = 0;
  private fitted = true;
  /** split position, fraction of view width */
  split = 0.5;
  mode: CompareMode = "split";
  /** tile currently being processed (output px) */
  activeTile: { x: number; y: number; width: number; height: number } | null = null;
  onZoom?: (zoom: number) => void;

  private dirty = true;
  private pointers = new Map<number, { x: number; y: number }>();
  private drag: { kind: "pan" | "split"; lastX: number; lastY: number } | null = null;
  private pinch: { dist: number; zoom: number } | null = null;

  constructor(private host: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewer-canvas";
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    new ResizeObserver(() => this.resize()).observe(host);
    this.bindInput();
    const loop = () => {
      if (this.dirty || this.activeTile) this.draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  setSource(src: ViewerSource | null, keepView = false) {
    const sizeChanged = !this.src || !src || this.src.enhanced.width !== src.enhanced.width || this.src.enhanced.height !== src.enhanced.height;
    this.src = src;
    if (!keepView || sizeChanged) this.fit();
    this.invalidate();
  }

  invalidate() {
    this.dirty = true;
  }

  get zoomLevel() {
    return this.zoom;
  }

  /** Centre of the view in output px - where to start enhancing. */
  get focus(): { x: number; y: number } {
    const { w, h } = this.viewSize();
    return { x: this.panX + w / 2 / this.zoom, y: this.panY + h / 2 / this.zoom };
  }

  fit() {
    this.fitted = true;
    if (!this.src) return;
    const { w, h } = this.viewSize();
    const iw = this.src.enhanced.width;
    const ih = this.src.enhanced.height;
    const pad = 24;
    this.zoom = Math.min(1, Math.min((w - pad * 2) / iw, (h - pad * 2) / ih));
    this.panX = iw / 2 - w / 2 / this.zoom;
    this.panY = ih / 2 - h / 2 / this.zoom;
    this.onZoom?.(this.zoom);
    this.invalidate();
  }

  /** Set zoom about a screen point (defaults to view centre). */
  zoomTo(z: number, sx?: number, sy?: number) {
    const { w, h } = this.viewSize();
    sx ??= w / 2;
    sy ??= h / 2;
    const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
    const px = this.panX + sx / this.zoom;
    const py = this.panY + sy / this.zoom;
    this.zoom = nz;
    this.panX = px - sx / nz;
    this.panY = py - sy / nz;
    this.fitted = false;
    this.onZoom?.(nz);
    this.invalidate();
  }

  private viewSize() {
    const r = this.host.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  private resize() {
    const { w, h } = this.viewSize();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    if (this.fitted) this.fit();
    this.invalidate();
  }

  private splitX() {
    return this.viewSize().w * this.split;
  }

  private bindInput() {
    const c = this.canvas;
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const r = c.getBoundingClientRect();
        const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
        this.zoomTo(this.zoom * factor, e.clientX - r.left, e.clientY - r.top);
      },
      { passive: false },
    );
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      const r = c.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      this.pointers.set(e.pointerId, { x, y });
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.zoom };
        this.drag = null;
        return;
      }
      const nearSplit = this.mode === "split" && Math.abs(x - this.splitX()) < (e.pointerType === "touch" ? 22 : 10);
      this.drag = { kind: nearSplit ? "split" : "pan", lastX: x, lastY: y };
      c.classList.toggle("grabbing", !nearSplit);
    });
    c.addEventListener("pointermove", (e) => {
      const r = c.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, { x, y });
      if (this.pinch && this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        this.zoomTo((this.pinch.zoom * d) / this.pinch.dist, (a.x + b.x) / 2, (a.y + b.y) / 2);
        return;
      }
      if (!this.drag) {
        const near = this.mode === "split" && Math.abs(x - this.splitX()) < 10;
        c.style.cursor = near ? "ew-resize" : "grab";
        return;
      }
      if (this.drag.kind === "split") {
        this.split = Math.min(1, Math.max(0, x / this.viewSize().w));
      } else {
        this.panX -= (x - this.drag.lastX) / this.zoom;
        this.panY -= (y - this.drag.lastY) / this.zoom;
        this.fitted = false;
      }
      this.drag.lastX = x;
      this.drag.lastY = y;
      this.invalidate();
    });
    const end = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (this.pointers.size < 2) this.pinch = null;
      this.drag = null;
      c.classList.remove("grabbing");
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("dblclick", () => (this.fitted ? this.zoomTo(1) : this.fit()));
  }

  private draw() {
    this.dirty = false;
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.src) return;
    const { w, h } = this.viewSize();
    const { original, enhanced } = this.src;
    const ow = enhanced.width;
    const oh = enhanced.height;
    const z = this.zoom * dpr;
    const dx = -this.panX * z;
    const dy = -this.panY * z;
    // past 200% show real pixels instead of a smoothed blur
    const smooth = this.zoom < 2;

    // checkerboard under transparent images
    ctx.fillStyle = getComputedStyle(this.host).getPropertyValue("--checker").trim() || "#222";
    ctx.fillRect(dx, dy, ow * z, oh * z);

    const drawPart = (img: CanvasImageSource & { width: number; height: number }, clipX0: number, clipX1: number) => {
      if (clipX1 <= clipX0) return;
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX0 * dpr, 0, (clipX1 - clipX0) * dpr, h * dpr);
      ctx.clip();
      ctx.imageSmoothingEnabled = smooth;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, img.width, img.height, dx, dy, ow * z, oh * z);
      ctx.restore();
    };

    const sx = this.mode === "split" ? this.splitX() : this.mode === "original" ? w : 0;
    drawPart(original, 0, sx);
    drawPart(enhanced, sx, w);

    if (this.activeTile) {
      const t = this.activeTile;
      const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 160);
      ctx.strokeStyle = `rgba(255, 112, 51, ${0.35 + 0.5 * pulse})`;
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeRect(dx + t.x * z, dy + t.y * z, t.width * z, t.height * z);
    }

    if (this.mode === "split") {
      const x = Math.round(sx * dpr) + 0.5;
      ctx.fillStyle = "rgba(236,231,223,0.92)";
      ctx.fillRect(x - dpr / 2, 0, dpr, h * dpr);
      // handle
      const cy = (h / 2) * dpr;
      ctx.beginPath();
      ctx.arc(x, cy, 14 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(20,20,19,0.85)";
      ctx.fill();
      ctx.strokeStyle = "rgba(236,231,223,0.92)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.stroke();
      ctx.beginPath();
      for (const s of [-1, 1]) {
        ctx.moveTo(x + s * 4 * dpr, cy - 5 * dpr);
        ctx.lineTo(x + s * 8 * dpr, cy);
        ctx.lineTo(x + s * 4 * dpr, cy + 5 * dpr);
      }
      ctx.stroke();
      // labels
      ctx.font = `${11 * dpr}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      const label = (text: string, lx: number, align: CanvasTextAlign) => {
        ctx.textAlign = align;
        const m = ctx.measureText(text).width;
        const bx = align === "right" ? lx - m - 8 * dpr : lx;
        ctx.fillStyle = "rgba(12,12,11,0.68)";
        ctx.fillRect(bx, 12 * dpr, m + 16 * dpr, 20 * dpr);
        ctx.fillStyle = "rgba(236,231,223,0.95)";
        ctx.fillText(text, align === "right" ? lx : lx + 8 * dpr, 16.5 * dpr);
      };
      if (sx > 90) label("Original", x - 12 * dpr, "right");
      if (w - sx > 90) label("Enhanced", x + 12 * dpr, "left");
    }
  }
}
