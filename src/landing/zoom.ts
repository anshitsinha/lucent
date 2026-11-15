/**
 * "Zoom. Enhance." Scrolling through the section zooms from the whole photo
 * to 16x on the visor. Left of the centre line: the browser's own (bilinear)
 * upscale of the original. Right: lucent's 4x and 16x renders, layered over
 * the regions they cover.
 */
import crops from "../../public/landing/crops.json";

const load = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (a: number, b: number, v: number) => {
  const t = clamp((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};

export async function startZoom() {
  const section = document.getElementById("zoom")!;
  const canvas = document.getElementById("zoom-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const hudZoom = document.getElementById("hud-zoom")!;
  const hudCoords = document.getElementById("hud-coords")!;
  const reticle = document.getElementById("hud-reticle")!;
  const words = ["w1", "w2", "w3"].map((id) => document.getElementById(id)!);
  const caption = document.getElementById("zoom-caption")!;
  const nav = document.getElementById("nav")!;

  const [base, hero, visor] = await Promise.all([
    load("/samples/apollo11-aldrin.jpg"),
    load("/landing/hero-4x.webp"),
    load("/landing/visor-16x.webp"),
  ]);
  const layers = [
    { img: hero, ...crops.hero },
    { img: visor, ...crops.visor },
  ];
  const start = { x: base.width / 2, y: base.height / 2 };
  const target = { x: crops.visor.x + crops.visor.w / 2, y: crops.visor.y + crops.visor.h / 2 };

  let W = 0;
  let H = 0;
  let dpr = 1;
  const resize = () => {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    draw();
  };

  function progress() {
    const r = section.getBoundingClientRect();
    return clamp(-r.top / (r.height - innerHeight));
  }

  function draw() {
    const p = progress();
    const r = section.getBoundingClientRect();
    // the HUD owns the screen while the section is pinned
    nav.classList.toggle("hide", r.top <= 1 && r.bottom >= innerHeight - 1);
    const zp = smooth(0.06, 0.84, p);
    const zoom = Math.pow(16, zp);
    const pan = smooth(0.0, 0.45, zp);
    const cx = start.x + (target.x - start.x) * pan;
    const cy = start.y + (target.y - start.y) * pan;
    const k = Math.min(W / base.width, H / base.height) * 0.92 * zoom; // css px per source px

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * (W / 2 - cx * k), dpr * (H / 2 - cy * k));
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const half = W / 2;
    // left: what the browser does with the original
    ctx.save();
    ctx.beginPath();
    ctx.rect(cx - half / k, -1e5, half / k, 2e5);
    ctx.clip();
    ctx.drawImage(base, 0, 0);
    ctx.restore();
    // right: lucent
    ctx.save();
    ctx.beginPath();
    ctx.rect(cx, -1e5, 1e5, 2e5);
    ctx.clip();
    ctx.drawImage(base, 0, 0);
    for (const l of layers) ctx.drawImage(l.img, l.x, l.y, l.w, l.h);
    ctx.restore();

    // divider
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const x = Math.round(half * dpr);
    const grd = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grd.addColorStop(0, "rgba(255,112,51,0)");
    grd.addColorStop(0.5, "rgba(255,112,51,0.95)");
    grd.addColorStop(1, "rgba(255,112,51,0)");
    ctx.fillStyle = grd;
    ctx.fillRect(x - dpr, 0, 2 * dpr, canvas.height);

    hudZoom.textContent = `${zoom.toFixed(1)}×`;
    hudCoords.textContent = `X ${cx.toFixed(1)} · Y ${cy.toFixed(1)}`;
    reticle.style.setProperty("--ret", String(1 - smooth(0.5, 0.8, p)));
    words[0].classList.toggle("on", p > 0.06 && p < 0.22);
    words[1].classList.toggle("on", p > 0.34 && p < 0.5);
    words[2].classList.toggle("on", p > 0.62 && p < 0.78);
    caption.classList.toggle("on", p > 0.86);
  }

  let queued = false;
  const request = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      draw();
    });
  };
  new ResizeObserver(resize).observe(canvas);
  window.addEventListener("scroll", request, { passive: true });
  resize();
}
