import "./landing.css";
import { startDemo } from "./demo";
import { startBackground } from "./shader";
import { startZoom } from "./zoom";

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fine = matchMedia("(pointer: fine)").matches;

startBackground(document.getElementById("bg") as HTMLCanvasElement);

// --- reveal on scroll (with a small stagger inside each parent), count-ups
function countUp(el: HTMLElement) {
  const to = Number(el.dataset.count);
  const decimals = Number(el.dataset.decimals ?? 0);
  const suffix = el.dataset.suffix ?? "";
  if (reduced || !Number.isFinite(to)) return;
  const t0 = performance.now();
  const dur = 1400;
  const tick = () => {
    const f = Math.min(1, (performance.now() - t0) / dur);
    const e = 1 - Math.pow(1 - f, 4);
    el.textContent = `${(to * e).toFixed(decimals)}${suffix}`;
    if (f < 1) requestAnimationFrame(tick);
  };
  tick();
}

const io = new IntersectionObserver(
  (entries) => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      const el = en.target as HTMLElement;
      el.classList.add("in");
      el.querySelectorAll<HTMLElement>("[data-count]").forEach(countUp);
      if (el.matches("[data-count]")) countUp(el);
      io.unobserve(el);
    }
  },
  { threshold: 0.15, rootMargin: "0px 0px -6% 0px" },
);
document.querySelectorAll<HTMLElement>(".reveal").forEach((el) => {
  const siblings = [...(el.parentElement?.children ?? [])].filter((s) => s.classList.contains("reveal"));
  el.style.setProperty("--d", `${Math.min(siblings.indexOf(el), 6) * 0.08}s`);
  io.observe(el);
});
const bars = document.getElementById("bars");
if (bars) io.observe(bars);
document.querySelectorAll<HTMLElement>(".num b[data-count]").forEach((b) => io.observe(b));

// --- "Enhance." resolves under a scanning bar
const enhance = document.querySelector<HTMLElement>(".enhance");
if (enhance) {
  const bar = document.createElement("span");
  bar.className = "scanbar";
  enhance.append(bar);
  const sweep = () => {
    if (reduced) {
      enhance.style.setProperty("--scan", "100%");
      enhance.style.setProperty("--scan-o", "0");
      return;
    }
    const t0 = performance.now();
    const dur = 1500;
    const tick = () => {
      const f = Math.min(1, (performance.now() - t0) / dur);
      const e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
      enhance.style.setProperty("--scan", `${(e * 100).toFixed(2)}%`);
      enhance.style.setProperty("--scan-o", f < 1 ? "1" : "0");
      if (f < 1) requestAnimationFrame(tick);
    };
    tick();
  };
  setTimeout(sweep, 650);
  enhance.addEventListener("pointerenter", sweep);
}

// --- nav state
const nav = document.getElementById("nav")!;
const onScroll = () => nav.classList.toggle("scrolled", scrollY > 24);
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

// --- the 34 conv layers, as a bar strip
const layers = document.getElementById("layers");
if (layers) {
  for (let i = 0; i < 34; i++) {
    const b = document.createElement("i");
    const h = i === 0 || i === 33 ? 0.55 : 0.75 + 0.25 * Math.sin(i * 1.7) ** 2;
    b.style.setProperty("--h", `${(h * 100).toFixed(0)}%`);
    b.style.setProperty("--i", String(i));
    layers.append(b);
  }
}

// --- pointer toys (desktop only)
if (fine && !reduced) {
  const glow = document.querySelector<HTMLElement>(".cursor-glow")!;
  window.addEventListener("pointermove", (e) => {
    glow.style.setProperty("--mx", `${e.clientX}px`);
    glow.style.setProperty("--my", `${e.clientY}px`);
  });

  for (const el of document.querySelectorAll<HTMLElement>(".spot")) {
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      el.style.setProperty("--x", `${e.clientX - r.left}px`);
      el.style.setProperty("--y", `${e.clientY - r.top}px`);
    });
  }

  for (const el of document.querySelectorAll<HTMLElement>(".magnetic")) {
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      el.style.transform = `translate(${dx * 0.22}px, ${dy * 0.32}px)`;
    });
    el.addEventListener("pointerleave", () => (el.style.transform = ""));
  }

  const frame = document.querySelector<HTMLElement>(".tilt");
  const demo = document.getElementById("demo");
  if (frame && demo) {
    demo.addEventListener("pointermove", (e) => {
      const r = demo.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      frame.style.setProperty("--ry", `${x * 10}deg`);
      frame.style.setProperty("--rx", `${-y * 8}deg`);
    });
    demo.addEventListener("pointerleave", () => {
      frame.style.removeProperty("--ry");
      frame.style.removeProperty("--rx");
    });
  }
}

void startZoom();
void startDemo();
