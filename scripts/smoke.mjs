#!/usr/bin/env node
/**
 * End-to-end smoke test of the built app in a real (headless) Chromium with
 * WebGPU: load the sample, wait for enhancement, check the export, and save
 * screenshots to smoke-out/.
 *
 *   npm run build && npm run smoke
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";
import { preview } from "vite";

const OUT = "smoke-out";
mkdirSync(OUT, { recursive: true });

const local = process.env.LOCALAPPDATA ?? "";
const executablePath = [
  process.env.LUCENT_BROWSER,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  `${local}\\Google\\Chrome\\Application\\chrome.exe`,
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((p) => p && existsSync(p));

// SMOKE_URL=https://... tests a deployment instead of a local preview of dist/
const server = process.env.SMOKE_URL ? null : await preview({ preview: { port: 4317, strictPort: false }, logLevel: "warn" });
const url = process.env.SMOKE_URL ?? server.resolvedUrls.local[0];
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
});
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failed = true;
};

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, acceptDownloads: true });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url);
  await page.waitForFunction(() => document.getElementById("gpu-badge")?.classList.contains("ok"), null, { timeout: 15000 });
  const badge = await page.textContent("#gpu-badge");
  // Playwright's bundled Chromium only offers SwiftShader: far too slow to be a real test
  if (/swiftshader/i.test(badge ?? "")) throw new Error(`software WebGPU adapter (${badge}); set LUCENT_BROWSER to a Chrome/Edge/Brave binary`);
  check(true, `WebGPU up: ${badge}`);
  await page.screenshot({ path: `${OUT}/01-empty.png` });

  for (const scale of ["4", "2"]) {
    await page.evaluate(() => localStorage.clear());
    if (scale === "4") await page.click("#sample-btn");
    await page.click(`#scale-seg [data-scale="${scale}"]`);
    // the first status may be from the previous scale; wait for a fresh "Done"
    await page.waitForFunction(() => /Enhancing|Loading/.test(document.getElementById("status-text")?.textContent ?? ""), null, { timeout: 10000 }).catch(() => {});
    await page.waitForFunction(() => document.getElementById("status-text")?.textContent?.startsWith("Done"), null, { timeout: 120000 });
    const status = `${await page.textContent("#status-text")} (${await page.textContent("#status-stats")})`;
    check(true, `${scale}x: ${status}`);
  }
  await page.screenshot({ path: `${OUT}/02-split.png` });

  // zoom to 100% on the visor and look closer
  await page.mouse.move(1440 / 2 - 150, 300);
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -240);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/03-zoomed.png` });

  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 60000 }), page.click("#export-btn")]);
  const path = `${OUT}/${download.suggestedFilename()}`;
  await download.saveAs(path);
  check(download.suggestedFilename() === "apollo11-aldrin-lucent-2x.png", `export: ${download.suggestedFilename()}`);

  // a large image: the sample upsized to 2560 x 2496 (6.4 MP) at 2x -> 25.6 MP out
  await page.evaluate(async () => {
    const img = await createImageBitmap(await (await fetch("/samples/apollo11-aldrin.jpg")).blob());
    const c = new OffscreenCanvas(2560, 2496);
    c.getContext("2d").drawImage(img, 0, 0, 2560, 2496);
    const blob = await c.convertToBlob({ type: "image/png" });
    window.lucent.addFiles([new File([blob], "large.png", { type: "image/png" })]);
  });
  await page.click("#queue .qitem:last-child");
  await page.waitForFunction(() => /Enhancing/.test(document.getElementById("status-text")?.textContent ?? ""), null, { timeout: 20000 });
  await page.waitForFunction(() => document.getElementById("status-text")?.textContent?.startsWith("Done"), null, { timeout: 300000 });
  check(true, `6.4 MP at 2x: ${await page.textContent("#status-text")} (${await page.textContent("#status-stats")})`);
  await page.screenshot({ path: `${OUT}/05-batch.png` });

  // mobile layout
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await phone.goto(url);
  await phone.waitForFunction(() => document.getElementById("gpu-badge")?.classList.contains("ok"), null, { timeout: 15000 });
  await phone.click("#sample-btn");
  await phone.waitForFunction(() => document.getElementById("status-text")?.textContent?.startsWith("Done"), null, { timeout: 120000 });
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  check(!overflow, "phone: no horizontal overflow");
  await phone.screenshot({ path: `${OUT}/04-phone.png` });

  check(errors.length === 0, `console errors: ${errors.length ? errors.join(" | ") : "none"}`);
  writeFileSync(`${OUT}/errors.txt`, errors.join("\n"));
} catch (e) {
  check(false, String(e));
} finally {
  await browser.close();
  if (server) await new Promise((r) => server.httpServer.close(r));
}
process.exit(failed ? 1 : 0);
