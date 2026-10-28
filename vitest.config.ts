import { existsSync } from "node:fs";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

/**
 * Kernel tests have to run on a real GPU, so they run in a real browser
 * (vitest browser mode + Playwright). Rather than downloading Playwright's
 * own Chromium, use whatever Chromium-based browser is installed - set
 * LUCENT_BROWSER to override.
 */
function findBrowser(): string | undefined {
  if (process.env.LUCENT_BROWSER) return process.env.LUCENT_BROWSER;
  const local = process.env.LOCALAPPDATA ?? "";
  const candidates = [
    // Windows
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    `${local}\\Google\\Chrome\\Application\\chrome.exe`,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
    // macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    // Linux
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  return candidates.find((p) => existsSync(p));
}

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts", "bench/**/*.test.ts"],
    testTimeout: 120_000,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          executablePath: findBrowser(),
          // headless Chromium leaves WebGPU off unless asked
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
        },
      }),
      instances: [{ browser: "chromium" }],
    },
  },
});
