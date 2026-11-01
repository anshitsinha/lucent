import "./styles.css";
import { App } from "./app/app";
import { WebGpuUnavailableError, getGpu } from "./gpu/device";

async function boot() {
  try {
    const gpu = await getGpu();
    const app = new App(gpu);
    // handy for poking at from the console
    (window as unknown as { lucent: App }).lucent = app;
  } catch (e) {
    document.getElementById("empty")!.hidden = true;
    document.getElementById("unsupported")!.hidden = false;
    const badge = document.getElementById("gpu-badge")!;
    badge.textContent = "no WebGPU";
    for (const id of ["open-btn", "export-btn"]) (document.getElementById(id) as HTMLButtonElement).disabled = true;
    if (!(e instanceof WebGpuUnavailableError)) {
      document.getElementById("unsupported-msg")!.textContent = `WebGPU failed to start: ${(e as Error).message}`;
    }
  }
}

void boot();
