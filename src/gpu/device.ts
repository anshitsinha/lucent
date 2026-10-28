/** WebGPU device acquisition with the limits lucent needs. */

export class WebGpuUnavailableError extends Error {}

export interface GpuContext {
  device: GPUDevice;
  adapterInfo: string;
  maxTextureSize: number;
}

let pending: Promise<GpuContext> | null = null;

export function getGpu(): Promise<GpuContext> {
  pending ??= acquire().catch((e) => {
    pending = null;
    throw e;
  });
  return pending;
}

async function acquire(): Promise<GpuContext> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    throw new WebGpuUnavailableError("This browser does not support WebGPU.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new WebGpuUnavailableError("No WebGPU adapter is available (GPU blocklisted or disabled?).");
  const a = adapter.limits;
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16") ? ["shader-f16"] : [];
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: a.maxStorageBufferBindingSize,
      maxBufferSize: a.maxBufferSize,
      maxTextureDimension2D: a.maxTextureDimension2D,
      maxComputeWorkgroupStorageSize: a.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: a.maxComputeInvocationsPerWorkgroup,
    },
  });
  device.addEventListener("uncapturederror", (e) => {
    console.error(`lucent: WebGPU error: ${(e as GPUUncapturedErrorEvent).error.message}`);
  });
  device.lost.then((info) => {
    pending = null;
    console.warn(`lucent: WebGPU device lost (${info.reason}): ${info.message}`);
  });
  const info = adapter.info;
  const adapterInfo = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" / ") || "unknown GPU";
  return { device, adapterInfo, maxTextureSize: a.maxTextureDimension2D };
}
