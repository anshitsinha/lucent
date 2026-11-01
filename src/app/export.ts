import type { OutputFormat } from "./settings";

const MIME: Record<OutputFormat, string> = { png: "image/png", jpeg: "image/jpeg", webp: "image/webp" };
const EXT: Record<OutputFormat, string> = { png: "png", jpeg: "jpg", webp: "webp" };

export function outputName(input: string, scale: number, format: OutputFormat): string {
  const base = input.replace(/\.[^.]+$/, "") || "image";
  const s = Number.isInteger(scale) ? `${scale}x` : `${scale.toFixed(2)}x`;
  return `${base}-lucent-${s}.${EXT[format]}`;
}

export function encode(canvas: HTMLCanvasElement, format: OutputFormat, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    if (format === "jpeg") {
      // JPEG has no alpha: flatten onto white instead of the browser's black
      const flat = document.createElement("canvas");
      flat.width = canvas.width;
      flat.height = canvas.height;
      const ctx = flat.getContext("2d")!;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, flat.width, flat.height);
      ctx.drawImage(canvas, 0, 0);
      canvas = flat;
    }
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("The browser could not encode an image this large."))),
      MIME[format],
      quality / 100,
    );
  });
}

export function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

interface DirHandle {
  getFileHandle(name: string, opts: { create: boolean }): Promise<{ createWritable(): Promise<{ write(b: Blob): Promise<void>; close(): Promise<void> }> }>;
}

/** Ask for a folder to save a batch into (Chromium); null if unsupported or cancelled. */
export async function pickFolder(): Promise<DirHandle | null> {
  const w = window as unknown as { showDirectoryPicker?: (o: object) => Promise<DirHandle> };
  if (!w.showDirectoryPicker) return null;
  try {
    return await w.showDirectoryPicker({ mode: "readwrite", id: "lucent-export" });
  } catch {
    return null;
  }
}

export async function saveTo(dir: DirHandle | null, blob: Blob, name: string) {
  if (!dir) return download(blob, name);
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(blob);
  await w.close();
}
