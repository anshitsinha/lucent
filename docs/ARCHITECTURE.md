# Architecture

## The model

Real-ESRGAN's `SRVGGNetCompact`: a plain stack of 3×3 convolutions at the
input resolution, which suits a GPU well (no attention, no resampling inside).

```
x ─▶ conv(3→64) ─ PReLU ─▶ [conv(64→64) ─ PReLU] × N ─▶ conv(64→48)
   ─▶ pixel_shuffle(4) ─▶ + nearest_upsample(x, 4)
```

N = 32 for the general models (34 convs, 1.2 M parameters) and 16 for the
anime model. Cost: about 2.4 MFLOP per input pixel.

**Denoise strength** follows Real-ESRGAN's inference script: weights are
`t·θ_general + (1−t)·θ_wdn`, blended on the CPU (~1 M floats) and written
into the existing GPU buffers (`GpuSrvgg.load`), so dragging the slider
allocates nothing.

## Weights

`scripts/convert-weights.mjs` reads the upstream `.pth` files directly. A
`.pth` is a zip holding a pickle plus raw tensor blobs, and the script carries
a ~100-line pickle VM that handles only the opcodes `torch.save` emits for a
state dict. It verifies the upstream sha256, checks contiguity and writes fp16
safetensors. The browser decodes fp16 through a 64K lookup table.

## Data layout

Activations are `array<vec4>`, channels packed by 4 and stored as planes:
`index = (group·H + y)·W + x`. Conv weights are `array<mat4x4>` indexed
`[outGroup][inGroup][tap]`, so one `mat4x4 × vec4` is 16 MACs mapping 4 input
channels to 4 output channels.

## The conv kernel (`src/gpu/kernels.ts`)

Generated per layer shape, with every loop over taps, rows and channel
groups unrolled:

- a workgroup of 16×8 threads covers 16×32 output pixels and 16 output
  channels (`og = 4` groups; the remaining groups go to `workgroup_id.z`)
- for each input group: cooperatively load an 18×34 input tile and that
  group's 36 weight blocks into workgroup memory, then barrier
- each thread accumulates 4 rows × 4 channel groups in registers, so a weight
  block is loaded once and used 4 times
- bias is the initial accumulator value, and PReLU is applied in the epilogue

The tiling was chosen by `bench/conv-sweep.test.ts`, not guessed. See
[PERFORMANCE.md](PERFORMANCE.md).

## Tiling (`src/gpu/enhancer.ts`)

Images are split into near-equal tiles of up to 512 px (smaller if the
device's buffer limits require). Each tile is processed with a 24 px apron of
real neighbouring pixels, and only its centre is kept. Zero padding therefore
appears only at the true image border, which is how the network was trained.
The network's receptive field is ±34 px; 24 px of context keeps seams within
2/255 (tested).

One command buffer per tile: pack → 34 convs → compose → finish → copy to
a readback buffer. Two readback buffers alternate, so the GPU works on tile
n+1 while tile n is mapped and painted. Tiles run nearest-first to a focus
point, which the app sets to the centre of the view.

## Beyond 4×: chained passes (`src/gpu/upscale.ts`)

The network is 4× by construction. For larger scales `planPasses` splits
the scale evenly, √s per pass (8× = 2.83 × 2.83, 16× = 4 × 4). That costs
about the same as running the second pass at full 4×, and the second pass
starts from an AI-upscaled image instead of a bicubic one. Each pass gets its
own weights. The app gives the second pass a quarter of the denoise strength,
because its input is already clean and denoising again over-smooths it.
"Preserve original" applies only in pass 1, where it blends toward the real
original. Sharpen and grain apply only in the last pass, at final
resolution. `EnhanceOptions.size` (independent x/y scale) makes the last
pass land exactly on the requested dimensions.

## Post-processing

- **compose:** reads the 48-channel network output as a 4× image
  (pixel shuffle and skip connection done on the fly), resamples it to the
  target scale with Lanczos-2 (support scaled by the downsampling factor),
  blends toward a Catmull-Rom bicubic upscale (*preserve original*) and
  carries alpha over bicubically
- **finish:** 3×3 unsharp mask at output resolution, luminance-weighted
  hash-noise grain, quantise to rgba8

## The app (`src/app/`)

Plain TypeScript and DOM, with no framework. `App` owns the queue, settings
and run lifecycle. Each settings change aborts the current run (the enhancer
serialises runs), waits for `idle()`, then starts a new run that paints tiles
over the previous output canvas. `Viewer` draws the original (scaled to
output size) and the enhanced canvas into one 2D canvas, split at a
draggable line.
