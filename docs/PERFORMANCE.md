# Performance notes

Every number here was measured, with the setup that produced it. Reproduce
with `npm run bench` (a real GPU in headless Chromium, see `vitest.config.ts`).

**Setup:** NVIDIA Turing laptop GPU, Chromium (Playwright), Windows 11.
Workload: `realesr-general-x4v3` (34 convs, 64 channels) on one
304×304 region — a 256 px tile plus its 24 px context apron, which is what
the app dispatches per tile. 2.36 MFLOP per region pixel, 218 GFLOP per tile.

## The conv kernel, step by step

| kernel | ms / tile | TFLOP/s | vs. first |
| --- | ---: | ---: | ---: |
| f32, weights read from storage, 1 px × 32 ch per thread | 377 | 0.59 | 1.0× |
| f32, 4 px × 16 ch per thread (`py=4, og=4`) | 185 | 1.21 | 2.0× |
| + weights staged in workgroup memory (`ws`) | 125 | 1.79 | 3.0× |
| + fp16 storage and math (`shader-f16`) | **71.5** | **3.13** | **5.3×** |

What mattered:

- **Register blocking, the right shape.** Each thread keeps `py × og`
  accumulators. Fewer output channels per thread (`og=4`) and more rows
  (`py=4`) won: a weight block is loaded once and reused for 4 pixels, while
  `og=16` blew the register budget and ran 2–5× slower.
- **Weights in workgroup memory.** Every thread in a workgroup reads the
  same weight block at the same time; staging the 36 blocks per input group
  in shared memory (one cooperative load) beat relying on the cache
  broadcasting storage reads by 1.48×.
- **fp16.** Halves activation traffic and shared-memory footprint, and
  Turing runs packed fp16 at twice the fp32 rate: 1.75× on top. Accuracy
  cost on the final image: at most 2/255 vs. the fp32 CPU reference
  (fp32 matches it exactly, 0/255) — see `test/network.test.ts`.

## Accuracy

| check | result |
| --- | --- |
| final conv features, f32 vs CPU reference | max abs error 2e-6 |
| final conv features, f16 vs CPU reference | max abs error 7e-3 |
| full 4× pipeline, f32 vs CPU reference (rgba8) | 0/255 |
| full 4× pipeline, f16 vs CPU reference (rgba8) | ≤ 2/255 |
| 16 px tiles vs one tile (24 px apron, noisy image) | ≤ 2/255, mean 0.08 |

With a 16 px apron the tile test showed up to 15/255 at seams on a noisy
image (the 34-conv network sees ±34 px); 24 px brings it to ≤ 2/255 for
~10% more work per tile than 16 px.

## End to end (the app)

`npm run build && npm run smoke` — the built app in Brave (Chromium),
same GPU, fp16, Standard model, default adjustments. Times are from
choosing the image to the last tile painted, as the status bar reports.

| input | output | 256 px tiles | 384 px | **512 px (default)** |
| --- | --- | ---: | ---: | ---: |
| 640×624 (0.4 MP) | 4×, 2560×2496 | 0.7 s | 0.5 s | **0.5 s** |
| 2560×2496 (6.4 MP) | 2×, 25.6 MP | 14 s | 9.1 s | **6.9 s** |

Tile size mattered far more than the apron arithmetic predicts (1.41× →
1.20× overhead): each tile is a submit plus a readback round trip, and
fewer, bigger tiles keep the GPU fed. The enhancer picks the largest of
512/384/256 px whose buffers fit `maxStorageBufferBindingSize`, so small
devices still work, with smaller tiles.
