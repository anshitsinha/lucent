/**
 * WGSL for the enhancement pipeline. Everything is generated as strings so the
 * hot loops can be fully unrolled for a given layer shape.
 *
 * Activation layout: array<vec4f>, channels packed by 4, planar by group:
 *   index = (group * H + y) * W + x
 *
 * Conv weights: array<mat4x4f> indexed [outGroup][inGroup][tap], where the
 * matrix maps 4 input channels to 4 output channels (column j = input j).
 */

export interface ConvShape {
  /** input channel groups (channels / 4, rounded up) */
  in4: number;
  /** output channel groups */
  out4: number;
  prelu: boolean;
}

export interface ConvTiling {
  /** threads in x / y */
  tx: number;
  ty: number;
  /** output rows per thread (spaced ty apart) */
  py: number;
  /** output channel groups per thread; workgroups split the rest over z */
  og: number;
}

export const DEFAULT_TILING: ConvTiling = { tx: 16, ty: 8, py: 2, og: 8 };

/** Pick an og that divides out4 (falls back toward smaller splits). */
export function tilingFor(shape: ConvShape, base: ConvTiling = DEFAULT_TILING): ConvTiling {
  let og = Math.min(base.og, shape.out4);
  while (shape.out4 % og !== 0) og--;
  return { ...base, og };
}

export function convWgsl(shape: ConvShape, t: ConvTiling): string {
  const { in4, prelu } = shape;
  const { tx, ty, py, og } = t;
  const sw = tx + 2;
  const sh = ty * py + 2;
  const acc: string[] = [];
  const decl: string[] = [];
  for (let o = 0; o < og; o++) {
    for (let p = 0; p < py; p++) decl.push(`  var a${o}_${p} = bias[og0 + ${o}u];`);
  }
  const body: string[] = [];
  for (let ky = 0; ky < 3; ky++) {
    for (let kx = 0; kx < 3; kx++) {
      const tap = ky * 3 + kx;
      for (let p = 0; p < py; p++) {
        body.push(`    let v${p}_${tap} = tile[(lid.y + ${p * ty + ky}u) * ${sw}u + lid.x + ${kx}u];`);
      }
      for (let o = 0; o < og; o++) {
        body.push(`    { let m = wt[wb + ${o * in4 * 9 + tap}u];`);
        for (let p = 0; p < py; p++) body.push(`      a${o}_${p} += m * v${p}_${tap};`);
        body.push(`    }`);
      }
    }
  }
  for (let p = 0; p < py; p++) {
    acc.push(`  { let y = y0 + lid.y + ${p * ty}u;`);
    acc.push(`    if (x < p.w && y < p.h) {`);
    for (let o = 0; o < og; o++) {
      const v = `a${o}_${p}`;
      if (prelu) {
        acc.push(`      { let s = alpha[og0 + ${o}u]; dst[((og0 + ${o}u) * p.h + y) * p.w + x] = select(${v} * s, ${v}, ${v} > vec4f(0.0)); }`);
      } else {
        acc.push(`      dst[((og0 + ${o}u) * p.h + y) * p.w + x] = ${v};`);
      }
    }
    acc.push(`    }`, `  }`);
  }

  return /* wgsl */ `
struct Params { w: u32, h: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read> wt: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> bias: array<vec4f>;
@group(0) @binding(3) var<storage, read> alpha: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(5) var<uniform> p: Params;

var<workgroup> tile: array<vec4f, ${sw * sh}>;

@compute @workgroup_size(${tx}, ${ty})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let x0 = wg.x * ${tx}u;
  let y0 = wg.y * ${ty * py}u;
  let og0 = wg.z * ${og}u;
  let x = x0 + lid.x;
  _ = alpha[0]; // keeps the binding in the auto layout even without PReLU
${decl.join("\n")}
  for (var ig = 0u; ig < ${in4}u; ig++) {
    for (var i = li; i < ${sw * sh}u; i += ${tx * ty}u) {
      let gx = i32(x0) + i32(i % ${sw}u) - 1;
      let gy = i32(y0) + i32(i / ${sw}u) - 1;
      var v = vec4f(0.0);
      if (gx >= 0 && gx < i32(p.w) && gy >= 0 && gy < i32(p.h)) {
        v = src[(ig * p.h + u32(gy)) * p.w + u32(gx)];
      }
      tile[i] = v;
    }
    workgroupBarrier();
    let wb = (og0 * ${in4}u + ig) * 9u;
${body.join("\n")}
    workgroupBarrier();
  }
${acc.join("\n")}
}
`;
}

/** Copy a region of the source image into a 1-group activation buffer (rgb, 0). */
export const PACK_WGSL = /* wgsl */ `
struct Params { rx: i32, ry: i32, rw: u32, rh: u32 }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.rw || id.y >= p.rh) { return; }
  let c = textureLoad(src, vec2i(p.rx + i32(id.x), p.ry + i32(id.y)), 0);
  dst[id.y * p.rw + id.x] = vec4f(c.rgb, 0.0);
}
`;

/**
 * Pixel-shuffle the network output (+ the nearest-upsampled input, as the
 * model expects), resample from 4x to the target scale with a Lanczos-2
 * filter, blend toward a bicubic upscale by `fidelity`, and carry alpha over
 * bicubically. Writes linear-in-value vec4 for the sharpen pass.
 */
export const COMPOSE_WGSL = /* wgsl */ `
struct Params {
  rx: i32, ry: i32, rw: i32, rh: i32,   // network region, source px
  ox: i32, oy: i32, ow: i32, oh: i32,   // rect to produce, output px
  iw: i32, ih: i32, scale: f32, fidelity: f32,
}
@group(0) @binding(0) var<storage, read> net: array<vec4f>;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(3) var<uniform> p: Params;

fn srcPx(x: i32, y: i32) -> vec4f {
  return textureLoad(src, clamp(vec2i(x, y), vec2i(0), vec2i(p.iw - 1, p.ih - 1)), 0);
}

// 4x network output at region-relative hi-res coords (clamped to the region)
fn hr(X: i32, Y: i32) -> vec3f {
  let xc = clamp(X, 0, p.rw * 4 - 1);
  let yc = clamp(Y, 0, p.rh * 4 - 1);
  let x = xc / 4;
  let y = yc / 4;
  let sub = u32((yc % 4) * 4 + xc % 4);
  let g = sub / 4u;
  let k = sub % 4u;
  let plane = u32(p.rw * p.rh);
  let i = u32(y * p.rw + x);
  let r = net[g * plane + i][k];
  let gg = net[(4u + g) * plane + i][k];
  let b = net[(8u + g) * plane + i][k];
  return vec3f(r, gg, b) + srcPx(p.rx + x, p.ry + y).rgb;
}

fn lanczos2(x: f32) -> f32 {
  let ax = abs(x);
  if (ax < 1e-5) { return 1.0; }
  if (ax >= 2.0) { return 0.0; }
  let px = 3.14159265 * x;
  return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}

fn cubic(x: f32) -> vec4f {   // Catmull-Rom weights for offsets -1, 0, 1, 2
  let x2 = x * x;
  let x3 = x2 * x;
  return vec4f(
    -0.5 * x3 + x2 - 0.5 * x,
    1.5 * x3 - 2.5 * x2 + 1.0,
    -1.5 * x3 + 2.0 * x2 + 0.5 * x,
    0.5 * x3 - 0.5 * x2);
}

fn bicubic(u: f32, v: f32) -> vec4f {
  let fx = floor(u);
  let fy = floor(v);
  let wx = cubic(u - fx);
  let wy = cubic(v - fy);
  let x = i32(fx);
  let y = i32(fy);
  var acc = vec4f(0.0);
  for (var j = 0; j < 4; j++) {
    let row = wx.x * srcPx(x - 1, y + j - 1) + wx.y * srcPx(x, y + j - 1)
            + wx.z * srcPx(x + 1, y + j - 1) + wx.w * srcPx(x + 2, y + j - 1);
    acc += wy[j] * row;
  }
  return acc;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let lx = i32(id.x);
  let ly = i32(id.y);
  if (lx >= p.ow || ly >= p.oh) { return; }
  let ox = f32(p.ox + lx) + 0.5;
  let oy = f32(p.oy + ly) + 0.5;

  // source-space sample position (pixel centres at integers)
  let base = bicubic(ox / p.scale - 0.5, oy / p.scale - 0.5);

  var ai = base.rgb;
  if (p.fidelity < 1.0) {
    let f = 4.0 / p.scale;                       // hi-res px per output px
    let hx = ox * f - 0.5 - f32(p.rx * 4);
    let hy = oy * f - 0.5 - f32(p.ry * 4);
    if (f <= 1.0001) {
      ai = hr(i32(round(hx)), i32(round(hy)));
    } else {
      let r = 2.0 * f;
      var acc = vec3f(0.0);
      var wsum = 0.0;
      for (var Y = i32(ceil(hy - r)); Y <= i32(floor(hy + r)); Y++) {
        let wy = lanczos2((f32(Y) - hy) / f);
        for (var X = i32(ceil(hx - r)); X <= i32(floor(hx + r)); X++) {
          let w = wy * lanczos2((f32(X) - hx) / f);
          acc += w * hr(X, Y);
          wsum += w;
        }
      }
      ai = acc / wsum;
    }
  }
  let rgb = mix(ai, base.rgb, p.fidelity);
  dst[u32(ly * p.ow + lx)] = vec4f(rgb, clamp(base.a, 0.0, 1.0));
}
`;

/**
 * Unsharp mask (3x3) + optional luminance film grain, then quantise to rgba8.
 * Input has a 1px apron around the (iw x ih) block it writes.
 */
export const FINISH_WGSL = /* wgsl */ `
struct Params { iw: u32, ih: u32, gx: u32, gy: u32, sharpen: f32, grain: f32, seed: u32, _p: u32 }
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn at(x: u32, y: u32) -> vec4f { return src[y * (p.iw + 2u) + x]; }

fn hash(x: u32, y: u32) -> f32 {
  var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (p.seed * 0xcb1ab31fu);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967296.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.iw || id.y >= p.ih) { return; }
  let x = id.x + 1u;
  let y = id.y + 1u;
  let c = at(x, y);
  var rgb = c.rgb;
  if (p.sharpen > 0.0) {
    let blur = (4.0 * c.rgb
      + 2.0 * (at(x - 1u, y).rgb + at(x + 1u, y).rgb + at(x, y - 1u).rgb + at(x, y + 1u).rgb)
      + at(x - 1u, y - 1u).rgb + at(x + 1u, y - 1u).rgb + at(x - 1u, y + 1u).rgb + at(x + 1u, y + 1u).rgb) / 16.0;
    rgb += p.sharpen * (c.rgb - blur);
  }
  if (p.grain > 0.0) {
    // two hashes -> roughly gaussian (triangular) noise, mid-tones get the most
    let n = hash(p.gx + id.x, p.gy + id.y) + hash(p.gy + id.y + 7919u, p.gx + id.x) - 1.0;
    let l = dot(rgb, vec3f(0.299, 0.587, 0.114));
    rgb += vec3f(n * p.grain * 0.12 * (0.35 + 2.6 * l * (1.0 - l)));
  }
  dst[id.y * p.iw + id.x] = pack4x8unorm(clamp(vec4f(rgb, c.a), vec4f(0.0), vec4f(1.0)));
}
`;
