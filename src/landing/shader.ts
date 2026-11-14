/**
 * Hero background: an iridescent field of crystal facets (Voronoi cells over
 * domain-warped noise) that drifts, follows the cursor and dims as you scroll.
 * WebGL1, rendered at reduced resolution, paused when the tab is hidden.
 */

const VERT = `attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 r;
uniform float t;
uniform vec2 m;
uniform float s;

float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2 h22(vec2 p) { float n = h21(p); return vec2(n, h21(p + n)); }

float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), u.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5; }
  return v;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * r) / r.y;
  vec2 mouse = (m - 0.5) * vec2(r.x / r.y, -1.0);
  vec2 p = uv * 1.5 + mouse * 0.08;

  float w = fbm(p * 1.3 + vec2(t * 0.04, -t * 0.03));
  float n = fbm(p * 1.1 + w * 1.6 + vec2(-t * 0.02, t * 0.025));

  // crystal facets, gently drifting and warped by the noise
  vec2 g = p * 2.6 + vec2(0.0, s * 1.5) + (w - 0.5) * 0.6;
  vec2 gi = floor(g), gf = fract(g);
  float d1 = 8.0, d2 = 8.0; float id = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 o = vec2(float(x), float(y));
    vec2 c = h22(gi + o);
    c = 0.5 + 0.42 * sin(t * 0.25 + 6.2831 * c);
    float d = length(o + c - gf);
    if (d < d1) { d2 = d1; d1 = d; id = h21(gi + o); }
    else if (d < d2) { d2 = d; }
  }
  float edge = 1.0 - smoothstep(0.0, 0.05, d2 - d1);

  vec3 iri = 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + n * 1.4 + id * 0.35 + t * 0.02));
  iri = mix(iri, vec3(0.36, 0.49, 1.0), 0.35);

  // light pools: top right, and wherever the cursor is
  float pool = smoothstep(1.5, 0.0, length(uv - vec2(0.45, 0.22)));
  float cur = exp(-3.2 * length(uv - mouse));
  float lift = pool * 0.9 + cur * 0.7;

  vec3 col = vec3(0.02, 0.023, 0.04);
  col += iri * (0.09 + 0.5 * n * n) * lift;
  col += iri * edge * 0.32 * lift;
  col += vec3(0.48, 0.95, 1.0) * edge * cur * 0.28;

  col *= 1.0 - 0.55 * clamp(s, 0.0, 1.0);          // dim as the page scrolls
  col *= 1.0 - 0.35 * dot(uv * 0.8, uv * 0.8);      // vignette
  gl_FragColor = vec4(col, 1.0);
}
`;

export function startBackground(canvas: HTMLCanvasElement) {
  const gl = canvas.getContext("webgl", { antialias: false, alpha: false, powerPreference: "low-power" });
  if (!gl) return; // CSS gradient fallback stays
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const prog = gl.createProgram()!;
  try {
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  } catch (e) {
    console.warn("lucent: background shader failed", e);
    return;
  }
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const uR = gl.getUniformLocation(prog, "r");
  const uT = gl.getUniformLocation(prog, "t");
  const uM = gl.getUniformLocation(prog, "m");
  const uS = gl.getUniformLocation(prog, "s");

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mouse = { x: 0.7, y: 0.35, tx: 0.7, ty: 0.35 };
  window.addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX / innerWidth;
    mouse.ty = e.clientY / innerHeight;
  });

  const resize = () => {
    // half resolution: it's a soft background, and this keeps it cheap
    const scale = Math.min(1, (window.devicePixelRatio || 1) * 0.5);
    canvas.width = Math.round(innerWidth * scale);
    canvas.height = Math.round(innerHeight * scale);
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  resize();
  window.addEventListener("resize", resize);

  const t0 = performance.now();
  const frame = () => {
    mouse.x += (mouse.tx - mouse.x) * 0.06;
    mouse.y += (mouse.ty - mouse.y) * 0.06;
    gl.uniform2f(uR, canvas.width, canvas.height);
    gl.uniform1f(uT, reduced ? 12 : (performance.now() - t0) / 1000);
    gl.uniform2f(uM, mouse.x, mouse.y);
    gl.uniform1f(uS, Math.min(1.5, scrollY / innerHeight));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!reduced && !document.hidden) requestAnimationFrame(frame);
  };
  document.addEventListener("visibilitychange", () => !document.hidden && requestAnimationFrame(frame));
  if (reduced) window.addEventListener("scroll", () => requestAnimationFrame(frame), { passive: true });
  requestAnimationFrame(frame);
}
