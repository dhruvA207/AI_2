// The orb.
//
// One big orb in the centre; a smaller one floats in on the side for each live tool
// call. Point clouds rather than meshes, because points on a sphere naturally crowd
// at the silhouette edge — that is what produces the glowing rim without any glow
// being drawn, and it is the characteristic the reference design gets its look from.
//
// Three independent inputs drive the centre orb, following the split in
// ~/projects/Jarvis/webui/js/core/Hologram.js:
//
//   listening (is your mic open?)      → COLOUR       (muted steel vs live accent)
//   activity  (is the engine busy?)    → ENERGY       (baseline size + turbulence)
//   level     (live amplitude, 0..1)   → DISPLACEMENT (the per-syllable deformation)
//
// The third is the one the reference video actually lives on: loudness buys surface
// turbulence, not just radius. A sphere that only scales with volume reads as a level
// meter wearing a costume, so radius moves very little (1.00 → 1.12) while noise
// displacement moves a lot (0.06 → 0.40).

import { ACTIVITY } from './state.js';

const CAT_VARS = {
  filesystem: '--cat-filesystem',
  web: '--cat-web',
  shell: '--cat-shell',
  screen: '--cat-screen',
  input: '--cat-input',
  code: '--cat-code',
  general: '--cat-general',
};

/* ── 3D value noise ────────────────────────────────────────────────────────── */
const P = new Uint8Array(512);
{
  const perm = [...Array(256).keys()];
  let s = 1337;
  for (let i = 255; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  for (let i = 0; i < 512; i++) P[i] = perm[i & 255];
}
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const lerp = (a, b, t) => a + (b - a) * t;
function grad(h, x, y, z) {
  h &= 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : (h === 12 || h === 14 ? x : z);
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}
function noise3(x, y, z) {
  const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
  x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
  const u = fade(x), v = fade(y), w = fade(z);
  const A = P[X] + Y, AA = P[A] + Z, AB = P[A + 1] + Z;
  const B = P[X + 1] + Y, BA = P[B] + Z, BB = P[B + 1] + Z;
  return lerp(
    lerp(lerp(grad(P[AA], x, y, z), grad(P[BA], x - 1, y, z), u),
         lerp(grad(P[AB], x, y - 1, z), grad(P[BB], x - 1, y - 1, z), u), v),
    lerp(lerp(grad(P[AA + 1], x, y, z - 1), grad(P[BA + 1], x - 1, y, z - 1), u),
         lerp(grad(P[AB + 1], x, y - 1, z - 1), grad(P[BB + 1], x - 1, y - 1, z - 1), u), v),
    w);
}

/** Fibonacci sphere — even point distribution, no polar clustering. */
function fib(n) {
  const p = new Float32Array(n * 3);
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = ga * i;
    p[i * 3] = Math.cos(th) * r;
    p[i * 3 + 1] = y;
    p[i * 3 + 2] = Math.sin(th) * r;
  }
  return p;
}

const N_MAIN = 2600;
const N_TOOL = 240;
const MAIN = fib(N_MAIN);
const TOOL = fib(N_TOOL);

export class OrbRenderer {
  constructor(canvas, state) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.state = state;
    this.reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.level = 0;      // smoothed amplitude
    this.mix = 0;        // 0 = muted steel, 1 = live accent
    this.energy = 0.5;

    // Screen-space record of each tool orb, refreshed every frame so clicks can be
    // hit-tested without duplicating the layout maths.
    this.hits = [];
    this.spawns = new Map();   // id -> spawn timestamp, for the grow-in animation

    this.css = {};
    this._readTokens();
    this._resize();
    addEventListener('resize', () => this._resize());

    canvas.addEventListener('click', (e) => this._click(e));
    canvas.addEventListener('mousemove', (e) => {
      this.cv.style.cursor = this._pick(e.clientX, e.clientY) ? 'pointer' : 'default';
    });

    requestAnimationFrame((t) => this._frame(t));
  }

  _readTokens() {
    const s = getComputedStyle(document.documentElement);
    const rgb = (v) => {
      const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(s.getPropertyValue(v).trim());
      return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
    };
    this.css.ground = s.getPropertyValue('--ground').trim() || '#070B11';
    this.css.accent = rgb('--accent');
    this.css.steel = [96, 116, 140];
    this.css.cat = {};
    for (const [k, v] of Object.entries(CAT_VARS)) this.css.cat[k] = rgb(v);
  }

  _resize() {
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = innerWidth;
    this.h = innerHeight;
    this.cv.width = Math.round(this.w * this.dpr);
    this.cv.height = Math.round(this.h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  _pick(x, y) {
    // Reverse order: the last drawn orb is the one on top.
    for (let i = this.hits.length - 1; i >= 0; i--) {
      const o = this.hits[i];
      if (Math.hypot(x - o.x, y - o.y) <= o.r * 1.15) return o.id;
    }
    return null;
  }

  _click(e) {
    const id = this._pick(e.clientX, e.clientY);
    this.state.select(id === this.state.selectedTool ? null : id);
  }

  /** Project and paint one point cloud.
   *
   * ``dot`` scales point radius independently of cloud radius. Without it a small
   * cloud gets sub-pixel points and vanishes: the tool orbs are a fifth of the
   * centre's size, so at a shared coefficient they render as a faint smudge.
   */
  _cloud(pts, n, cx, cy, scale, disp, freq, nt, rot, colour, alpha, out, dot = 0.0092) {
    const { ctx } = this;
    const cr = Math.cos(rot), sr = Math.sin(rot);
    const [R, G, B] = colour;
    const d = 4.2;

    for (let i = 0; i < n; i++) {
      let x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
      const nd = noise3(x * freq + nt, y * freq - nt * 0.6, z * freq + nt * 0.35);
      const rr = 1 + disp * nd;
      x *= rr; y *= rr; z *= rr;

      const rx = x * cr - z * sr;
      const rz = x * sr + z * cr;
      const p = d / (d - rz);
      const px = cx + rx * scale * p;
      const py = cy + y * scale * p;
      if (px < -8 || px > this.w + 8 || py < -8 || py > this.h + 8) continue;

      const a = Math.min(1, (rz + 1.75) / 2.55) * alpha;
      out.push({ x: px, y: py, z: rz, s: Math.max(0.55, p * scale * dot), a, R, G, B });
    }
  }

  _frame(ts) {
    const t = ts / 1000;
    const st = this.state;

    // Smooth the drive signal. Fast attack, slow release: a late response reads as
    // lag, and a fast release makes consonant gaps look like stutter.
    const target = st.level;
    this.level += (target - this.level) * (target > this.level ? 0.35 : 0.08);

    // colour ← mic state, energy ← engine activity
    const targetMix = st.listening ? 1 : 0;
    let targetEnergy = st.listening ? 0.85 : 0.5;
    if (st.activity === ACTIVITY.SPEAKING) targetEnergy = 1.25;
    else if (st.activity === ACTIVITY.THINKING) targetEnergy = Math.max(targetEnergy, 0.95);

    this.mix += (targetMix - this.mix) * 0.09;
    this.energy += (targetEnergy - this.energy) * 0.06;

    const ctx = this.ctx;
    ctx.fillStyle = this.css.ground;
    ctx.fillRect(0, 0, this.w, this.h);

    const nt = this.reduce ? 0 : t * 0.42;
    const rot = this.reduce ? 0.6 : t * 0.25;
    const base = Math.min(this.w, this.h) * 0.19;
    const pts = [];

    // ── centre orb ─────────────────────────────────────────────────────────
    const disp = 0.06 + 0.34 * this.level + 0.06 * (this.energy - 0.5);
    const scale = base * (1 + 0.12 * this.level) * (0.94 + 0.10 * this.energy);
    const bright = 0.55 + 0.45 * Math.max(this.level, this.energy - 0.5);

    const col = [
      Math.round(lerp(this.css.steel[0], this.css.accent[0], this.mix)),
      Math.round(lerp(this.css.steel[1], this.css.accent[1], this.mix)),
      Math.round(lerp(this.css.steel[2], this.css.accent[2], this.mix)),
    ];
    this._cloud(MAIN, N_MAIN, this.w / 2, this.h / 2, scale, disp, 1.7, nt, rot, col,
                Math.min(1, bright), pts);

    // ── tool orbs ──────────────────────────────────────────────────────────
    // A loose vertical column on the right, floating. Only drawn when tools are
    // live — the resting view is one orb and nothing else.
    this.hits.length = 0;
    const tools = st.activeTools();
    if (tools.length) {
      const colX = this.w / 2 + base * 2.15;
      const span = Math.min(this.h * 0.62, tools.length * base * 0.92);
      const step = tools.length > 1 ? span / (tools.length - 1) : 0;
      const top = this.h / 2 - span / 2;

      tools.forEach((tool, i) => {
        if (!this.spawns.has(tool.id)) this.spawns.set(tool.id, ts);
        const age = (ts - this.spawns.get(tool.id)) / 1000;
        const grow = this.reduce ? 1 : Math.min(1, age / 0.42);
        const ease = 1 - Math.pow(1 - grow, 3);

        const drift = this.reduce ? 0 : Math.sin(t * 0.7 + i * 1.7) * base * 0.07;
        const cx = colX + (this.reduce ? 0 : Math.cos(t * 0.5 + i) * base * 0.05);
        const cy = top + step * i + drift;

        const running = tool.state === 'running';
        // A running tool breathes; a finished one goes quiet and just sits there.
        const pulse = running && !this.reduce ? 0.10 + 0.06 * Math.sin(t * 3.4 + i) : 0.05;
        const sel = st.selectedTool === tool.id;
        const r = base * 0.34 * ease * (sel ? 1.16 : 1);
        const c = this.css.cat[tool.category] || this.css.cat.general;
        // A finished tool dims but stays legible — it is still clickable until it
        // ages out, so it must not fade to the point of looking like a rendering bug.
        const alpha = (running ? 1 : 0.72) * ease * (sel ? 1 : 0.9);

        this._cloud(TOOL, N_TOOL, cx, cy, r, pulse, 2.4, nt * 0.8 + i * 3, rot, c, alpha, pts,
                    0.030);
        this.hits.push({ id: tool.id, x: cx, y: cy, r });
      });
    }
    for (const id of [...this.spawns.keys()]) if (!st.tools.has(id)) this.spawns.delete(id);

    // Painter's algorithm across every cloud at once, so tool orbs correctly
    // interleave with the centre orb instead of always sitting on top of it.
    pts.sort((a, b) => a.z - b.z);
    for (const p of pts) {
      ctx.fillStyle = `rgba(${p.R},${p.G},${p.B},${p.a.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.s, 0, 6.2832);
      ctx.fill();
    }

    requestAnimationFrame((n) => this._frame(n));
  }
}
