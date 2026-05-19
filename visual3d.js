"use strict";

const state = {
  dataset: "swiss",
  projection: "pca",
  count: 420,
  collapse: 0,
  playing: false,
  yaw: -0.7,
  pitch: 0.55,
  zoom: 1.8,
  drag: false,
  lastMouse: [0, 0],
  points: [],
  points4: [],
  labels: [],
  projected: [],
  edges: [],
  seed: 11
};

const el = {};

window.addEventListener("DOMContentLoaded", () => {
  cache();
  bind();
  resize();
  window.addEventListener("resize", resize);
  regenerate();
  requestAnimationFrame(tick);
});

function cache() {
  ["dataset", "projection", "collapse", "collapseLabel", "count", "countLabel", "play", "resetView", "showGraph", "showTrail", "autoRotate", "explain", "canvas3d", "canvas2d", "leftMetric", "rightMetric"].forEach((id) => {
    el[id] = document.getElementById(id);
  });
}

function bind() {
  el.dataset.addEventListener("change", () => {
    state.dataset = el.dataset.value;
    regenerate();
  });
  el.projection.addEventListener("change", () => {
    state.projection = el.projection.value;
    updateProjection();
    draw();
  });
  el.count.addEventListener("input", () => {
    state.count = Number(el.count.value);
    el.countLabel.textContent = state.count;
    regenerate();
  });
  el.collapse.addEventListener("input", () => {
    state.collapse = Number(el.collapse.value) / 100;
    el.collapseLabel.textContent = `${el.collapse.value}%`;
    draw();
  });
  el.play.addEventListener("click", () => {
    state.playing = !state.playing;
    el.play.textContent = state.playing ? "Pause" : "Play";
  });
  el.resetView.addEventListener("click", () => {
    state.yaw = -0.7;
    state.pitch = 0.55;
    state.zoom = 1.8;
    draw();
  });
  for (const id of ["showGraph", "showTrail", "autoRotate"]) el[id].addEventListener("change", draw);
  bindOrbit(el.canvas3d);
}

function bindOrbit(canvas) {
  canvas.addEventListener("pointerdown", (event) => {
    state.drag = true;
    state.lastMouse = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    const dx = event.clientX - state.lastMouse[0];
    const dy = event.clientY - state.lastMouse[1];
    state.lastMouse = [event.clientX, event.clientY];
    state.yaw += dx * 0.008;
    state.pitch = clamp(state.pitch + dy * 0.008, -1.35, 1.35);
    draw();
  });
  canvas.addEventListener("pointerup", () => {
    state.drag = false;
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = clamp(state.zoom + event.deltaY * 0.001, 0.75, 4.2);
    draw();
  }, { passive: false });
}

function resize() {
  for (const canvas of [el.canvas3d, el.canvas2d]) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(500, Math.floor(rect.width * dpr));
    canvas.height = Math.max(420, Math.floor(rect.height * dpr));
  }
  draw();
}

function regenerate() {
  const data = makeDataset(state.dataset, state.count);
  state.points = normalize3(data.points3);
  state.points4 = data.points4;
  state.labels = data.labels;
  state.edges = buildEdges(data.features, 5, 900);
  updateProjection();
  updateText();
  draw();
}

function updateProjection() {
  const features = state.points4.length ? state.points4 : state.points;
  if (state.projection === "pca") state.projected = pca2D(features);
  if (state.projection === "random") state.projected = randomProjection(features);
  if (state.projection === "radial") state.projected = radialProjection(state.points);
  state.projected = normalize2(state.projected);
}

function tick() {
  if (state.playing) {
    state.collapse += 0.006;
    if (state.collapse > 1) state.collapse = 0;
    el.collapse.value = Math.round(state.collapse * 100);
    el.collapseLabel.textContent = `${el.collapse.value}%`;
  }
  if (el.autoRotate.checked && !state.drag) state.yaw += 0.0025;
  draw();
  requestAnimationFrame(tick);
}

function draw() {
  if (!state.points.length) return;
  updateText();
  draw3D();
  draw2D();
}

function draw3D() {
  const canvas = el.canvas3d;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  drawFrame(ctx, w, h);

  const screen = state.points.map((p, i) => {
    const collapsed = [
      lerp(p[0], state.projected[i][0], state.collapse),
      lerp(p[1], state.projected[i][1], state.collapse),
      lerp(p[2], 0, state.collapse)
    ];
    return projectPoint(collapsed, w, h);
  });

  if (el.showGraph.checked) drawEdges(ctx, screen);
  if (el.showTrail.checked) drawTrails(ctx, w, h);
  drawPointCloud(ctx, screen);
}

function draw2D() {
  const canvas = el.canvas2d;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  drawFrame(ctx, w, h);
  const points = state.projected.map((p) => toScreen2(p, w, h));
  if (el.showGraph.checked) drawEdges(ctx, points, 0.13);
  drawPointCloud(ctx, points, false);
}

function drawFrame(ctx, w, h) {
  const pad = Math.min(w, h) * 0.075;
  ctx.save();
  ctx.strokeStyle = "rgba(15,118,110,0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.restore();
}

function drawEdges(ctx, screen, alpha = 0.09) {
  ctx.save();
  ctx.strokeStyle = `rgba(15,118,110,${alpha})`;
  ctx.lineWidth = 1;
  for (const [i, j] of state.edges) {
    ctx.beginPath();
    ctx.moveTo(screen[i].x, screen[i].y);
    ctx.lineTo(screen[j].x, screen[j].y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawTrails(ctx, w, h) {
  ctx.save();
  ctx.lineWidth = 1;
  for (let i = 0; i < state.points.length; i += 8) {
    const a = projectPoint(state.points[i], w, h);
    const b = projectPoint([state.projected[i][0], state.projected[i][1], 0], w, h);
    ctx.strokeStyle = "rgba(180,83,9,0.13)";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPointCloud(ctx, screen, depth = true) {
  const order = [...screen.keys()].sort((a, b) => (screen[a].z || 0) - (screen[b].z || 0));
  ctx.save();
  for (const i of order) {
    const p = screen[i];
    const r = depth ? clamp(4.2 + (p.z || 0) * 1.5, 2.5, 7.5) : 4.6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(state.labels[i]);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
  ctx.restore();
}

function projectPoint(p, w, h) {
  const cy = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  let x = p[0] * cy + p[2] * sy;
  let z = -p[0] * sy + p[2] * cy;
  let y = p[1] * cp - z * sp;
  z = p[1] * sp + z * cp;
  const scale = Math.min(w, h) * 0.34 * state.zoom / (state.zoom + z * 0.45);
  return { x: w / 2 + x * scale, y: h / 2 - y * scale, z };
}

function toScreen2(p, w, h) {
  const scale = Math.min(w, h) * 0.36;
  return { x: w / 2 + p[0] * scale, y: h / 2 - p[1] * scale, z: 0 };
}

function makeDataset(type, n) {
  const rand = mulberry32(state.seed + n);
  const points3 = [];
  const points4 = [];
  const labels = [];
  for (let i = 0; i < n; i += 1) {
    const t = i / Math.max(1, n - 1);
    let p;
    let p4;
    let label = t;
    if (type === "swiss") {
      const a = 1.5 * Math.PI + 3.4 * Math.PI * t;
      p = [a * Math.cos(a) / 8, 2.2 * (rand() - 0.5), a * Math.sin(a) / 8];
      p4 = [...p, Math.sin(a)];
    } else if (type === "lorenz") {
      p = lorenzPoint(i + 100);
      p4 = [...p, Math.sin(0.05 * i)];
    } else if (type === "helix") {
      const a = 8 * Math.PI * t;
      p = [Math.cos(a), 2.2 * (t - 0.5), Math.sin(a)];
      p4 = [...p, Math.cos(2 * a)];
    } else if (type === "rings") {
      const ring = i % 4;
      const a = 2 * Math.PI * i / n * 7 + ring * 0.65;
      const r = 0.35 + ring * 0.25;
      p = [r * Math.cos(a), (ring - 1.5) * 0.24 + 0.08 * normal(rand), r * Math.sin(a)];
      p4 = [...p, ring / 3];
      label = ring / 3;
    } else {
      const bits = [(i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1, (i & 8) ? 1 : -1];
      const noise = [normal(rand), normal(rand), normal(rand), normal(rand)].map((v) => v * 0.08);
      p4 = bits.map((v, j) => v + noise[j]);
      p = [p4[0] + 0.35 * p4[3], p4[1] - 0.25 * p4[3], p4[2] + 0.2 * p4[3]];
      label = (bits[3] + 1) / 2;
    }
    points3.push(p);
    points4.push(p4);
    labels.push(label);
  }
  return { points3, points4, features: points4, labels };
}

function lorenzPoint(steps) {
  let p = [0.1, 1.0, 1.05];
  const dt = 0.01;
  for (let i = 0; i < steps; i += 1) {
    const f = ([x, y, z]) => [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z];
    const k1 = f(p);
    const k2 = f(p.map((v, j) => v + 0.5 * dt * k1[j]));
    const k3 = f(p.map((v, j) => v + 0.5 * dt * k2[j]));
    const k4 = f(p.map((v, j) => v + dt * k3[j]));
    p = p.map((v, j) => v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
  }
  return p;
}

function buildEdges(features, k, limit) {
  const edges = [];
  for (let i = 0; i < features.length; i += 1) {
    const neighbors = features.map((p, j) => ({ j, d: squared(features[i], p) }))
      .filter((x) => x.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    for (const nb of neighbors) {
      if (i < nb.j) edges.push([i, nb.j]);
      if (edges.length >= limit) return edges;
    }
  }
  return edges;
}

function pca2D(points) {
  const centered = center(points);
  const cov = covariance(centered);
  const v1 = power(cov, 60);
  const l1 = dot(v1, matvec(cov, v1));
  const deflated = cov.map((row, i) => row.map((v, j) => v - l1 * v1[i] * v1[j]));
  const v2 = power(deflated, 60);
  return centered.map((p) => [dot(p, v1), dot(p, v2)]);
}

function randomProjection(points) {
  const dims = points[0].length;
  const rand = mulberry32(99);
  const a = Array.from({ length: dims }, () => rand() - 0.5);
  const b = Array.from({ length: dims }, () => rand() - 0.5);
  return points.map((p) => [dot(p, a), dot(p, b)]);
}

function radialProjection(points) {
  return points.map(([x, y, z]) => [Math.atan2(z, x), y + 0.25 * Math.hypot(x, z)]);
}

function updateText() {
  const text = {
    swiss: "Swiss roll: a curved manifold where PCA gives a shadow but cannot unroll the surface.",
    lorenz: "Lorenz attractor: a nonlinear dynamical-system trajectory with structured local neighborhoods.",
    helix: "Helix: local neighborhoods follow the coil, while global Euclidean shortcuts can mislead.",
    rings: "Nested rings: nonlinear projections reveal how loop-like structures separate.",
    hypercube: "4D hypercube shadow: a higher-dimensional object viewed through a 3D projection, then collapsed to 2D."
  };
  el.explain.textContent = text[state.dataset];
  el.leftMetric.textContent = `${state.count} points | ${state.edges.length} graph edges`;
  el.rightMetric.textContent = `${state.projection} target | collapse ${(state.collapse * 100).toFixed(0)}%`;
}

function normalize3(points) {
  const c = center(points);
  let m = 0;
  for (const p of c) m = Math.max(m, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
  return c.map((p) => p.map((v) => v / (m || 1)));
}

function normalize2(points) {
  const c = center(points);
  let m = 0;
  for (const p of c) m = Math.max(m, Math.abs(p[0]), Math.abs(p[1]));
  return c.map((p) => [p[0] / (m || 1), p[1] / (m || 1)]);
}

function center(points) {
  const dims = points[0].length;
  const mean = Array(dims).fill(0);
  for (const p of points) for (let j = 0; j < dims; j += 1) mean[j] += p[j];
  for (let j = 0; j < dims; j += 1) mean[j] /= points.length;
  return points.map((p) => p.map((v, j) => v - mean[j]));
}

function covariance(points) {
  const dims = points[0].length;
  const cov = Array.from({ length: dims }, () => Array(dims).fill(0));
  for (const p of points) for (let i = 0; i < dims; i += 1) for (let j = 0; j < dims; j += 1) cov[i][j] += p[i] * p[j];
  return cov.map((row) => row.map((v) => v / Math.max(1, points.length - 1)));
}

function power(matrix, steps) {
  let v = Array(matrix.length).fill(0).map((_, i) => i === 0 ? 1 : 0.2);
  v = norm(v);
  for (let s = 0; s < steps; s += 1) v = norm(matvec(matrix, v));
  return v;
}

function matvec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function norm(v) {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map((x) => x / n);
}

function squared(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += (a[i] - b[i]) ** 2;
  return s;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function colorFor(t) {
  return `hsl(${210 - 175 * t}, 72%, 46%)`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normal(rand) {
  const u = Math.max(rand(), 1e-9);
  const v = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mulberry32(seed) {
  return function random() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
