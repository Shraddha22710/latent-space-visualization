"use strict";

const fs = require("fs");
const path = require("path");

const outDir = path.join(__dirname, "..", "results");
fs.mkdirSync(outDir, { recursive: true });

const experiments = [
  { dataset: "rings", n: 180, noiseDims: 2, k: 18, iterations: 260, title: "Nested rings: nonlinear 2D geometry with noise dimensions" },
  { dataset: "lorenz", n: 260, noiseDims: 4, k: 28, iterations: 320, title: "Lorenz attractor: nonlinear dynamical-system state space" }
];

const summaries = [];

for (const experiment of experiments) {
  const data = createDataset(experiment.dataset, experiment.n, experiment.noiseDims, 42);
  const dist = pairwiseDistances(data.points);
  const pca = pca2D(data.points);
  const tsne = runSNE(data.points, dist, experiment.k, experiment.iterations * 2, true, pca);
  const umap = runUMAP(data.points, dist, experiment.k, experiment.iterations * 2, pca);

  const embeddings = { pca, tsne: tsne.embedding, umap: umap.embedding };
  const metrics = {
    pca: embeddingMetrics(dist, pca, experiment.k),
    tsne: { ...embeddingMetrics(dist, tsne.embedding, experiment.k), loss: tsne.loss },
    umap: { ...embeddingMetrics(dist, umap.embedding, experiment.k), loss: umap.loss }
  };

  const jsonPath = path.join(outDir, `${experiment.dataset}_embeddings.json`);
  fs.writeFileSync(jsonPath, JSON.stringify({ experiment, metrics, labels: data.labels, embeddings }, null, 2));

  for (const [method, embedding] of Object.entries(embeddings)) {
    const svgPath = path.join(outDir, `${experiment.dataset}_${method}.svg`);
    fs.writeFileSync(svgPath, renderSVG(embedding, data.labels, `${experiment.title} - ${method.toUpperCase()}`, metrics[method]));
  }

  summaries.push({ experiment, metrics });
}

fs.writeFileSync(path.join(outDir, "summary.md"), renderSummary(summaries));
console.log(`Generated ${experiments.length} experiments in ${outDir}`);

function createDataset(type, n, noiseDims, seed) {
  const rand = mulberry32(seed);
  const points3 = [];
  const labels = [];

  for (let i = 0; i < n; i += 1) {
    const t = n === 1 ? 0 : i / (n - 1);
    let point;
    let label;

    if (type === "rings") {
      const ring = i % 3;
      const angle = 2 * Math.PI * (i / n) * 5 + ring * 0.8;
      const radius = 0.45 + ring * 0.38;
      point = [
        radius * Math.cos(angle),
        0.45 * (rand() - 0.5) + ring * 0.18,
        radius * Math.sin(angle)
      ];
      label = ring / 2;
    } else if (type === "lorenz") {
      const trajectory = lorenzTrajectory(n + 120, 0.01).slice(120);
      point = trajectory[i];
      label = i / Math.max(1, n - 1);
    } else {
      throw new Error(`Unknown dataset: ${type}`);
    }

    points3.push(point);
    labels.push(label);
  }

  const points = points3.map((p) => {
    const extra = [];
    for (let j = 0; j < noiseDims; j += 1) {
      const signal = Math.sin((j + 1) * p[0]) + Math.cos((j + 2) * p[2]);
      extra.push(0.18 * signal + normal(rand) * 0.13);
    }
    return [...p, ...extra];
  });

  return { points: standardize(points), labels };
}

function lorenzTrajectory(count, dt) {
  const sigma = 10;
  const rho = 28;
  const beta = 8 / 3;
  let p = [0.1, 1.0, 1.05];
  const points = [];

  function derivative([x, y, z]) {
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z];
  }

  for (let i = 0; i < count; i += 1) {
    const k1 = derivative(p);
    const k2 = derivative(p.map((value, j) => value + 0.5 * dt * k1[j]));
    const k3 = derivative(p.map((value, j) => value + 0.5 * dt * k2[j]));
    const k4 = derivative(p.map((value, j) => value + dt * k3[j]));
    p = p.map((value, j) => value + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    points.push([...p]);
  }

  return points;
}

function pca2D(points) {
  const centered = centerColumns(points);
  const cov = covariance(centered);
  const v1 = powerIteration(cov, 80);
  const lambda1 = quadraticForm(cov, v1);
  const deflated = deflate(cov, v1, lambda1);
  const v2 = powerIteration(deflated, 80);
  return normalize2D(centered.map((row) => [dot(row, v1), dot(row, v2)]));
}

function runSNE(points, dist, k, iterations, useStudentT, initialEmbedding) {
  const highProb = buildSNEProbabilities(dist, k);
  let embedding = addJitter(initialEmbedding, 91, 0.015);
  let loss = 0;

  for (let iter = 0; iter < iterations; iter += 1) {
    const exaggeration = iter < Math.floor(iterations * 0.25) ? 4 : 1;
    const result = stepSNE(embedding, highProb, useStudentT, 55, exaggeration);
    embedding = result.embedding;
    loss = result.loss;
    if (iter % 25 === 0) centerEmbedding(embedding);
  }
  centerAndScaleEmbedding(embedding);

  return { embedding, loss };
}

function stepSNE(y, highProb, useStudentT, lr, exaggeration) {
  const n = y.length;
  const q = Array.from({ length: n }, () => Array(n).fill(0));
  let total = 0;

  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const d2 = squaredDistance2(y[i], y[j]);
      const value = useStudentT ? 1 / (1 + d2) : Math.exp(-d2);
      q[i][j] = value;
      q[j][i] = value;
      total += 2 * value;
    }
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) q[i][j] = total > 0 ? q[i][j] / total : 0;
  }

  const gradients = Array.from({ length: n }, () => [0, 0]);
  let loss = 0;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      const p = highProb[i][j] * exaggeration;
      const qq = Math.max(q[i][j], 1e-12);
      if (p > 0) loss += p * Math.log(p / qq);
      const dx = y[i][0] - y[j][0];
      const dy = y[i][1] - y[j][1];
      const tail = useStudentT ? 1 / (1 + dx * dx + dy * dy) : 1;
      const diff = p - qq;
      gradients[i][0] += 4 * diff * dx * tail;
      gradients[i][1] += 4 * diff * dy * tail;
    }
  }

  const stepScale = useStudentT ? 1 : 0.45;
  const next = y.map((point, i) => [
    point[0] + lr * stepScale * gradients[i][0],
    point[1] + lr * stepScale * gradients[i][1]
  ]);

  return { embedding: next, loss };
}

function runUMAP(points, dist, k, iterations, initialEmbedding) {
  const graph = buildUmapGraph(dist, k);
  let embedding = addJitter(initialEmbedding, 123, 0.02);
  let loss = 0;

  for (let iter = 0; iter < iterations; iter += 1) {
    const result = stepUMAP(embedding, graph, iter, 0.08);
    embedding = result.embedding;
    loss = result.loss;
    if (iter % 25 === 0) centerEmbedding(embedding);
  }
  centerAndScaleEmbedding(embedding);

  return { embedding, loss };
}

function stepUMAP(y, graph, iter, lr) {
  const gradients = Array.from({ length: y.length }, () => [0, 0]);
  let loss = 0;

  for (const edge of graph) {
    const dx = y[edge.i][0] - y[edge.j][0];
    const dy = y[edge.i][1] - y[edge.j][1];
    const d2 = dx * dx + dy * dy + 1e-4;
    const attraction = edge.w / (1 + d2);
    gradients[edge.i][0] -= attraction * dx;
    gradients[edge.i][1] -= attraction * dy;
    gradients[edge.j][0] += attraction * dx;
    gradients[edge.j][1] += attraction * dy;
    loss += edge.w * Math.log(1 + d2);
  }

  const rand = mulberry32(9000 + iter * 997);
  for (let s = 0; s < y.length * 4; s += 1) {
    const i = Math.floor(rand() * y.length);
    let j = Math.floor(rand() * y.length);
    if (i === j) j = (j + 1) % y.length;
    const dx = y[i][0] - y[j][0];
    const dy = y[i][1] - y[j][1];
    const d2 = dx * dx + dy * dy + 1e-3;
    const force = 0.00035 / d2;
    gradients[i][0] += force * dx;
    gradients[i][1] += force * dy;
    gradients[j][0] -= force * dx;
    gradients[j][1] -= force * dy;
  }

  return {
    embedding: y.map((point, i) => [
      point[0] + lr * gradients[i][0],
      point[1] + lr * gradients[i][1]
    ]),
    loss: loss / Math.max(1, graph.length)
  };
}

function embeddingMetrics(highDist, embedding, k) {
  const lowDist = pairwiseDistances(embedding);
  let trust = 0;
  for (let i = 0; i < highDist.length; i += 1) {
    const high = new Set(nearestNeighbors(highDist, i, k).map((item) => item.j));
    const low = nearestNeighbors(lowDist, i, k).map((item) => item.j);
    trust += low.filter((j) => high.has(j)).length / k;
  }
  return { neighborOverlap: trust / highDist.length };
}

function renderSVG(points, labels, title, metrics) {
  const width = 900;
  const height = 680;
  const pad = 70;
  const normalized = normalize2D(points);
  const circles = normalized.map((point, i) => {
    const x = pad + (point[0] * 0.42 + 0.5) * (width - 2 * pad);
    const y = pad + (0.5 - point[1] * 0.42) * (height - 2 * pad);
    return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="5" fill="${colorFor(labels[i])}" fill-opacity="0.86" stroke="white" stroke-width="1.4" />`;
  }).join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <g stroke="#e2e8f0" stroke-width="1">
    ${Array.from({ length: 16 }, (_, i) => `<line x1="${pad + i * 50}" y1="${pad}" x2="${pad + i * 50}" y2="${height - pad}"/>`).join("\n")}
    ${Array.from({ length: 11 }, (_, i) => `<line x1="${pad}" y1="${pad + i * 50}" x2="${width - pad}" y2="${pad + i * 50}"/>`).join("\n")}
  </g>
  <rect x="${pad}" y="${pad}" width="${width - 2 * pad}" height="${height - 2 * pad}" fill="none" stroke="#94a3b8"/>
  <text x="${pad}" y="38" font-family="system-ui, sans-serif" font-size="22" font-weight="700" fill="#17202a">${escapeXml(title)}</text>
  <text x="${pad}" y="62" font-family="system-ui, sans-serif" font-size="14" fill="#64748b">kNN overlap: ${metrics.neighborOverlap.toFixed(3)}${metrics.loss ? ` | optimizer loss: ${metrics.loss.toFixed(3)}` : ""}</text>
  <g>${circles}</g>
</svg>
`;
}

function renderSummary(summaries) {
  const lines = [
    "# Generated Results",
    "",
    "These deterministic artifacts were generated by `node scripts/generate_results.js`.",
    "",
    "| Dataset | Method | kNN overlap | Optimizer loss |",
    "| --- | --- | ---: | ---: |"
  ];

  for (const { experiment, metrics } of summaries) {
    for (const method of ["pca", "tsne", "umap"]) {
      lines.push(`| ${experiment.dataset} | ${method.toUpperCase()} | ${metrics[method].neighborOverlap.toFixed(3)} | ${metrics[method].loss ? metrics[method].loss.toFixed(3) : "-"} |`);
    }
  }

  lines.push(
    "",
    "Interpretation:",
    "",
    "- `kNN overlap` measures how many high-dimensional nearest neighbors remain neighbors in 2D.",
    "- The t-SNE and UMAP implementations here are educational approximations used for reproducible visual artifacts.",
    "- For scientific benchmarking, compare against `scikit-learn`, `openTSNE`, and `umap-learn`."
  );

  return `${lines.join("\n")}\n`;
}

function buildSNEProbabilities(dist, k) {
  const n = dist.length;
  const p = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i += 1) {
    const neighbors = nearestNeighbors(dist, i, k);
    const sigma = Math.max(0.15, neighbors[neighbors.length - 1].d);
    let rowSum = 0;
    for (const nb of neighbors) {
      const value = Math.exp(-(nb.d * nb.d) / (2 * sigma * sigma));
      p[i][nb.j] = value;
      rowSum += value;
    }
    for (const nb of neighbors) p[i][nb.j] /= rowSum || 1;
  }

  const sym = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) sym[i][j] = (p[i][j] + p[j][i]) / (2 * n);
  }
  return sym;
}

function buildUmapGraph(dist, k) {
  const n = dist.length;
  const directed = Array.from({ length: n }, () => new Map());
  for (let i = 0; i < n; i += 1) {
    const neighbors = nearestNeighbors(dist, i, k);
    const rho = neighbors[0]?.d || 0;
    const sigma = Math.max(0.15, neighbors[neighbors.length - 1]?.d || 1);
    for (const nb of neighbors) directed[i].set(nb.j, Math.exp(-Math.max(0, nb.d - rho) / sigma));
  }

  const merged = new Map();
  for (let i = 0; i < n; i += 1) {
    for (const [j, vij] of directed[i]) {
      const vji = directed[j].get(i) || 0;
      const w = vij + vji - vij * vji;
      const a = Math.min(i, j);
      const b = Math.max(i, j);
      merged.set(`${a}:${b}`, { i: a, j: b, w });
    }
  }

  return [...merged.values()].filter((edge) => edge.w > 0.02);
}

function pairwiseDistances(points) {
  const n = points.length;
  const dist = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const d = Math.sqrt(squaredDistance(points[i], points[j]));
      dist[i][j] = d;
      dist[j][i] = d;
    }
  }
  return dist;
}

function nearestNeighbors(dist, i, k) {
  return dist[i].map((d, j) => ({ j, d })).filter((item) => item.j !== i).sort((a, b) => a.d - b.d).slice(0, Math.min(k, dist.length - 1));
}

function standardize(points) {
  const centered = centerColumns(points);
  const dims = points[0].length;
  const variance = Array(dims).fill(0);
  for (const row of centered) {
    for (let j = 0; j < dims; j += 1) variance[j] += row[j] * row[j];
  }
  const scale = variance.map((v) => Math.sqrt(v / Math.max(1, points.length - 1)) || 1);
  return centered.map((row) => row.map((value, j) => value / scale[j]));
}

function centerColumns(points) {
  const dims = points[0].length;
  const mean = Array(dims).fill(0);
  for (const row of points) for (let j = 0; j < dims; j += 1) mean[j] += row[j];
  for (let j = 0; j < dims; j += 1) mean[j] /= points.length;
  return points.map((row) => row.map((value, j) => value - mean[j]));
}

function covariance(points) {
  const dims = points[0].length;
  const cov = Array.from({ length: dims }, () => Array(dims).fill(0));
  for (const row of points) {
    for (let i = 0; i < dims; i += 1) {
      for (let j = 0; j < dims; j += 1) cov[i][j] += row[i] * row[j];
    }
  }
  return cov.map((row) => row.map((value) => value / Math.max(1, points.length - 1)));
}

function powerIteration(matrix, steps) {
  let vector = Array(matrix.length).fill(0).map((_, i) => (i === 0 ? 1 : 0.3 / (i + 1)));
  vector = normalizeVector(vector);
  for (let s = 0; s < steps; s += 1) vector = normalizeVector(matrixVector(matrix, vector));
  return vector;
}

function matrixVector(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function quadraticForm(matrix, vector) {
  return dot(vector, matrixVector(matrix, vector));
}

function deflate(matrix, vector, lambda) {
  return matrix.map((row, i) => row.map((value, j) => value - lambda * vector[i] * vector[j]));
}

function normalizeVector(vector) {
  const norm = Math.sqrt(dot(vector, vector)) || 1;
  return vector.map((value) => value / norm);
}

function normalize2D(points) {
  const centered = centerColumns(points);
  let maxAbs = 0;
  for (const [x, y] of centered) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
  const scale = maxAbs || 1;
  return centered.map(([x, y]) => [x / scale, y / scale]);
}

function centerAndScaleEmbedding(points) {
  const normalized = normalize2D(points);
  for (let i = 0; i < points.length; i += 1) {
    points[i][0] = normalized[i][0];
    points[i][1] = normalized[i][1];
  }
}

function centerEmbedding(points) {
  const centered = centerColumns(points);
  for (let i = 0; i < points.length; i += 1) {
    points[i][0] = centered[i][0];
    points[i][1] = centered[i][1];
  }
}

function addJitter(points, seed, amount) {
  const rand = mulberry32(seed);
  return points.map((point) => [point[0] + normal(rand) * amount, point[1] + normal(rand) * amount]);
}

function colorFor(t) {
  const hue = 210 - 170 * t;
  return `hsl(${hue}, 72%, 46%)`;
}

function escapeXml(text) {
  return String(text).replace(/[<>&'"]/g, (char) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", "'": "&apos;", "\"": "&quot;" }[char]));
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function squaredDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function squaredDistance2(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
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
