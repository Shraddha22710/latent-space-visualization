"use strict";

const state = {
  dataset: "swiss",
  method: "pca",
  n: 140,
  noiseDims: 3,
  k: 18,
  lr: 0.3,
  pcaBlend: 1,
  playing: false,
  iteration: 0,
  data: null,
  embedding: [],
  pcaEmbedding: [],
  rawProjection: [],
  pairwise: [],
  highProb: [],
  graph: [],
  stress: 0,
  seed: 7,
  selectedIndex: -1,
  activeDetail: "intuition"
};

const el = {};

window.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindControls();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  resetExperiment();
  requestAnimationFrame(tick);
});

function cacheElements() {
  [
    "datasetSelect", "pointCount", "pointCountLabel", "noiseDims", "noiseDimsLabel",
    "methodTabs", "pcaControls", "neighborControls", "pcaBlend", "pcaBlendLabel",
    "neighborSize", "neighborSizeLabel", "learningRate", "learningRateLabel",
    "playButton", "stepButton", "resetButton", "showLinks", "showGhost",
    "lessonList", "detailTabs", "detailContent", "pointInspector",
    "methodTitle", "methodDescription", "iterationStat", "stressStat",
    "dimensionStat", "plotCanvas", "stepTitle", "stepText", "algorithmText",
    "readText"
  ].forEach((id) => {
    el[id] = document.getElementById(id);
  });
}

function bindControls() {
  el.datasetSelect.addEventListener("change", () => {
    state.dataset = el.datasetSelect.value;
    resetExperiment();
  });

  el.pointCount.addEventListener("input", () => {
    state.n = Number(el.pointCount.value);
    el.pointCountLabel.textContent = state.n;
    resetExperiment();
  });

  el.noiseDims.addEventListener("input", () => {
    state.noiseDims = Number(el.noiseDims.value);
    el.noiseDimsLabel.textContent = state.noiseDims;
    resetExperiment();
  });

  el.methodTabs.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-method]");
    if (!button) return;
    state.method = button.dataset.method;
    [...el.methodTabs.querySelectorAll("button")].forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    el.pcaControls.classList.toggle("hidden", state.method !== "pca");
    el.neighborControls.classList.toggle("hidden", state.method === "pca");
    resetEmbeddingOnly();
  });

  el.pcaBlend.addEventListener("input", () => {
    state.pcaBlend = Number(el.pcaBlend.value) / 100;
    el.pcaBlendLabel.textContent = `${el.pcaBlend.value}%`;
    updatePcaBlend();
    draw();
  });

  el.neighborSize.addEventListener("input", () => {
    state.k = Number(el.neighborSize.value);
    el.neighborSizeLabel.textContent = state.k;
    resetEmbeddingOnly();
  });

  el.learningRate.addEventListener("input", () => {
    state.lr = Number(el.learningRate.value) / 100;
    el.learningRateLabel.textContent = state.lr.toFixed(2);
  });

  el.playButton.addEventListener("click", () => {
    state.playing = !state.playing;
    el.playButton.textContent = state.playing ? "Pause" : "Play";
  });

  el.stepButton.addEventListener("click", () => {
    runSteps(1);
    draw();
  });

  el.resetButton.addEventListener("click", () => {
    resetExperiment();
  });

  el.showLinks.addEventListener("change", draw);
  el.showGhost.addEventListener("change", draw);

  el.lessonList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-lesson]");
    if (!button) return;
    applyLesson(button.dataset.lesson);
  });

  el.detailTabs.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-detail]");
    if (!button) return;
    state.activeDetail = button.dataset.detail;
    [...el.detailTabs.querySelectorAll("button")].forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    updateDetailContent();
  });

  el.plotCanvas.addEventListener("mousemove", inspectPoint);
  el.plotCanvas.addEventListener("mouseleave", () => {
    state.selectedIndex = -1;
    el.pointInspector.classList.add("hidden");
    draw();
  });
}

function resizeCanvas() {
  const rect = el.plotCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  el.plotCanvas.width = Math.max(600, Math.floor(rect.width * dpr));
  el.plotCanvas.height = Math.max(420, Math.floor(rect.height * dpr));
  draw();
}

function resetExperiment() {
  state.seed += 17;
  state.selectedIndex = -1;
  state.data = createDataset(state.dataset, state.n, state.noiseDims, state.seed);
  state.rawProjection = project3D(state.data.points3);
  state.pairwise = pairwiseDistances(state.data.points);
  state.pcaEmbedding = pca2D(state.data.points);
  resetEmbeddingOnly();
}

function resetEmbeddingOnly() {
  state.playing = false;
  state.selectedIndex = -1;
  el.pointInspector.classList.add("hidden");
  el.playButton.textContent = "Play";
  state.iteration = 0;
  state.stress = 0;

  if (state.method === "pca") {
    updatePcaBlend();
  } else {
    state.embedding = jitteredCircle(state.n, state.seed + 101);
    if (state.method === "umap") {
      state.graph = buildUmapGraph(state.pairwise, state.k);
      state.highProb = [];
    } else {
      state.highProb = buildSNEProbabilities(state.pairwise, state.k);
      state.graph = probabilitiesToEdges(state.highProb, state.k);
    }
  }

  updateText();
  updateStats();
  draw();
}

function updatePcaBlend() {
  const t = state.pcaBlend;
  state.embedding = state.pcaEmbedding.map((point, index) => [
    lerp(state.rawProjection[index][0], point[0], t),
    lerp(state.rawProjection[index][1], point[1], t)
  ]);
}

function tick() {
  if (state.playing) {
    runSteps(state.method === "pca" ? 0 : 2);
    draw();
  }
  requestAnimationFrame(tick);
}

function applyLesson(lesson) {
  const lessons = {
    pcaSwiss: { dataset: "swiss", method: "pca", n: 160, noiseDims: 2, k: 18, lr: 0.3, detail: "intuition" },
    sneCrowding: { dataset: "rings", method: "sne", n: 140, noiseDims: 2, k: 14, lr: 0.22, detail: "paper" },
    tsneClusters: { dataset: "blobs", method: "tsne", n: 180, noiseDims: 4, k: 22, lr: 0.32, detail: "math" },
    umapNeighbors: { dataset: "helix", method: "umap", n: 180, noiseDims: 3, k: 35, lr: 0.28, detail: "algorithm" },
    lorenzFlow: { dataset: "lorenz", method: "umap", n: 220, noiseDims: 4, k: 28, lr: 0.26, detail: "intuition" }
  };
  const preset = lessons[lesson];
  if (!preset) return;

  state.dataset = preset.dataset;
  state.method = preset.method;
  state.n = preset.n;
  state.noiseDims = preset.noiseDims;
  state.k = preset.k;
  state.lr = preset.lr;
  state.activeDetail = preset.detail;

  el.datasetSelect.value = state.dataset;
  el.pointCount.value = state.n;
  el.pointCountLabel.textContent = state.n;
  el.noiseDims.value = state.noiseDims;
  el.noiseDimsLabel.textContent = state.noiseDims;
  el.neighborSize.value = state.k;
  el.neighborSizeLabel.textContent = state.k;
  el.learningRate.value = Math.round(state.lr * 100);
  el.learningRateLabel.textContent = state.lr.toFixed(2);
  el.pcaControls.classList.toggle("hidden", state.method !== "pca");
  el.neighborControls.classList.toggle("hidden", state.method === "pca");

  [...el.methodTabs.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.method === state.method);
  });
  [...el.detailTabs.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.detail === state.activeDetail);
  });

  resetExperiment();
}

function runSteps(count) {
  if (state.method === "pca") {
    state.pcaBlend = Math.min(1, state.pcaBlend + 0.01);
    el.pcaBlend.value = Math.round(state.pcaBlend * 100);
    el.pcaBlendLabel.textContent = `${el.pcaBlend.value}%`;
    updatePcaBlend();
    return;
  }

  for (let i = 0; i < count; i += 1) {
    if (state.method === "sne") stepSNE(false);
    if (state.method === "tsne") stepSNE(true);
    if (state.method === "umap") stepUMAP();
    state.iteration += 1;
  }
  centerAndScaleEmbedding(state.embedding);
  updateStats();
}

function createDataset(type, n, noiseDims, seed) {
  const rand = mulberry32(seed);
  const points3 = [];
  const labels = [];

  for (let i = 0; i < n; i += 1) {
    const t = n === 1 ? 0 : i / (n - 1);
    let point;
    let label;

    if (type === "swiss") {
      const angle = 1.5 * Math.PI + 3.3 * Math.PI * t;
      const radius = angle;
      point = [
        radius * Math.cos(angle) / 7,
        2.2 * (rand() - 0.5),
        radius * Math.sin(angle) / 7
      ];
      label = t;
    } else if (type === "helix") {
      const angle = 7 * Math.PI * t;
      point = [
        Math.cos(angle),
        2.4 * (t - 0.5),
        Math.sin(angle)
      ];
      label = t;
    } else if (type === "rings") {
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
      const trajectory = lorenzTrajectory(n + 90, 0.01).slice(90);
      point = trajectory[i];
      label = i / Math.max(1, n - 1);
    } else {
      const centers = [
        [-1.2, -0.6, -0.5],
        [1.1, -0.5, 0.65],
        [-0.2, 1.0, 0.15],
        [0.55, 0.35, -1.15]
      ];
      const cluster = i % centers.length;
      const c = centers[cluster];
      point = [
        c[0] + normal(rand) * 0.18,
        c[1] + normal(rand) * 0.18,
        c[2] + normal(rand) * 0.18
      ];
      label = cluster / (centers.length - 1);
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

  return {
    points: standardize(points),
    points3: standardize(points3),
    labels,
    dims: 3 + noiseDims
  };
}

function lorenzTrajectory(count, dt) {
  const sigma = 10;
  const rho = 28;
  const beta = 8 / 3;
  let p = [0.1, 1.0, 1.05];
  const points = [];

  function derivative([x, y, z]) {
    return [
      sigma * (y - x),
      x * (rho - z) - y,
      x * y - beta * z
    ];
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
  const v1 = powerIteration(cov, 60);
  const lambda1 = quadraticForm(cov, v1);
  const deflated = deflate(cov, v1, lambda1);
  const v2 = powerIteration(deflated, 60);
  const projected = centered.map((row) => [dot(row, v1), dot(row, v2)]);
  return normalize2D(projected);
}

function stepSNE(useStudentT) {
  const n = state.n;
  const y = state.embedding;
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
    for (let j = 0; j < n; j += 1) {
      q[i][j] = total > 0 ? q[i][j] / total : 0;
    }
  }

  const gradients = Array.from({ length: n }, () => [0, 0]);
  let loss = 0;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      const p = state.highProb[i][j];
      const qq = Math.max(q[i][j], 1e-12);
      if (p > 0) loss += p * Math.log(p / qq);
      const diff = p - qq;
      const dx = y[i][0] - y[j][0];
      const dy = y[i][1] - y[j][1];
      const tail = useStudentT ? 1 / (1 + dx * dx + dy * dy) : 1;
      gradients[i][0] += 4 * diff * dx * tail;
      gradients[i][1] += 4 * diff * dy * tail;
    }
  }

  const lr = state.lr * (useStudentT ? 2.8 : 1.5);
  for (let i = 0; i < n; i += 1) {
    y[i][0] += lr * gradients[i][0];
    y[i][1] += lr * gradients[i][1];
  }

  state.stress = loss;
}

function stepUMAP() {
  const y = state.embedding;
  const gradients = Array.from({ length: state.n }, () => [0, 0]);
  let loss = 0;

  for (const edge of state.graph) {
    const i = edge.i;
    const j = edge.j;
    const w = edge.w;
    const dx = y[i][0] - y[j][0];
    const dy = y[i][1] - y[j][1];
    const d2 = dx * dx + dy * dy + 1e-4;
    const attraction = w / (1 + d2);
    gradients[i][0] -= attraction * dx;
    gradients[i][1] -= attraction * dy;
    gradients[j][0] += attraction * dx;
    gradients[j][1] += attraction * dy;
    loss += w * Math.log(1 + d2);
  }

  const rand = mulberry32(state.seed + state.iteration * 997);
  const repulsionSamples = state.n * 4;
  for (let s = 0; s < repulsionSamples; s += 1) {
    const i = Math.floor(rand() * state.n);
    let j = Math.floor(rand() * state.n);
    if (i === j) j = (j + 1) % state.n;
    const dx = y[i][0] - y[j][0];
    const dy = y[i][1] - y[j][1];
    const d2 = dx * dx + dy * dy + 1e-3;
    const force = 0.0018 / d2;
    gradients[i][0] += force * dx;
    gradients[i][1] += force * dy;
    gradients[j][0] -= force * dx;
    gradients[j][1] -= force * dy;
  }

  for (let i = 0; i < state.n; i += 1) {
    y[i][0] += state.lr * gradients[i][0] * 2.5;
    y[i][1] += state.lr * gradients[i][1] * 2.5;
  }

  state.stress = loss / Math.max(1, state.graph.length);
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
    for (const nb of neighbors) {
      p[i][nb.j] /= rowSum || 1;
    }
  }

  const sym = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      sym[i][j] = (p[i][j] + p[j][i]) / (2 * n);
    }
  }
  return sym;
}

function buildUmapGraph(dist, k) {
  const n = dist.length;
  const directed = Array.from({ length: n }, () => new Map());
  for (let i = 0; i < n; i += 1) {
    const neighbors = nearestNeighbors(dist, i, k);
    const rho = neighbors[0]?.d || 0;
    const sigma = Math.max(0.15, neighbors[Math.max(0, neighbors.length - 1)]?.d || 1);
    for (const nb of neighbors) {
      const value = Math.exp(-Math.max(0, nb.d - rho) / sigma);
      directed[i].set(nb.j, value);
    }
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

function probabilitiesToEdges(prob, k) {
  const edges = [];
  for (let i = 0; i < prob.length; i += 1) {
    const ranked = prob[i]
      .map((w, j) => ({ i, j, w }))
      .filter((edge) => edge.i !== edge.j && edge.w > 0)
      .sort((a, b) => b.w - a.w)
      .slice(0, k);
    edges.push(...ranked);
  }
  return edges;
}

function draw() {
  if (!el.plotCanvas || !state.data) return;
  const canvas = el.plotCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = Math.min(w, h) * 0.08;
  const box = { x: pad, y: pad, w: w - pad * 2, h: h - pad * 2 };
  drawGrid(ctx, box);

  if (el.showGhost.checked) {
    drawGhost(ctx, box);
  }

  if (state.method !== "pca" && el.showLinks.checked) {
    drawLinks(ctx, box);
  }

  drawPoints(ctx, box);
  drawLegend(ctx, w, h);
}

function drawGrid(ctx, box) {
  ctx.save();
  ctx.strokeStyle = "rgba(15, 118, 110, 0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.restore();
}

function drawGhost(ctx, box) {
  const projected = normalize2D(state.rawProjection);
  ctx.save();
  ctx.globalAlpha = 0.22;
  for (let i = 0; i < projected.length; i += 1) {
    const [x, y] = toScreen(projected[i], box);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(state.data.labels[i]);
    ctx.fill();
  }
  ctx.restore();
}

function drawLinks(ctx, box) {
  const edges = state.graph.length ? state.graph : probabilitiesToEdges(state.highProb, state.k);
  const limit = Math.min(edges.length, 1200);
  ctx.save();
  ctx.lineWidth = 1;
  for (let e = 0; e < limit; e += 1) {
    const edge = edges[e];
    const a = toScreen(state.embedding[edge.i], box);
    const b = toScreen(state.embedding[edge.j], box);
    const selected = edge.i === state.selectedIndex || edge.j === state.selectedIndex;
    ctx.strokeStyle = selected
      ? "rgba(180, 83, 9, 0.7)"
      : `rgba(15, 118, 110, ${Math.min(0.18, 0.03 + edge.w * 0.8)})`;
    ctx.lineWidth = selected ? 2.5 : 1;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPoints(ctx, box) {
  ctx.save();
  for (let i = 0; i < state.embedding.length; i += 1) {
    const [x, y] = toScreen(state.embedding[i], box);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(state.data.labels[i]);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.86)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if (state.selectedIndex >= 0) {
    const [x, y] = toScreen(state.embedding[state.selectedIndex], box);
    ctx.beginPath();
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(180, 83, 9, 0.95)";
    ctx.lineWidth = 4;
    ctx.stroke();
  }
  ctx.restore();
}

function drawLegend(ctx, w, h) {
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "rgba(217,224,232,0.95)";
  ctx.lineWidth = 1;
  const x = 24;
  const y = h - 86;
  ctx.fillRect(x, y, 300, 58);
  ctx.strokeRect(x, y, 300, 58);
  ctx.fillStyle = "#334155";
  ctx.font = `${Math.max(12, Math.round(w / 95))}px system-ui, sans-serif`;
  ctx.fillText("Color = hidden position or class", x + 16, y + 24);
  ctx.fillStyle = "#64748b";
  ctx.fillText("Faint points = original 3D shadow", x + 16, y + 44);
  ctx.restore();
}

const methodDetails = {
  pca: {
    intuition: "<p>PCA asks a geometric question: if the high-dimensional cloud had to cast a 2D shadow, which flat shadow keeps the largest spread? It is a rotation plus projection, so it is easy to reason about, but it cannot unroll a curved manifold.</p><p>Use it as the first baseline. If PCA already separates the structure, your representation is probably carrying a strong linear signal.</p>",
    math: "<p>Given centered data <code>X_c = X - mean(X)</code>, PCA computes the covariance matrix <code>C = (1 / (n - 1)) X_c^T X_c</code>.</p><p>Then solve <code>C v_i = lambda_i v_i</code>. The eigenvectors <code>v_i</code> are principal directions; eigenvalues <code>lambda_i</code> measure variance captured along each direction. The 2D projection is <code>Y = X_c [v_1, v_2]</code>.</p>",
    algorithm: "<ol><li>Standardize or center the features.</li><li>Compute covariance or use SVD directly on the centered matrix.</li><li>Sort components by eigenvalue or singular value.</li><li>Project onto the first two components.</li><li>Inspect explained variance before trusting a 2D picture.</li></ol>",
    paper: "<p>PCA traces back to Pearson's least-squares view of fitting lines and planes to point systems, and Hotelling's principal components formulation. The important practical detail is that PCA optimizes global variance, not neighborhood preservation.</p><p>Failure mode to watch here: spirals, rolls, rings, and other nonlinear manifolds can overlap because PCA only gives a flat linear view.</p>"
  },
  sne: {
    intuition: "<p>SNE changes the question from axes to neighbors. Instead of finding a projection direction, it asks: for each point, who are my likely neighbors? The animation moves 2D points until those neighbor probabilities look similar.</p><p>This makes SNE a bridge between PCA and t-SNE: it is nonlinear and local, but its Gaussian low-dimensional map can crowd points near the center.</p>",
    math: "<p>For high-dimensional points, SNE defines conditional probabilities like <code>p_{j|i} = exp(-||x_i - x_j||^2 / (2 sigma_i^2)) / sum_{k != i} exp(-||x_i - x_k||^2 / (2 sigma_i^2))</code>.</p><p>In the 2D map it defines <code>q_{j|i}</code> similarly from <code>y_i</code>. The objective is a sum of KL divergences: <code>C = sum_i KL(P_i || Q_i)</code>.</p>",
    algorithm: "<ol><li>Choose a neighborhood scale. In full SNE this is tied to perplexity.</li><li>Compute high-dimensional neighbor probabilities.</li><li>Initialize points in 2D.</li><li>Compute low-dimensional Gaussian probabilities.</li><li>Use gradient descent to make important high-dimensional neighbors close in 2D.</li></ol>",
    paper: "<p>Hinton and Roweis introduced SNE as a probabilistic neighbor embedding method. The core insight is still powerful: preserve neighborhoods by matching probability distributions rather than preserving raw distances.</p><p>Its biggest teaching value is showing why t-SNE changed the low-dimensional distribution to a heavy-tailed Student-t form.</p>"
  },
  tsne: {
    intuition: "<p>t-SNE keeps SNE's neighbor-probability idea but gives the 2D map heavier tails. Nearby points still attract, but moderately far points get more room, which helps clusters stop collapsing into a crowded center.</p><p>It is excellent for seeing local cluster structure, but it is not a map with trustworthy compass directions or large-scale distances.</p>",
    math: "<p>The high-dimensional similarities are symmetrized probabilities <code>p_ij</code>. The low-dimensional similarities use a Student-t kernel with one degree of freedom: <code>q_ij = (1 + ||y_i - y_j||^2)^-1 / sum_{k != l} (1 + ||y_k - y_l||^2)^-1</code>.</p><p>The objective is <code>KL(P || Q) = sum_ij p_ij log(p_ij / q_ij)</code>. The gradient has an attractive term weighted by <code>p_ij</code> and a repulsive term weighted by <code>q_ij</code>.</p>",
    algorithm: "<ol><li>Pick perplexity, which controls effective neighbor count.</li><li>Binary-search local bandwidths so each point has the target entropy.</li><li>Symmetrize high-dimensional probabilities.</li><li>Initialize the 2D map, often with PCA.</li><li>Run gradient descent with early exaggeration and momentum in production implementations.</li></ol>",
    paper: "<p>van der Maaten and Hinton's 2008 t-SNE paper introduced the heavy-tailed low-dimensional distribution to address SNE's crowding problem. Important implementation details include perplexity, early exaggeration, optimization schedule, and initialization.</p><p>Interpretation warning from the method's behavior: cluster gaps and cluster sizes can be artifacts of optimization and parameters.</p>"
  },
  umap: {
    intuition: "<p>UMAP treats the dataset as a fuzzy neighbor graph. The visual task is to draw a 2D graph that keeps strong high-dimensional connections strong while separating points that should not be connected.</p><p>The neighbor count acts like a focus knob: small values reveal local texture; larger values preserve broader continuity.</p>",
    math: "<p>UMAP estimates local connectivity with distances to nearest neighbors. A directed edge weight can be written as <code>w_{i|j} = exp(-(d(x_i, x_j) - rho_i) / sigma_i)</code> for neighbors beyond the local radius <code>rho_i</code>.</p><p>Directed weights are combined with fuzzy union: <code>w_ij = w_{i|j} + w_{j|i} - w_{i|j} w_{j|i}</code>. The low-dimensional graph is optimized by minimizing a cross-entropy between high-dimensional and low-dimensional fuzzy sets.</p>",
    algorithm: "<ol><li>Find k-nearest neighbors for every point.</li><li>Estimate local distance scales so neighborhoods adapt to density.</li><li>Create directed weighted edges.</li><li>Symmetrize edges with fuzzy union.</li><li>Optimize the 2D embedding with attractive edge samples and repulsive negative samples.</li></ol>",
    paper: "<p>McInnes, Healy, and Melville frame UMAP using manifold assumptions and fuzzy simplicial sets. For practical learners, the key details are the k-neighbor graph, local density normalization, fuzzy union, and cross-entropy optimization.</p><p>UMAP is often faster and more stable than t-SNE, but it still makes a visualization, not proof of discrete clusters.</p>"
  }
};

function updateText() {
  const copy = {
    pca: {
      title: "PCA: project onto directions of maximum variance",
      desc: "PCA finds linear axes that preserve the biggest spread in the data.",
      step: "The plot blends from a simple 3D camera view into the first two principal components. PCA is fast and global, but it cannot unfold curved shapes.",
      algorithm: "Center X, compute the covariance matrix, estimate the top eigenvectors, then project X onto those principal axes.",
      read: "Distances and axes are more meaningful here than in t-SNE or UMAP, but only linear structure is being preserved."
    },
    sne: {
      title: "SNE: match high-dimensional neighbor probabilities",
      desc: "SNE converts distances into Gaussian neighborhood probabilities and moves 2D points to match them.",
      step: "Each step compares high-dimensional neighbor probabilities with 2D neighbor probabilities, then pulls missed neighbors together and pushes false neighbors apart.",
      algorithm: "Convert high-dimensional distances into Gaussian probabilities P, initialize 2D points, compute low-dimensional Gaussian probabilities Q, then reduce KL(P || Q).",
      read: "Trust small neighborhoods. SNE can struggle with crowding because it uses Gaussian similarities in the low-dimensional map."
    },
    tsne: {
      title: "t-SNE: SNE with heavy-tailed low-dimensional distances",
      desc: "t-SNE uses a Student-t curve in 2D, giving distant points more room and making clusters easier to see.",
      step: "The optimizer minimizes KL divergence between high-dimensional similarities and low-dimensional similarities. The heavy tail helps reduce center crowding.",
      algorithm: "Build high-dimensional Gaussian neighbor probabilities P, compute 2D Student-t probabilities Q, then use gradient descent to minimize KL divergence.",
      read: "Local clusters are the main story. Large gaps between faraway groups are visually useful but not always quantitative."
    },
    umap: {
      title: "UMAP: optimize a nearest-neighbor graph",
      desc: "UMAP builds a weighted neighbor graph in high dimensions and lays out a similar graph in 2D.",
      step: "Connected neighbors attract each other, while sampled non-neighbors repel each other. This simplified version mirrors the intuition behind UMAP's graph objective.",
      algorithm: "Find k-nearest neighbors, turn distances into weighted graph edges, symmetrize the graph, then optimize a low-dimensional graph with attraction and repulsion.",
      read: "UMAP often balances local clusters with broader shape. Change neighborhood size to shift between detail and global continuity."
    }
  }[state.method];

  el.methodTitle.textContent = copy.title;
  el.methodDescription.textContent = copy.desc;
  el.stepText.textContent = copy.step;
  el.algorithmText.textContent = copy.algorithm;
  el.readText.textContent = copy.read;
  updateDetailContent();
}

function updateDetailContent() {
  const details = methodDetails[state.method];
  if (!details) return;
  el.detailContent.innerHTML = details[state.activeDetail];
}

function inspectPoint(event) {
  if (!state.embedding.length) return;
  const canvas = el.plotCanvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const y = (event.clientY - rect.top) * dpr;
  const pad = Math.min(canvas.width, canvas.height) * 0.08;
  const box = { x: pad, y: pad, w: canvas.width - pad * 2, h: canvas.height - pad * 2 };

  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < state.embedding.length; i += 1) {
    const p = toScreen(state.embedding[i], box);
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }

  if (bestD > (18 * dpr) ** 2) {
    state.selectedIndex = -1;
    el.pointInspector.classList.add("hidden");
    draw();
    return;
  }

  state.selectedIndex = best;
  const neighbors = nearestNeighbors(state.pairwise, best, Math.min(5, state.n - 1))
    .map((neighbor) => neighbor.j)
    .join(", ");
  const original = state.data.points3[best].map((value) => value.toFixed(2)).join(", ");
  const embedding = state.embedding[best].map((value) => value.toFixed(2)).join(", ");
  el.pointInspector.innerHTML = `<strong>Point ${best}</strong>Original 3D: [${original}]<br>Embedding: [${embedding}]<br>Nearest high-dimensional neighbors: ${neighbors}`;
  el.pointInspector.classList.remove("hidden");
  draw();
}

function updateStats() {
  el.iterationStat.textContent = state.iteration;
  el.stressStat.textContent = state.stress.toFixed(3);
  el.dimensionStat.textContent = `${state.data.dims} -> 2`;
}

function project3D(points) {
  return points.map(([x, y, z]) => [
    x * 0.86 + z * 0.34,
    y * 0.9 - z * 0.28
  ]);
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
  return dist[i]
    .map((d, j) => ({ j, d }))
    .filter((item) => item.j !== i)
    .sort((a, b) => a.d - b.d)
    .slice(0, Math.min(k, dist.length - 1));
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
  for (const row of points) {
    for (let j = 0; j < dims; j += 1) mean[j] += row[j];
  }
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
  const denom = Math.max(1, points.length - 1);
  return cov.map((row) => row.map((value) => value / denom));
}

function powerIteration(matrix, steps) {
  let vector = Array(matrix.length).fill(0).map((_, i) => (i === 0 ? 1 : 0.3 / (i + 1)));
  vector = normalizeVector(vector);
  for (let s = 0; s < steps; s += 1) {
    vector = normalizeVector(matrixVector(matrix, vector));
  }
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
  for (const [x, y] of centered) {
    maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
  }
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

function jitteredCircle(n, seed) {
  const rand = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i += 1) {
    const angle = 2 * Math.PI * i / n;
    const radius = 0.04 + rand() * 0.02;
    points.push([
      Math.cos(angle) * radius + normal(rand) * 0.01,
      Math.sin(angle) * radius + normal(rand) * 0.01
    ]);
  }
  return points;
}

function toScreen(point, box) {
  const x = box.x + (point[0] * 0.42 + 0.5) * box.w;
  const y = box.y + (0.5 - point[1] * 0.42) * box.h;
  return [x, y];
}

function colorFor(t) {
  const hue = 210 - 170 * t;
  return `hsl(${hue}, 72%, 46%)`;
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

function lerp(a, b, t) {
  return a + (b - a) * t;
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
