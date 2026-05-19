"use strict";

const state = {
  data: null,
  selected: ["", "", ""],
  hoverIndex: -1
};

const panels = [
  { select: "selectA", canvas: "canvasA", title: "titleA", metric: "metricA" },
  { select: "selectB", canvas: "canvasB", title: "titleB", metric: "metricB" },
  { select: "selectC", canvas: "canvasC", title: "titleC", metric: "metricC" }
];

const el = {};

window.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  await loadData();
  bindControls();
  populateControls();
  renderTable();
  resizeCanvases();
  window.addEventListener("resize", resizeCanvases);
});

function cacheElements() {
  ["linkHover", "showLabels", "metricsBody", "tooltip", ...panels.flatMap((p) => [p.select, p.canvas, p.title, p.metric])].forEach((id) => {
    el[id] = document.getElementById(id);
  });
}

async function loadData() {
  const response = await fetch("results_digits/embeddings.json");
  if (!response.ok) {
    throw new Error("Could not load results_digits/embeddings.json. Run python scripts/generate_digits_sensitivity.py first.");
  }
  state.data = await response.json();
}

function bindControls() {
  panels.forEach((panel, index) => {
    el[panel.select].addEventListener("change", () => {
      state.selected[index] = el[panel.select].value;
      drawAll();
    });

    el[panel.canvas].addEventListener("mousemove", (event) => handleHover(event, index));
    el[panel.canvas].addEventListener("mouseleave", () => {
      state.hoverIndex = -1;
      el.tooltip.classList.add("hidden");
      drawAll();
    });
  });

  el.linkHover.addEventListener("change", drawAll);
  el.showLabels.addEventListener("change", drawAll);
}

function populateControls() {
  const embeddings = state.data.embeddings;
  const defaults = [
    findId("PCA", "n_components", 2),
    findId("t-SNE", "perplexity", 30),
    findId("UMAP", "n_neighbors", 15)
  ];

  panels.forEach((panel, index) => {
    el[panel.select].innerHTML = embeddings.map((embedding) => {
      return `<option value="${embedding.id}">${embedding.title}</option>`;
    }).join("");
    state.selected[index] = defaults[index] || embeddings[index].id;
    el[panel.select].value = state.selected[index];
  });
}

function findId(method, hyperparameter, value) {
  const match = state.data.embeddings.find((embedding) => {
    return embedding.method === method && embedding.hyperparameter === hyperparameter && Number(embedding.value) === Number(value);
  });
  return match?.id;
}

function resizeCanvases() {
  for (const panel of panels) {
    const canvas = el[panel.canvas];
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(420, Math.floor(rect.width * dpr));
    canvas.height = Math.max(340, Math.floor(rect.height * dpr));
  }
  drawAll();
}

function drawAll() {
  panels.forEach((panel, index) => drawPanel(panel, state.selected[index]));
}

function drawPanel(panel, id) {
  const embedding = getEmbedding(id);
  if (!embedding) return;

  const canvas = el[panel.canvas];
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  el[panel.title].textContent = embedding.title;
  el[panel.metric].textContent = `kNN ${embedding.metrics.knn_overlap.toFixed(3)} | trust ${embedding.metrics.trustworthiness.toFixed(3)} | ${embedding.metrics.seconds.toFixed(2)}s`;

  const bounds = getBounds(embedding.points);
  drawAxes(ctx, w, h);
  drawPoints(ctx, embedding, bounds, w, h);
}

function drawAxes(ctx, w, h) {
  const pad = Math.min(w, h) * 0.08;
  ctx.save();
  ctx.strokeStyle = "rgba(15, 118, 110, 0.16)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.restore();
}

function drawPoints(ctx, embedding, bounds, w, h) {
  const labels = state.data.labels;
  ctx.save();
  for (let i = 0; i < embedding.points.length; i += 1) {
    const [x, y] = toScreen(embedding.points[i], bounds, w, h);
    const active = el.linkHover.checked && i === state.hoverIndex;
    ctx.beginPath();
    ctx.arc(x, y, active ? 8 : 4.2, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(labels[i]);
    ctx.fill();
    ctx.strokeStyle = active ? "rgba(180, 83, 9, 0.95)" : "rgba(255,255,255,0.88)";
    ctx.lineWidth = active ? 3 : 1.2;
    ctx.stroke();

    if (el.showLabels.checked && active) {
      ctx.fillStyle = "#17202a";
      ctx.font = `${Math.max(12, Math.round(w / 55))}px system-ui, sans-serif`;
      ctx.fillText(String(labels[i]), x + 9, y - 9);
    }
  }
  ctx.restore();
}

function handleHover(event, panelIndex) {
  const embedding = getEmbedding(state.selected[panelIndex]);
  if (!embedding) return;

  const canvas = el[panels[panelIndex].canvas];
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const y = (event.clientY - rect.top) * dpr;
  const bounds = getBounds(embedding.points);

  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < embedding.points.length; i += 1) {
    const point = toScreen(embedding.points[i], bounds, canvas.width, canvas.height);
    const dist = (point[0] - x) ** 2 + (point[1] - y) ** 2;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }

  if (bestDist > (15 * dpr) ** 2) {
    state.hoverIndex = -1;
    el.tooltip.classList.add("hidden");
  } else {
    state.hoverIndex = best;
    el.tooltip.innerHTML = `<strong>Digit ${state.data.labels[best]}</strong><br>Point ${best}<br>${embedding.title}`;
    el.tooltip.style.left = `${event.clientX + 14}px`;
    el.tooltip.style.top = `${event.clientY + 14}px`;
    el.tooltip.classList.remove("hidden");
  }
  drawAll();
}

function renderTable() {
  el.metricsBody.innerHTML = state.data.embeddings.map((embedding) => {
    return `<tr>
      <td>${embedding.title}</td>
      <td>${embedding.method}</td>
      <td>${embedding.hyperparameter}=${embedding.value}</td>
      <td>${embedding.metrics.seconds.toFixed(2)}</td>
      <td>${embedding.metrics.knn_overlap.toFixed(3)}</td>
      <td>${embedding.metrics.trustworthiness.toFixed(3)}</td>
    </tr>`;
  }).join("");
}

function getEmbedding(id) {
  return state.data.embeddings.find((embedding) => embedding.id === id);
}

function getBounds(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xPad = (maxX - minX || 1) * 0.08;
  const yPad = (maxY - minY || 1) * 0.08;
  return { minX: minX - xPad, maxX: maxX + xPad, minY: minY - yPad, maxY: maxY + yPad };
}

function toScreen(point, bounds, w, h) {
  const pad = Math.min(w, h) * 0.08;
  const x = pad + ((point[0] - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (w - 2 * pad);
  const y = h - pad - ((point[1] - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (h - 2 * pad);
  return [x, y];
}

function colorFor(label) {
  const colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#b45309"];
  return colors[label % colors.length];
}
