from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_3d"
OUT.mkdir(exist_ok=True)


def swiss_roll(n=520, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    angle = 1.5 * np.pi + 3.4 * np.pi * t
    x = angle * np.cos(angle) / 8
    y = 2.2 * (rng.random(n) - 0.5)
    z = angle * np.sin(angle) / 8
    pts = np.column_stack([x, y, z])
    features = np.column_stack([pts, np.sin(angle)])
    return normalize(pts), normalize(features), t


def helix(n=520):
    t = np.linspace(0, 1, n)
    angle = 8 * np.pi * t
    pts = np.column_stack([np.cos(angle), 2.2 * (t - 0.5), np.sin(angle)])
    features = np.column_stack([pts, np.cos(2 * angle)])
    return normalize(pts), normalize(features), t


def lorenz(n=620, discard=120, dt=0.01):
    p = np.array([0.1, 1.0, 1.05], dtype=float)
    points = []

    def f(v):
        x, y, z = v
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z])

    for _ in range(n + discard):
        k1 = f(p)
        k2 = f(p + 0.5 * dt * k1)
        k3 = f(p + 0.5 * dt * k2)
        k4 = f(p + dt * k3)
        p = p + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        points.append(p.copy())

    pts = np.asarray(points[discard:])
    t = np.linspace(0, 1, len(pts))
    features = np.column_stack([pts, np.sin(np.arange(len(pts)) * 0.05)])
    return normalize(pts), normalize(features), t


def hypercube(n=640, seed=9):
    rng = np.random.default_rng(seed)
    corners = []
    labels = []
    for i in range(16):
        bits = np.array([1 if i & (1 << j) else -1 for j in range(4)], dtype=float)
        for _ in range(n // 16):
            corners.append(bits + rng.normal(0, 0.08, size=4))
            labels.append((bits[3] + 1) / 2)
    features = np.asarray(corners)
    pts = np.column_stack([
        features[:, 0] + 0.35 * features[:, 3],
        features[:, 1] - 0.25 * features[:, 3],
        features[:, 2] + 0.2 * features[:, 3],
    ])
    return normalize(pts), normalize(features), np.asarray(labels)


def normalize(x):
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(x)) or 1
    return x / scale


def projection_target(features):
    return normalize(PCA(n_components=2, random_state=42).fit_transform(features))


def knn_edges(features, k=5, limit=1100):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(return_distance=False)
    edges = []
    for i, row in enumerate(idx):
        for j in row[1:]:
            if i < j:
                edges.append((i, j))
                if len(edges) >= limit:
                    return edges
    return edges


def plot_3d_static(name, pts, features, labels):
    edges = knn_edges(features)
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for i, j in edges[::3]:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]], color="#0f766e", alpha=0.035, linewidth=0.6)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=labels, cmap="viridis", s=9, alpha=0.88, depthshade=True)
    ax.view_init(elev=24, azim=-55)
    ax.set_title(f"{name}: original 3D / high-dimensional shadow")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_3d_static.png")
    plt.close(fig)


def plot_projection_static(name, pts, features, labels):
    target = projection_target(features)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    axes[0].scatter(pts[:, 0], pts[:, 2], c=labels, cmap="viridis", s=10, alpha=0.85)
    axes[0].set_title("3D shadow: x-z view")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].scatter(target[:, 0], target[:, 1], c=labels, cmap="viridis", s=10, alpha=0.85)
    axes[1].set_title("PCA target projection")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle(f"{name}: original geometry versus 2D projection")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_projection_comparison.png")
    plt.close(fig)


def animate_rotation(name, pts, features, labels):
    edges = knn_edges(features, limit=700)
    fig = plt.figure(figsize=(7, 5.8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=12)
    path = OUT / f"{name}_3d_rotation.gif"
    with writer.saving(fig, path, dpi=120):
        for frame in range(72):
            ax.clear()
            for i, j in edges[::4]:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]], color="#0f766e", alpha=0.035, linewidth=0.5)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=labels, cmap="viridis", s=9, alpha=0.88)
            ax.view_init(elev=24, azim=frame * 5)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            ax.set_title(f"{name}: orbiting original geometry")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            writer.grab_frame()
    plt.close(fig)


def animate_collapse(name, pts, features, labels):
    target = projection_target(features)
    target3 = np.column_stack([target[:, 0], target[:, 1], np.zeros(len(target))])
    fig = plt.figure(figsize=(7, 5.8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=10)
    path = OUT / f"{name}_collapse_to_2d.gif"
    with writer.saving(fig, path, dpi=120):
        phases = np.r_[np.linspace(0, 1, 36), np.linspace(1, 0, 36)]
        for t in phases:
            current = (1 - t) * pts + t * target3
            ax.clear()
            ax.scatter(current[:, 0], current[:, 1], current[:, 2], c=labels, cmap="viridis", s=10, alpha=0.88)
            ax.view_init(elev=24 - 10 * t, azim=-55 + 20 * t)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_zlim(-1.15, 1.15)
            ax.set_title(f"{name}: 3D/high-D shadow -> PCA plane ({t:.0%})")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            writer.grab_frame()
    plt.close(fig)


def main():
    datasets = {
        "swiss": swiss_roll(),
        "helix": helix(),
        "lorenz": lorenz(),
        "hypercube4d": hypercube(),
    }
    lines = [
        "# 3D and Higher-Dimensional Artifacts",
        "",
        "Generated by `python scripts/generate_3d_artifacts.py`.",
        "",
        "| Dataset | Static 3D | Projection Comparison | Rotation GIF | Collapse GIF |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, (pts, features, labels) in datasets.items():
        plot_3d_static(name, pts, features, labels)
        plot_projection_static(name, pts, features, labels)
        animate_rotation(name, pts, features, labels)
        animate_collapse(name, pts, features, labels)
        lines.append(
            f"| {name} | `{name}_3d_static.png` | `{name}_projection_comparison.png` | `{name}_3d_rotation.gif` | `{name}_collapse_to_2d.gif` |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
