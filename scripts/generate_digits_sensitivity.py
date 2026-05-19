from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.animation import PillowWriter
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_digits"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_METRIC = 15


def load_data():
    digits = load_digits()
    x = StandardScaler().fit_transform(digits.data)
    y = digits.target
    return x, y


def neighbor_overlap(x_high, x_low, k=15):
    high = NearestNeighbors(n_neighbors=k + 1).fit(x_high).kneighbors(return_distance=False)[:, 1:]
    low = NearestNeighbors(n_neighbors=k + 1).fit(x_low).kneighbors(return_distance=False)[:, 1:]
    return np.mean([len(set(a).intersection(b)) / k for a, b in zip(high, low)])


def run_reducer(name, reducer, x, labels, params):
    start = time.perf_counter()
    embedding = reducer.fit_transform(x)
    elapsed = time.perf_counter() - start
    metrics = {
        "method": name,
        **params,
        "seconds": elapsed,
        "knn_overlap": neighbor_overlap(x, embedding, K_METRIC),
        "trustworthiness": trustworthiness(x, embedding, n_neighbors=K_METRIC),
    }
    return embedding, metrics


def plot_embedding(ax, embedding, labels, title, metrics=None):
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.82,
        linewidth=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if metrics:
        subtitle = f"kNN={metrics['knn_overlap']:.3f}  trust={metrics['trustworthiness']:.3f}  {metrics['seconds']:.1f}s"
        ax.text(
            0.02,
            0.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )


def save_single(path, embedding, labels, title, metrics):
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    plot_embedding(ax, embedding, labels, title, metrics)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_grid(path, items, labels, suptitle):
    cols = min(4, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), dpi=150)
    axes = np.array(axes).reshape(-1)
    for ax, item in zip(axes, items):
        plot_embedding(ax, item["embedding"], labels, item["title"], item["metrics"])
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def align_embedding(reference, target):
    ref = reference - reference.mean(axis=0)
    tgt = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(tgt.T @ ref)
    aligned = tgt @ (u @ vt)
    ref_scale = np.sqrt((ref ** 2).sum())
    tgt_scale = np.sqrt((aligned ** 2).sum()) or 1
    return aligned * (ref_scale / tgt_scale)


def save_sweep_animation(path, items, labels, title):
    reference = items[0]["embedding"]
    aligned = []
    for item in items:
        aligned.append({**item, "embedding": align_embedding(reference, item["embedding"])})

    all_points = np.vstack([item["embedding"] for item in aligned])
    xlim = np.percentile(all_points[:, 0], [1, 99])
    ylim = np.percentile(all_points[:, 1], [1, 99])
    xpad = 0.08 * (xlim[1] - xlim[0])
    ypad = 0.08 * (ylim[1] - ylim[0])

    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=120)

    def draw_frame(frame_index):
        ax.clear()
        item = aligned[frame_index % len(aligned)]
        plot_embedding(ax, item["embedding"], labels, f"{title}\n{item['title']}", item["metrics"])
        ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
        ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)

    writer = PillowWriter(fps=0.8)
    with writer.saving(fig, path, dpi=120):
        for frame in range(len(aligned) * 2):
            draw_frame(frame)
            writer.grab_frame()
    plt.close(fig)


def save_umap_graph_explainer(path, x, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x)
    ks = [5, 15, 50]
    fig, axes = plt.subplots(1, len(ks), figsize=(15, 4.5), dpi=150)
    subset = np.arange(160)
    for ax, k in zip(axes, ks):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
        indices = nn.kneighbors(return_distance=False)
        ax.scatter(pca[subset, 0], pca[subset, 1], c=labels[subset], cmap="tab10", s=16, zorder=2)
        edge_count = 0
        for i in subset:
            for j in indices[i, 1:4]:
                if j in subset:
                    ax.plot(
                        [pca[i, 0], pca[j, 0]],
                        [pca[i, 1], pca[j, 1]],
                        color="#0f766e",
                        alpha=0.09,
                        linewidth=0.8,
                        zorder=1,
                    )
                    edge_count += 1
        ax.set_title(f"UMAP graph intuition: k={k}\nfirst 3 neighbor edges shown per point")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.02,
            f"larger k = broader graph context\nsample edges drawn: {edge_count}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
    x, labels = load_data()
    rows = []
    browser_embeddings = {
        "dataset": {
            "name": "sklearn_digits",
            "description": "Built-in scikit-learn handwritten digits dataset.",
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "label_names": [str(i) for i in range(10)],
        },
        "labels": labels.astype(int).tolist(),
        "embeddings": [],
    }

    pca_embedding, pca_metrics = run_reducer(
        "PCA",
        PCA(n_components=2, random_state=RANDOM_STATE),
        x,
        labels,
        {"hyperparameter": "n_components", "value": 2},
    )
    rows.append(pca_metrics)
    browser_embeddings["embeddings"].append(to_browser_embedding("PCA baseline", "PCA", "n_components", 2, pca_embedding, pca_metrics))
    save_single(OUT / "digits_pca.png", pca_embedding, labels, "Digits PCA baseline", pca_metrics)

    tsne_items = []
    for perplexity in [5, 15, 30, 50]:
        embedding, metrics = run_reducer(
            "t-SNE",
            TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=RANDOM_STATE,
                max_iter=1000,
            ),
            x,
            labels,
            {"hyperparameter": "perplexity", "value": perplexity},
        )
        rows.append(metrics)
        browser_embeddings["embeddings"].append(to_browser_embedding(f"t-SNE perplexity={perplexity}", "t-SNE", "perplexity", perplexity, embedding, metrics))
        tsne_items.append({"embedding": embedding, "metrics": metrics, "title": f"perplexity={perplexity}"})
        save_single(OUT / f"digits_tsne_perplexity_{perplexity}.png", embedding, labels, f"t-SNE perplexity={perplexity}", metrics)

    umap_k_items = []
    for k in [5, 15, 50, 100]:
        embedding, metrics = run_reducer(
            "UMAP",
            umap.UMAP(
                n_components=2,
                n_neighbors=k,
                min_dist=0.1,
                metric="euclidean",
                random_state=RANDOM_STATE,
            ),
            x,
            labels,
            {"hyperparameter": "n_neighbors", "value": k},
        )
        rows.append(metrics)
        browser_embeddings["embeddings"].append(to_browser_embedding(f"UMAP n_neighbors={k}", "UMAP", "n_neighbors", k, embedding, metrics))
        umap_k_items.append({"embedding": embedding, "metrics": metrics, "title": f"n_neighbors={k}"})
        save_single(OUT / f"digits_umap_neighbors_{k}.png", embedding, labels, f"UMAP n_neighbors={k}", metrics)

    umap_dist_items = []
    for min_dist in [0.0, 0.1, 0.35, 0.7]:
        embedding, metrics = run_reducer(
            "UMAP",
            umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=min_dist,
                metric="euclidean",
                random_state=RANDOM_STATE,
            ),
            x,
            labels,
            {"hyperparameter": "min_dist", "value": min_dist},
        )
        rows.append(metrics)
        browser_embeddings["embeddings"].append(to_browser_embedding(f"UMAP min_dist={min_dist}", "UMAP", "min_dist", min_dist, embedding, metrics))
        umap_dist_items.append({"embedding": embedding, "metrics": metrics, "title": f"min_dist={min_dist}"})
        save_single(OUT / f"digits_umap_min_dist_{min_dist}.png", embedding, labels, f"UMAP min_dist={min_dist}", metrics)

    save_grid(OUT / "digits_tsne_perplexity_grid.png", tsne_items, labels, "t-SNE sensitivity to perplexity")
    save_grid(OUT / "digits_umap_neighbors_grid.png", umap_k_items, labels, "UMAP sensitivity to n_neighbors")
    save_grid(OUT / "digits_umap_min_dist_grid.png", umap_dist_items, labels, "UMAP sensitivity to min_dist")
    save_sweep_animation(OUT / "digits_tsne_perplexity_sweep.gif", tsne_items, labels, "t-SNE perplexity sweep")
    save_sweep_animation(OUT / "digits_umap_neighbors_sweep.gif", umap_k_items, labels, "UMAP n_neighbors sweep")
    save_sweep_animation(OUT / "digits_umap_min_dist_sweep.gif", umap_dist_items, labels, "UMAP min_dist sweep")
    save_umap_graph_explainer(OUT / "digits_umap_graph_construction.png", x, labels)

    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "hyperparameter", "value", "seconds", "knn_overlap", "trustworthiness"])
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "embeddings.json").write_text(json.dumps(browser_embeddings, separators=(",", ":")), encoding="utf-8")

    lines = [
        "# Digits Hyperparameter Sensitivity",
        "",
        "Generated by `python scripts/generate_digits_sensitivity.py` using the built-in scikit-learn handwritten digits dataset.",
        "",
        "| Method | Hyperparameter | Value | Seconds | kNN overlap | Trustworthiness |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['hyperparameter']} | {row['value']} | "
            f"{row['seconds']:.2f} | {row['knn_overlap']:.3f} | {row['trustworthiness']:.3f} |"
        )
    lines.extend(
        [
            "",
            "How to read this:",
            "",
            "- t-SNE perplexity changes the effective neighborhood size.",
            "- UMAP n_neighbors changes the graph scale from local detail to broader structure.",
            "- UMAP min_dist changes how tightly clusters are allowed to pack.",
            "- kNN overlap and trustworthiness measure local-neighborhood preservation, not global truth.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_browser_embedding(title, method, hyperparameter, value, embedding, metrics):
    rounded = np.round(embedding.astype(float), 5)
    return {
        "id": f"{method.lower().replace('-', '').replace(' ', '_')}_{hyperparameter}_{value}",
        "title": title,
        "method": method,
        "hyperparameter": hyperparameter,
        "value": value,
        "metrics": {
            "seconds": round(float(metrics["seconds"]), 4),
            "knn_overlap": round(float(metrics["knn_overlap"]), 5),
            "trustworthiness": round(float(metrics["trustworthiness"]), 5),
        },
        "points": rounded.tolist(),
    }


if __name__ == "__main__":
    main()
