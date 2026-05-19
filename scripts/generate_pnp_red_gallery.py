from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_pnp_red_gallery"
OUT.mkdir(exist_ok=True)


def make_image(n=96):
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.zeros((n, n, 3), dtype=np.float32)
    img[..., 0] = 0.25 + 0.55 * np.exp(-((x - 0.35) ** 2 + (y - 0.35) ** 2) / 0.035)
    img[..., 1] = 0.2 + 0.55 * np.exp(-((x - 0.68) ** 2 + (y - 0.43) ** 2) / 0.045)
    img[..., 2] = 0.2 + 0.45 * (np.sin(10 * x) * np.cos(8 * y) + 1) / 2
    img[18:42, 58:82, 1] = 0.95
    img[58:82, 18:43, 0] = 0.9
    return np.clip(img, 0, 1)


def blur(x, kind="gaussian"):
    if kind == "identity":
        return x
    if kind == "sr_x2":
        low = x[::2, ::2]
        return np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)[: x.shape[0], : x.shape[1]]
    if kind == "motion":
        out = np.zeros_like(x)
        for shift in range(-5, 6):
            out += np.roll(x, shift, axis=1)
        return out / 11
    out = x.copy()
    for _ in range(3):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5
    return out


def denoise(x, profile="drunet"):
    if profile == "dncnn":
        alpha, rounds = 0.45, 1
    elif profile == "diffunet":
        alpha, rounds = 0.72, 3
    else:
        alpha, rounds = 0.58, 2
    smooth = x.copy()
    for _ in range(rounds):
        smooth = (
            smooth
            + np.roll(smooth, 1, axis=0)
            + np.roll(smooth, -1, axis=0)
            + np.roll(smooth, 1, axis=1)
            + np.roll(smooth, -1, axis=1)
        ) / 5
    detail = x - smooth
    return np.clip(alpha * smooth + (1 - alpha) * x + 0.08 * detail, 0, 1)


def data_grad(x, y, task):
    return blur(blur(x, task) - y, task)


def features(x):
    gray = x.mean(axis=2)
    feats = [x.mean(), x.std(), x.min(), x.max()]
    for i in range(4):
        for j in range(4):
            patch = gray[i * gray.shape[0] // 4 : (i + 1) * gray.shape[0] // 4, j * gray.shape[1] // 4 : (j + 1) * gray.shape[1] // 4]
            feats.extend([patch.mean(), patch.std()])
    return np.asarray(feats)


def run_case(task, denoiser, steps=420, record_every=7, seed=3):
    rng = np.random.default_rng(seed)
    truth = make_image()
    y = np.clip(blur(truth, task) + rng.normal(0, 0.025, truth.shape), 0, 1)
    x0 = y.copy()
    if task == "sr_x2":
        x0 = blur(y, "sr_x2")
    traces = {}
    for algo in ["RED", "PnP-HQS", "PnP-ADMM", "PnP-FISTA"]:
        x = x0.copy()
        z = x0.copy()
        u = np.zeros_like(x0)
        q = x0.copy()
        t = 1.0
        xs, residual, step_residual, feat = [], [], [], []
        for it in range(steps + 1):
            if it % record_every == 0 or it == steps:
                xs.append(x.copy())
                residual.append(float(np.linalg.norm((blur(x, task) - y).ravel())))
                feat.append(features(x))
            x_prev = x.copy()
            if algo == "RED":
                dx = denoise(x, denoiser)
                x = np.clip(x - 0.18 * (data_grad(x, y, task) + 0.14 * (x - dx)), 0, 1)
            elif algo == "PnP-HQS":
                x = np.clip(z - 0.2 * data_grad(z, y, task), 0, 1)
                z = denoise(x, denoiser)
                x = z
            elif algo == "PnP-ADMM":
                x = np.clip(z - u - 0.18 * data_grad(z - u, y, task), 0, 1)
                z = denoise(x + u, denoiser)
                u = u + x - z
                x = z
            else:
                grad = data_grad(q, y, task)
                x_next = denoise(np.clip(q - 0.18 * grad, 0, 1), denoiser)
                t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
                q = x_next + ((t - 1) / t_next) * (x_next - x)
                x = x_next
                t = t_next
            step_residual.append(float(np.linalg.norm((x - x_prev).ravel())))
        traces[algo] = {"snapshots": xs, "residual": residual, "step_residual": step_residual, "features": np.vstack(feat)}
    return truth, y, traces


def save_snapshots(task, denoiser, truth, y, traces):
    fig, axes = plt.subplots(5, 4, figsize=(11, 13), dpi=150)
    axes[0, 0].imshow(truth)
    axes[0, 0].set_title("truth")
    axes[0, 1].imshow(y)
    axes[0, 1].set_title("measurement")
    for ax in axes[0, 2:]:
        ax.axis("off")
    for ax in axes[0, :2]:
        ax.axis("off")
    for row, (algo, trace) in enumerate(traces.items(), start=1):
        picks = [0, len(trace["snapshots"]) // 3, 2 * len(trace["snapshots"]) // 3, -1]
        for col, idx in enumerate(picks):
            axes[row, col].imshow(trace["snapshots"][idx])
            axes[row, col].set_title(f"{algo} record {idx}")
            axes[row, col].axis("off")
    fig.suptitle(f"{task} with {denoiser}-style denoising: reconstruction snapshots")
    fig.tight_layout()
    fig.savefig(OUT / f"{task}_{denoiser}_snapshots.png")
    plt.close(fig)


def save_residuals(task, denoiser, traces):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for algo, trace in traces.items():
        axes[0].plot(trace["residual"], label=algo)
        axes[1].plot(trace["step_residual"], label=algo, alpha=0.9)
    axes[0].set_title("Data residual ||A x_t - y||")
    axes[1].set_title("Step residual ||x_t - x_{t-1}||")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("recorded iteration index")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"{task} with {denoiser}-style denoising")
    fig.tight_layout()
    fig.savefig(OUT / f"{task}_{denoiser}_residuals.png")
    plt.close(fig)


def save_embeddings(task, denoiser, traces):
    X = np.vstack([trace["features"] for trace in traces.values()])
    X = StandardScaler().fit_transform(X)
    slices = {}
    start = 0
    for algo, trace in traces.items():
        slices[algo] = slice(start, start + len(trace["features"]))
        start += len(trace["features"])
    reducers = {
        "pca": PCA(n_components=2, random_state=42),
        "tsne": TSNE(n_components=2, perplexity=12, init="pca", learning_rate="auto", random_state=42),
    }
    for method, reducer in reducers.items():
        Y = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        for algo, sl in slices.items():
            pts = Y[sl]
            ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2.5, linewidth=1.2, label=algo)
            ax.scatter(pts[0, 0], pts[0, 1], marker="s", s=50)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=90)
        ax.set_title(f"{task} / {denoiser}: trajectory embedding ({method.upper()})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"{task}_{denoiser}_trajectory_{method}.png")
        plt.close(fig)


def save_case_matrix(cases):
    fig, axes = plt.subplots(len(cases), 4, figsize=(12, 3 * len(cases)), dpi=150)
    for row, (task, denoiser, truth, y, traces) in enumerate(cases):
        axes[row, 0].imshow(y)
        axes[row, 0].set_title(f"{task}\nmeasurement")
        for col, algo in enumerate(["RED", "PnP-HQS", "PnP-ADMM"], start=1):
            axes[row, col].imshow(traces[algo]["snapshots"][-1])
            axes[row, col].set_title(f"{denoiser}\n{algo} final")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("PnP/RED public gallery: forward models, denoiser profiles, algorithm outcomes")
    fig.tight_layout()
    fig.savefig(OUT / "pnp_red_case_matrix.png")
    plt.close(fig)


def main():
    configs = [
        ("gaussian_deblur", "drunet"),
        ("motion", "dncnn"),
        ("sr_x2", "diffunet"),
    ]
    cases = []
    lines = [
        "# PnP/RED Trajectory Gallery",
        "",
        "Generated by `python scripts/generate_pnp_red_gallery.py`.",
        "",
        "These are lightweight reproducible visual demonstrators. For pretrained DeepInverse denoisers, use `scripts/run_pnp_red_trajectory.py`.",
        "",
        "| Case | Snapshots | Residuals | PCA trajectory | t-SNE trajectory |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task, denoiser in configs:
        truth, y, traces = run_case(task, denoiser)
        save_snapshots(task, denoiser, truth, y, traces)
        save_residuals(task, denoiser, traces)
        save_embeddings(task, denoiser, traces)
        cases.append((task, denoiser, truth, y, traces))
        prefix = f"{task}_{denoiser}"
        lines.append(
            f"| {task} / {denoiser} | `{prefix}_snapshots.png` | `{prefix}_residuals.png` | `{prefix}_trajectory_pca.png` | `{prefix}_trajectory_tsne.png` |"
        )
    save_case_matrix(cases)
    lines.append("")
    lines.append("Also see `pnp_red_case_matrix.png` for a compact cross-case overview.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
