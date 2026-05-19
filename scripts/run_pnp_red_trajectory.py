from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Record RED/PnP trajectories with DeepInverse denoisers.")
    parser.add_argument("--output", default="results_pnp_red")
    parser.add_argument("--image", default=None, help="Optional RGB image path. If omitted, DeepInverse cameraman is used.")
    parser.add_argument("--task", choices=["denoising", "gaussian_deblur", "motion_deblur", "sr_x2"], default="gaussian_deblur")
    parser.add_argument("--algorithms", nargs="+", default=["red", "pnp_hqs", "pnp_admm", "pnp_fista"])
    parser.add_argument("--denoiser", choices=["drunet", "dncnn", "diffunet", "median"], default="drunet")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--step", type=float, default=0.08)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def require_runtime():
    try:
        import torch
        import torch.nn.functional as F
        import deepinv as dinv
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from torchvision import transforms
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Install optional dependencies with `pip install -r requirements-inverse.txt`.") from exc
    return torch, F, dinv, peak_signal_noise_ratio, structural_similarity, transforms, Image


def load_image(args, torch, dinv, transforms, Image, device):
    if args.image is None:
        return dinv.utils.load_example("cameraman.png", img_size=args.img_size, grayscale=False, device=device)
    transform = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
    return transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)


def gaussian_kernel(torch, size, sigma, device):
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(torch, size, device):
    kernel = torch.zeros(size, size, device=device)
    kernel[size // 2, :] = 1.0 / size
    return kernel


def conv(F, x, kernel):
    c = x.shape[1]
    w = kernel[None, None].repeat(c, 1, 1, 1)
    p = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), w, groups=c)


class Operator:
    def __init__(self, task, torch, F, device):
        self.task = task
        self.torch = torch
        self.F = F
        self.device = device
        if task == "gaussian_deblur":
            self.kernel = gaussian_kernel(torch, 25, 1.6, device)
        elif task == "motion_deblur":
            self.kernel = motion_kernel(torch, 25, device)
        elif task == "sr_x2":
            self.kernel = gaussian_kernel(torch, 9, 1.0, device)
        else:
            self.kernel = None

    def A(self, x):
        if self.task == "denoising":
            return x
        if self.task in {"gaussian_deblur", "motion_deblur"}:
            return conv(self.F, x, self.kernel)
        if self.task == "sr_x2":
            return self.F.interpolate(conv(self.F, x, self.kernel), scale_factor=0.5, mode="bicubic", align_corners=False)
        raise ValueError(self.task)

    def AT(self, y, out_shape):
        if self.task == "denoising":
            return y
        if self.task in {"gaussian_deblur", "motion_deblur"}:
            return conv(self.F, y, self.torch.flip(self.kernel, dims=(-2, -1)))
        if self.task == "sr_x2":
            return self.F.interpolate(y, size=out_shape[-2:], mode="bicubic", align_corners=False)
        raise ValueError(self.task)

    def data_step(self, z, y, rho, steps=6, lr=0.18):
        x = z.clone()
        for _ in range(steps):
            grad = self.AT(self.A(x) - y, x.shape) + rho * (x - z)
            x = (x - lr * grad).clamp(0, 1)
        return x


def load_denoiser(name, dinv, torch, device):
    name = name.lower()
    if name == "median":
        model = dinv.models.MedianFilter()
    elif name == "drunet":
        model = dinv.models.DRUNet(pretrained="download", device=device)
    elif name == "dncnn":
        model = dinv.models.DnCNN(pretrained="download").to(device)
    elif name == "diffunet":
        if not hasattr(dinv.models, "DiffUNet"):
            raise RuntimeError("This DeepInverse version does not expose dinv.models.DiffUNet.")
        model = dinv.models.DiffUNet(pretrained="download").to(device)
    else:
        raise ValueError(name)
    return model.eval()


def denoise(torch, model, x, sigma):
    sig = torch.tensor([sigma], device=x.device, dtype=x.dtype)
    with torch.no_grad():
        try:
            return model(x, sigma=sig).clamp(0, 1)
        except TypeError:
            try:
                return model(x, sig).clamp(0, 1)
            except TypeError:
                return model(x).clamp(0, 1)


def record_features(torch, x):
    arr = x.detach().cpu()[0].permute(1, 2, 0).numpy()
    gray = arr.mean(axis=2)
    feats = [arr.mean(), arr.std(), arr.min(), arr.max()]
    for i in range(4):
        for j in range(4):
            patch = gray[i * gray.shape[0] // 4 : (i + 1) * gray.shape[0] // 4, j * gray.shape[1] // 4 : (j + 1) * gray.shape[1] // 4]
            feats.extend([patch.mean(), patch.std()])
    return np.asarray(feats, dtype=np.float32)


def tensor_image(x):
    return x.detach().cpu()[0].permute(1, 2, 0).numpy().clip(0, 1)


def run_algorithm(name, torch, op, denoiser, y, x0, args):
    x = x0.clone()
    z = x0.clone()
    u = torch.zeros_like(x0)
    q = x0.clone()
    t = torch.tensor(1.0, device=x0.device)
    features, residuals, step_residuals, snapshots = [], [], [], []
    start = time.perf_counter()
    for it in range(args.iters + 1):
        if it % args.save_every == 0 or it == args.iters:
            features.append(record_features(torch, x))
            residuals.append(float(torch.linalg.norm((op.A(x) - y).reshape(-1)).detach().cpu()))
            snapshots.append(x.detach().cpu())
        x_prev = x.clone()
        if name == "red":
            dx = denoise(torch, denoiser, x, args.noise_std)
            grad = op.AT(op.A(x) - y, x.shape) + args.lam * (x - dx)
            x = (x - args.step * grad).clamp(0, 1)
        elif name == "pnp_hqs":
            x = op.data_step(z, y, args.rho)
            z = denoise(torch, denoiser, x, args.noise_std)
            x = z
        elif name == "pnp_admm":
            x = op.data_step(z - u, y, args.rho)
            z = denoise(torch, denoiser, x + u, args.noise_std)
            u = u + x - z
            x = z
        elif name == "pnp_fista":
            grad = op.AT(op.A(q) - y, q.shape)
            x_next = denoise(torch, denoiser, (q - args.step * grad).clamp(0, 1), args.noise_std)
            t_next = (1 + torch.sqrt(1 + 4 * t * t)) / 2
            q = x_next + ((t - 1) / t_next) * (x_next - x)
            x = x_next
            t = t_next
        else:
            raise ValueError(name)
        step_residuals.append(float(torch.linalg.norm((x - x_prev).reshape(-1)).detach().cpu()))
    return {
        "name": name,
        "features": np.vstack(features),
        "residuals": residuals,
        "step_residuals": step_residuals,
        "snapshots": snapshots,
        "final": x.detach().cpu(),
        "seconds": time.perf_counter() - start,
    }


def plot_residuals(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for res in results:
        axes[0].plot(res["residuals"], label=res["name"])
        axes[1].plot(res["step_residuals"], label=res["name"], alpha=0.9)
    axes[0].set_title("Data residual ||A x_t - y||")
    axes[1].set_title("Step residual ||x_t - x_{t-1}||")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("recorded iteration index")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "residual_trajectories.png")
    plt.close(fig)


def plot_embedding(results, out):
    X = np.vstack([r["features"] for r in results])
    X = StandardScaler().fit_transform(X)
    slices = {}
    start = 0
    for r in results:
        slices[r["name"]] = slice(start, start + len(r["features"]))
        start += len(r["features"])
    reducers = {
        "pca": PCA(n_components=2, random_state=42),
        "tsne": TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 8)), init="pca", learning_rate="auto", random_state=42),
    }
    for method, reducer in reducers.items():
        Y = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
        for name, sl in slices.items():
            pts = Y[sl]
            ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2.4, linewidth=1.1, label=name)
            ax.scatter(pts[0, 0], pts[0, 1], marker="s", s=45)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=90)
        ax.set_title(f"Reconstruction trajectory embedded with {method.upper()}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"trajectory_embedding_{method}.png")
        plt.close(fig)


def plot_snapshots(results, x_true, y, out):
    rows = len(results) + 1
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), dpi=150)
    axes[0, 0].imshow(tensor_image(x_true))
    axes[0, 0].set_title("ground truth")
    axes[0, 1].imshow(tensor_image(y if y.shape == x_true.shape else x_true * 0 + x_true.mean()))
    axes[0, 1].set_title("measurement view")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    for col in range(2):
        axes[0, col].axis("off")
    for row, res in enumerate(results, start=1):
        picks = [0, len(res["snapshots"]) // 3, 2 * len(res["snapshots"]) // 3, len(res["snapshots"]) - 1]
        for col, idx in enumerate(picks):
            axes[row, col].imshow(tensor_image(res["snapshots"][idx]))
            axes[row, col].set_title(f"{res['name']} record {idx}")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "reconstruction_snapshots.png")
    plt.close(fig)


def main():
    args = parse_args()
    torch, F, dinv, psnr, ssim, transforms, Image = require_runtime()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    x_true = load_image(args, torch, dinv, transforms, Image, device)
    op = Operator(args.task, torch, F, device)
    denoiser = load_denoiser(args.denoiser, dinv, torch, device)
    noise = args.noise_std * torch.randn_like(op.A(x_true))
    y = (op.A(x_true) + noise).clamp(0, 1)
    x0 = op.AT(y, x_true.shape).clamp(0, 1)
    results = [run_algorithm(name, torch, op, denoiser, y, x0, args) for name in args.algorithms]
    plot_residuals(results, out)
    plot_embedding(results, out)
    plot_snapshots(results, x_true.detach().cpu(), y.detach().cpu(), out)
    summary = []
    truth = tensor_image(x_true)
    for res in results:
        pred = tensor_image(res["final"])
        summary.append({
            "algorithm": res["name"],
            "task": args.task,
            "denoiser": args.denoiser,
            "iterations": args.iters,
            "seconds": res["seconds"],
            "psnr": float(psnr(truth, pred, data_range=1.0)),
            "ssim": float(ssim(truth, pred, channel_axis=2, data_range=1.0)),
            "final_data_residual": res["residuals"][-1],
            "final_step_residual": res["step_residuals"][-1],
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved PnP/RED trajectory visualizations to {out}")


if __name__ == "__main__":
    main()
