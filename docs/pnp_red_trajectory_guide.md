# PnP and RED Reconstruction Trajectory Visualization

This optional module records long-run reconstruction trajectories:

```text
{x_t}_{t=0}^T
```

and residual trajectories:

```text
data residual: ||A x_t - y||
step residual: ||x_t - x_{t-1}||
```

The goal is to visualize inverse-problem algorithms as paths through image space.

## Algorithms

The public runner supports:

- RED gradient descent;
- PnP-HQS;
- PnP-ADMM;
- PnP-FISTA.

## Forward Models

Supported cases:

- denoising;
- Gaussian deblurring;
- motion deblurring;
- 2x super-resolution.

## Denoisers

Denoiser weights are **not stored in this repository**.

They are loaded at runtime through DeepInverse:

```bash
pip install -r requirements-inverse.txt
```

Then:

```bash
python scripts/run_pnp_red_trajectory.py --denoiser drunet --task gaussian_deblur --iters 2000
```

Supported denoiser names:

- `drunet`
- `dncnn`
- `diffunet`, if available in the installed DeepInverse version
- `median`, for a lightweight non-deep baseline

## Example Commands

Gaussian deblurring:

```bash
python scripts/run_pnp_red_trajectory.py --task gaussian_deblur --denoiser drunet --iters 2000
```

Motion deblurring:

```bash
python scripts/run_pnp_red_trajectory.py --task motion_deblur --denoiser drunet --iters 2000
```

2x super-resolution:

```bash
python scripts/run_pnp_red_trajectory.py --task sr_x2 --denoiser dncnn --iters 2000
```

## Outputs

The script writes:

- `residual_trajectories.png`
- `trajectory_embedding_pca.png`
- `trajectory_embedding_tsne.png`
- `reconstruction_snapshots.png`
- `summary.json`

These outputs are designed to answer:

- Which algorithm stabilizes fastest?
- Does acceleration create oscillations?
- Do RED and PnP methods follow similar paths?
- Is residual improvement aligned with perceptual improvement?
- Do trajectories cluster by algorithm or by forward model?

## Why This Belongs With Latent-Space Visualization

An inverse-problem algorithm is not just a final reconstruction. It is a sequence of image estimates.

By embedding the iterates, we can inspect the geometry of reconstruction dynamics:

```text
algorithm behavior becomes a visual trajectory
```

This is especially useful for comparing RED and plug-and-play methods with learned denoisers.
