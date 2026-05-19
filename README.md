# Latent Space Visualization Lab

An interactive, research-friendly learning lab for understanding **PCA, SNE, t-SNE, UMAP, and modern latent-space visualization**.

This public release focuses on:

- interactive browser animations;
- paper-grounded theory notes;
- reproducible PCA/t-SNE/UMAP experiments;
- generated visualizations and GIFs;
- hyperparameter sensitivity on handwritten digits;
- examples from toy geometry, nonlinear dynamics, and handwritten-digit embeddings.

The central question:

```text
How do we turn high-dimensional representations into 2D maps without fooling ourselves?
```

## Quick Start

Open the interactive animation lab:

```text
index.html
```

For the comparison dashboard, start a local server:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/dashboard.html
```

## What Is Included

### Interactive Tools

- `index.html`: animated PCA, SNE, t-SNE, and UMAP intuition lab.
- `dashboard.html`: side-by-side comparison dashboard for generated handwritten-digit embeddings.

### Guides

- `docs/theory_guide.md`: PCA, SNE, t-SNE, UMAP, TriMap, and PaCMAP theory.
- `docs/comparison_guide.md`: hyperparameter sensitivity, timing, metrics, and UMAP graph intuition.
- `latent_space_visualization_tutorial.md`: beginner-friendly tutorial.

### Generated Results

- `results_canonical/`: rings and Lorenz attractor PCA/t-SNE/UMAP results.
- `results_digits/`: handwritten-digits sweeps, GIFs, metrics, and browser-loadable embeddings.

### Reproducible Scripts

- `scripts/generate_canonical_results.py`
- `scripts/generate_digits_sensitivity.py`
- `scripts/generate_results.js`

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate canonical results:

```bash
python scripts/generate_canonical_results.py
```

Regenerate handwritten-digits sweeps:

```bash
python scripts/generate_digits_sensitivity.py
```

On Windows, if UMAP/Numba needs a local cache:

```powershell
$env:NUMBA_CACHE_DIR="$PWD\.numba_cache"
python scripts\generate_digits_sensitivity.py
```

## Learning Path

1. Open `index.html`.
2. Try PCA on Swiss roll.
3. Switch to SNE and t-SNE to see local neighborhoods form.
4. Switch to UMAP and inspect graph links.
5. Open `dashboard.html` and compare PCA, t-SNE, and UMAP side by side.
6. Read `docs/comparison_guide.md`.
7. Read `docs/theory_guide.md`.
8. Inspect `results_digits/summary.md` and `results_digits/metrics.csv`.

## Current Handwritten-Digits Results

| Method | Hyperparameter | Value | kNN overlap | Trustworthiness |
| --- | --- | ---: | ---: | ---: |
| PCA | n_components | 2 | 0.137 | 0.817 |
| t-SNE | perplexity | 5 | 0.468 | 0.974 |
| t-SNE | perplexity | 15 | 0.508 | 0.980 |
| t-SNE | perplexity | 30 | 0.517 | 0.982 |
| t-SNE | perplexity | 50 | 0.513 | 0.982 |
| UMAP | n_neighbors | 5 | 0.433 | 0.968 |
| UMAP | n_neighbors | 15 | 0.448 | 0.974 |
| UMAP | n_neighbors | 50 | 0.425 | 0.970 |
| UMAP | n_neighbors | 100 | 0.418 | 0.965 |
| UMAP | min_dist | 0.0 | 0.435 | 0.974 |
| UMAP | min_dist | 0.1 | 0.433 | 0.971 |
| UMAP | min_dist | 0.35 | 0.431 | 0.969 |
| UMAP | min_dist | 0.7 | 0.404 | 0.962 |

## Interpretation Rules

- Always ask what the method preserves.
- Always compare against PCA.
- Always sweep important hyperparameters.
- Treat t-SNE/UMAP axes as arbitrary.
- Treat beautiful maps as hypothesis generators, not proof.

## Private Research Extensions

Some heavier research modules are intentionally not included in this public release. The public repo is designed to be useful, reproducible, and educational while leaving room for private experimental extensions.

## References

- Pearson, K. "On Lines and Planes of Closest Fit to Systems of Points in Space." 1901.
- Hotelling, H. "Analysis of a Complex of Statistical Variables into Principal Components." 1933.
- Hinton, G. E. and Roweis, S. T. "Stochastic Neighbor Embedding." 2002.
- van der Maaten, L. and Hinton, G. "Visualizing Data using t-SNE." JMLR, 2008.
- McInnes, L., Healy, J., and Melville, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." 2018.
- Amid, E. and Warmuth, M. K. "TriMap: Large-scale Dimensionality Reduction Using Triplets." 2019.
- Wang, Y., Huang, H., Rudin, C., and Shaposhnik, Y. "Understanding How Dimension Reduction Tools Work..." 2021.
