# Comparison Guide: Hyperparameters, Cost, and Sensitivity

This guide explains the generated handwritten-digits sensitivity results.

The script uses `sklearn.datasets.load_digits`, a small MNIST-like handwritten digit dataset bundled with scikit-learn. It is not the full 70,000-image MNIST dataset, but it is ideal for a reproducible teaching repo because it requires no download.

Generate the results with:

```bash
python scripts/generate_digits_sensitivity.py
```

On Windows, if UMAP/Numba needs an explicit cache directory:

```powershell
$env:NUMBA_CACHE_DIR="$PWD\.numba_cache"
python scripts\generate_digits_sensitivity.py
```

## 1. Generated Artifacts

The script writes to `results_digits/`.

Main comparison figures:

- `digits_pca.png`
- `digits_tsne_perplexity_grid.png`
- `digits_umap_neighbors_grid.png`
- `digits_umap_min_dist_grid.png`
- `digits_umap_graph_construction.png`

Animations:

- `digits_tsne_perplexity_sweep.gif`
- `digits_umap_neighbors_sweep.gif`
- `digits_umap_min_dist_sweep.gif`

Metrics:

- `summary.md`
- `metrics.csv`

## 2. Metrics Used

### kNN Overlap

For each point, compute its `k` nearest neighbors in the original high-dimensional space and in the 2D map.

```text
kNN overlap =
average fraction of original neighbors that remain neighbors in 2D
```

This is easy to explain and useful for local-neighborhood preservation.

### Trustworthiness

Trustworthiness penalizes points that become neighbors in the 2D map even though they were not close in high-dimensional space.

High trustworthiness means:

```text
If points look close in the map, they are usually close in the original space.
```

It does not prove global geometry is preserved.

## 3. Current Digits Results

| Method | Hyperparameter | Value | Seconds | kNN overlap | Trustworthiness |
| --- | --- | ---: | ---: | ---: | ---: |
| PCA | n_components | 2 | 0.02 | 0.137 | 0.817 |
| t-SNE | perplexity | 5 | 6.69 | 0.468 | 0.974 |
| t-SNE | perplexity | 15 | 6.86 | 0.508 | 0.980 |
| t-SNE | perplexity | 30 | 9.01 | 0.517 | 0.982 |
| t-SNE | perplexity | 50 | 9.66 | 0.513 | 0.982 |
| UMAP | n_neighbors | 5 | 15.49 | 0.433 | 0.968 |
| UMAP | n_neighbors | 15 | 6.19 | 0.448 | 0.974 |
| UMAP | n_neighbors | 50 | 7.19 | 0.425 | 0.970 |
| UMAP | n_neighbors | 100 | 7.92 | 0.418 | 0.965 |
| UMAP | min_dist | 0.0 | 6.77 | 0.435 | 0.974 |
| UMAP | min_dist | 0.1 | 6.48 | 0.433 | 0.971 |
| UMAP | min_dist | 0.35 | 6.69 | 0.431 | 0.969 |
| UMAP | min_dist | 0.7 | 6.64 | 0.404 | 0.962 |

These timings are machine-dependent. The relative patterns are the important part.

## 4. What Changes When t-SNE Perplexity Changes?

Perplexity controls effective neighborhood size.

Small perplexity:

- focuses on very local neighborhoods;
- can split classes into small islands;
- may reveal subclusters such as different handwriting styles;
- can over-fragment the map.

Larger perplexity:

- uses broader neighborhoods;
- usually makes clusters more stable;
- can smooth over small substructures;
- increases compute because each point considers a broader neighborhood.

In the generated digits run, t-SNE improves from perplexity `5` to around `30`, then mostly plateaus.

Teaching line:

```text
Perplexity is a neighborhood lens. Too small is myopic; too large can blur local detail.
```

## 5. What Changes When UMAP n_neighbors Changes?

UMAP starts by constructing a k-nearest-neighbor graph.

`n_neighbors` is the graph scale.

Small `n_neighbors`:

- local graph;
- sharper clusters;
- more fragmented global structure;
- strong focus on fine details.

Large `n_neighbors`:

- broader graph;
- more global continuity;
- less fragmented layout;
- local clusters may become less crisp.

In the generated digits run, `n_neighbors=15` gives the best local metric among tested values. Larger values are still useful when the goal is to see broader continuity rather than pure local preservation.

Teaching line:

```text
n_neighbors asks: how large is the neighborhood from which each point understands the world?
```

## 6. What Changes When UMAP min_dist Changes?

`min_dist` controls how tightly points are allowed to pack in the low-dimensional map.

Low `min_dist`:

- compact clusters;
- visually crisp class islands;
- stronger apparent separation;
- can exaggerate cluster discreteness.

High `min_dist`:

- points spread out;
- clusters look less dense;
- continuous transitions become easier to inspect;
- class boundaries can look softer.

Teaching line:

```text
min_dist is not the neighborhood definition. It is the packing rule for the final map.
```

## 7. UMAP Graph Construction, Step by Step

UMAP is easier when treated as a graph algorithm first and an embedding algorithm second.

### Step 1: Find k-nearest neighbors

For each point `x_i`, find `k` nearby points.

This creates a directed graph:

```text
i -> nearest neighbors of i
```

### Step 2: Estimate local radius

UMAP defines a local connectivity radius:

```text
rho_i = distance to the nearest neighbor of x_i
```

This says every point gets at least one strong connection.

### Step 3: Estimate local scale

UMAP chooses `sigma_i` so each point has a comparable effective neighborhood size despite changing data density.

Dense regions get smaller scales. Sparse regions get larger scales.

### Step 4: Convert distances to fuzzy memberships

For neighbor `j` of `i`:

```text
w_{i|j} = exp(-(d(x_i, x_j) - rho_i) / sigma_i)
```

This is a directed membership strength.

### Step 5: Symmetrize with fuzzy union

Directed edges are combined:

```text
w_ij = w_{i|j} + w_{j|i} - w_{i|j}w_{j|i}
```

This is why UMAP is often described as building a fuzzy topological graph.

### Step 6: Optimize a 2D graph

UMAP creates low-dimensional memberships:

```text
v_ij = 1 / (1 + a ||y_i - y_j||^{2b})
```

Then it minimizes cross-entropy between `w_ij` and `v_ij`.

High-weight graph edges pull together. Negative samples push unrelated points apart.

## 8. Compute Cost and Practical Tradeoffs

| Method | Rough Cost | Strength | Weakness |
| --- | --- | --- | --- |
| PCA | very low; SVD/eigendecomposition | fast, deterministic, interpretable | linear only |
| t-SNE | moderate to high; neighbor search plus iterative optimization | excellent local clusters | sensitive, slower, no natural transform in classic form |
| UMAP | moderate; approximate neighbor graph plus SGD | fast, scalable, can transform new points | graph construction and parameters are less intuitive |

Practical notes:

- PCA is the baseline. Always run it first.
- t-SNE is often excellent for cluster discovery, but parameter sweeps are mandatory.
- UMAP is a strong default for larger datasets and workflows where new points may need to be embedded later.
- Timing depends heavily on implementation, dataset size, nearest-neighbor backend, CPU, and random seed settings.

## 9. Suggested Teaching Sequence

1. Show PCA on digits.
2. Ask students which digits overlap.
3. Show t-SNE perplexity sweep.
4. Discuss local neighborhoods and cluster fragmentation.
5. Show UMAP graph construction.
6. Show UMAP `n_neighbors` sweep.
7. Show UMAP `min_dist` sweep.
8. Compare metrics with plots.
9. End with the warning: good visualization is a hypothesis generator, not a proof.

## 10. What To Add Next

Good next modules:

- Full MNIST via `fetch_openml("mnist_784")` with a cached subset.
- Fashion-MNIST embeddings from a small autoencoder.
- Lorenz trajectory delay embeddings.
- TriMap triplet animation.
- PaCMAP near, mid-near, far pair animation.
