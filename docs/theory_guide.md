# Deep Theory Guide: Latent Space Visualization

This guide explains the mathematical objectives behind PCA, SNE, t-SNE, UMAP, TriMap, and PaCMAP.

The app is intentionally visual and playful. This file is the researcher-facing companion: it explains what each method optimizes, why the objective has that shape, and what kinds of structure the method can and cannot justify.

## 1. Shared Problem

We start with high-dimensional data:

```text
X = {x_1, ..., x_n}, x_i in R^D
```

We want a low-dimensional map:

```text
Y = {y_1, ..., y_n}, y_i in R^d, usually d = 2 or 3
```

The impossible wish is:

```text
all meaningful relationships in X are preserved in Y
```

The realistic version is:

```text
choose one notion of relationship, then preserve that as well as possible
```

Different algorithms choose different relationships:

| Method | Preserved Structure | Main Mathematical Object |
| --- | --- | --- |
| PCA | Global linear variance | covariance eigenvectors |
| SNE | Local neighbor identity | conditional probability distributions |
| t-SNE | Local neighborhoods with less crowding | symmetric probabilities and Student-t kernel |
| UMAP | Fuzzy topological neighbor graph | fuzzy simplicial set cross-entropy |
| TriMap | Relative triplet comparisons | triplet ranking loss |
| PaCMAP | Local, mid-near, and far pair relations | staged pairwise loss |

## 2. PCA

### 2.1 Objective

PCA finds a linear projection that maximizes variance in the projected data.

For a unit vector `v`, the variance of the projection onto `v` is:

```text
Var(X v) = v^T C v
```

where:

```text
X_c = X - mean(X)
C = (1 / (n - 1)) X_c^T X_c
```

The first principal component solves:

```text
maximize_v v^T C v
subject to ||v|| = 1
```

Using a Lagrange multiplier:

```text
L(v, lambda) = v^T C v - lambda(v^T v - 1)
```

Differentiate with respect to `v`:

```text
dL/dv = 2 C v - 2 lambda v = 0
```

Therefore:

```text
C v = lambda v
```

So PCA directions are eigenvectors of the covariance matrix.

### 2.2 Projection

Sort eigenvectors by eigenvalue:

```text
lambda_1 >= lambda_2 >= ... >= lambda_D
```

For 2D visualization:

```text
Y = X_c [v_1, v_2]
```

### 2.3 Intuition

PCA is like turning a flashlight around the data cloud until the shadow is as spread out as possible.

That is beautiful, fast, and honest. But it is still a flat shadow. If the data is a spiral, roll, or attractor, PCA can only flatten it. It cannot peel it open.

### 2.4 What PCA Justifies

PCA justifies statements like:

- "Most global variance lies along this direction."
- "This component explains this fraction of total variance."
- "The data has a strong linear organization."

PCA does not justify:

- "These local neighbors are preserved."
- "This curved manifold has been unfolded."
- "Clusters in 2D are necessarily real semantic clusters."

## 3. Stochastic Neighbor Embedding

Primary source: Hinton and Roweis, "Stochastic Neighbor Embedding", NeurIPS 2002.

### 3.1 Core Move

SNE stops asking for a linear projection. It asks a probabilistic neighbor question:

```text
If x_i chooses one neighbor, how likely is it to choose x_j?
```

For high-dimensional data, define:

```text
p_{j|i} =
  exp(-||x_i - x_j||^2 / (2 sigma_i^2))
  /
  sum_{k != i} exp(-||x_i - x_k||^2 / (2 sigma_i^2))
```

with:

```text
p_{i|i} = 0
```

Nearby points get high probability. Far points get probability near zero.

### 3.2 Low-Dimensional Probabilities

In the map:

```text
q_{j|i} =
  exp(-||y_i - y_j||^2)
  /
  sum_{k != i} exp(-||y_i - y_k||^2)
```

Again:

```text
q_{i|i} = 0
```

### 3.3 Loss

SNE wants every conditional distribution `Q_i` to imitate `P_i`.

It minimizes:

```text
C = sum_i KL(P_i || Q_i)
  = sum_i sum_j p_{j|i} log(p_{j|i} / q_{j|i})
```

### 3.4 Why KL(P || Q)?

KL divergence is asymmetric. With `KL(P || Q)`, there is a large penalty when:

```text
p_{j|i} is large but q_{j|i} is small
```

In plain language:

```text
If two points are real neighbors in high dimensions, do not place them far apart in 2D.
```

But if:

```text
p_{j|i} is small but q_{j|i} is large
```

the penalty is weaker.

This means SNE mainly protects true high-dimensional neighbors.

### 3.5 Gradient Intuition

The SNE gradient can be understood as a force system:

```text
force roughly proportional to (p_{j|i} - q_{j|i})
```

- If `p > q`, the points are too far apart in the map, so pull them together.
- If `p < q`, the points are too close in the map, so push them apart.

### 3.6 Perplexity

SNE/t-SNE usually choose each `sigma_i` by targeting a fixed entropy:

```text
H(P_i) = - sum_j p_{j|i} log_2 p_{j|i}
```

Perplexity is:

```text
Perp(P_i) = 2^{H(P_i)}
```

Intuition:

```text
perplexity ~= effective number of neighbors
```

Small perplexity makes the method very local. Large perplexity asks it to consider broader neighborhoods.

## 4. t-SNE

Primary source: van der Maaten and Hinton, "Visualizing Data using t-SNE", JMLR 2008.

### 4.1 What t-SNE Fixes

SNE has two major difficulties:

1. It is hard to optimize.
2. It has a crowding problem.

The crowding problem comes from trying to represent high-dimensional neighborhoods in a tiny 2D area. There is not enough room near each point in 2D to place all moderately close high-dimensional neighbors at comfortable distances.

t-SNE changes two things:

1. It symmetrizes the high-dimensional probabilities.
2. It uses a heavy-tailed Student-t distribution in the low-dimensional map.

### 4.2 Symmetric High-Dimensional Similarities

First compute conditional probabilities `p_{j|i}` as in SNE.

Then define:

```text
p_{ij} = (p_{j|i} + p_{i|j}) / (2n)
```

This gives:

```text
sum_{i != j} p_{ij} = 1
```

and:

```text
p_{ij} = p_{ji}
```

### 4.3 Student-t Low-Dimensional Similarities

t-SNE defines:

```text
q_{ij} =
  (1 + ||y_i - y_j||^2)^(-1)
  /
  sum_{k != l} (1 + ||y_k - y_l||^2)^(-1)
```

This is a Student-t distribution with one degree of freedom.

The heavy tail means moderately distant 2D points still have non-negligible similarity. That gives repulsive forces enough reach to open up the map.

### 4.4 Loss

t-SNE minimizes:

```text
C = KL(P || Q)
  = sum_{i != j} p_{ij} log(p_{ij} / q_{ij})
```

### 4.5 Gradient

The famous t-SNE gradient is:

```text
dC/dy_i =
  4 sum_j (p_{ij} - q_{ij})
    (y_i - y_j)
    (1 + ||y_i - y_j||^2)^(-1)
```

Read it as a force law:

- `p_{ij}` creates attraction for real high-dimensional neighbors.
- `q_{ij}` creates repulsion in the low-dimensional map.
- the Student-t factor changes how force decays with distance.

### 4.6 Early Exaggeration

Many t-SNE implementations temporarily multiply `p_{ij}` by a constant early in training.

Intuition:

```text
First let real neighborhoods form strongly, then relax the map.
```

This often improves separation of clusters.

### 4.7 What t-SNE Justifies

t-SNE is strong evidence for:

- local neighborhoods,
- cluster-like local structure,
- subclusters and local modes,
- comparing stability across parameter sweeps.

t-SNE is weak evidence for:

- exact global distances,
- cluster area,
- axis direction,
- density without careful checks.

## 5. UMAP

Primary source: McInnes, Healy, and Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", arXiv 2018.

### 5.1 The Big Theoretical Story

UMAP assumes the data was sampled from a manifold. It then tries to approximate the manifold's local topology.

The paper frames this with:

- Riemannian geometry,
- local metrics,
- fuzzy simplicial sets,
- cross-entropy between fuzzy topological representations.

For implementation intuition, keep this translation:

```text
high-dimensional data -> weighted neighbor graph
2D embedding -> weighted neighbor graph
optimize so the 2D graph resembles the original graph
```

### 5.2 Local Connectivity

For each point `x_i`, find its `k` nearest neighbors.

Let:

```text
rho_i = distance from x_i to its nearest neighbor
```

This enforces local connectivity: each point should have at least one strong connection.

Then choose `sigma_i` so the local neighborhood has a controlled effective size. UMAP solves a smooth-k-neighbor equation of the form:

```text
sum_j exp(-(d(x_i, x_j) - rho_i) / sigma_i) ~= log_2(k)
```

over the selected neighbors.

### 5.3 Directed Edge Weights

For a neighbor `j` of `i`:

```text
w_{i|j} =
  exp(-(d(x_i, x_j) - rho_i) / sigma_i)
```

with clipping so distances inside the local radius are treated as maximally connected.

This creates a directed fuzzy graph. Why directed? Because local density differs. A distance can be "close" from a sparse point's perspective and "not very close" from a dense point's perspective.

### 5.4 Fuzzy Union

UMAP combines directed edges into undirected fuzzy memberships:

```text
w_{ij} = w_{i|j} + w_{j|i} - w_{i|j} w_{j|i}
```

This is the probabilistic union:

```text
P(A or B) = P(A) + P(B) - P(A)P(B)
```

### 5.5 Low-Dimensional Similarity

UMAP uses a differentiable curve in low dimensions, often written:

```text
v_{ij} = 1 / (1 + a ||y_i - y_j||^{2b})
```

The parameters `a` and `b` are fitted from user-facing controls such as `min_dist` and `spread`.

### 5.6 Cross-Entropy Objective

UMAP minimizes cross-entropy between high-dimensional fuzzy memberships `w_ij` and low-dimensional memberships `v_ij`:

```text
C =
sum_{i,j}
  w_ij log(w_ij / v_ij)
  +
  (1 - w_ij) log((1 - w_ij) / (1 - v_ij))
```

This has two terms:

- if `w_ij` is large, make `v_ij` large: attraction;
- if `w_ij` is small, make `v_ij` small: repulsion.

In practice, UMAP samples positive edges and negative examples rather than summing over all pairs.

### 5.7 What UMAP Justifies

UMAP is strong evidence for:

- local graph neighborhoods,
- connected manifold-like structure,
- local-to-mid-scale organization,
- stable patterns under `n_neighbors` sweeps.

UMAP is weak evidence for:

- exact densities,
- exact metric distances,
- isolated clusters as absolute truth.

### 5.8 Practical Computation

UMAP has two computational phases:

1. Build the high-dimensional neighbor graph.
2. Optimize the low-dimensional layout.

For large datasets, exact all-pairs distances are too expensive:

```text
all-pairs distance cost ~= O(n^2 D)
```

Practical UMAP implementations use approximate nearest-neighbor search. The reference Python ecosystem uses neighbor-descent style algorithms through `pynndescent`.

The optimization phase does not sum over every pair. Instead, it samples:

- positive edges from the fuzzy neighbor graph;
- negative examples for repulsion.

This is why UMAP scales better than a naive pairwise objective.

### 5.9 Parameter Sensitivity

`n_neighbors` controls the graph scale.

```text
small n_neighbors -> local detail, possible fragmentation
large n_neighbors -> broader continuity, less local crispness
```

`min_dist` controls final packing.

```text
small min_dist -> tight clusters
large min_dist -> spread-out continuous maps
```

`metric` controls what "near" means before the graph is built.

For neural embeddings, try:

- `cosine`
- `euclidean`
- `correlation`

The graph is only as meaningful as the metric used to construct it.

## 6. TriMap

Primary source: Amid and Warmuth, "TriMap: Large-scale Dimensionality Reduction Using Triplets", arXiv 2019.

### 6.1 Motivation

t-SNE and UMAP are mostly pairwise methods. TriMap uses triplets:

```text
(i, j, k)
```

meaning:

```text
x_i should be closer to x_j than to x_k
```

This relative comparison can preserve more global structure because it compares near and far relationships explicitly.

### 6.2 Triplet Loss

A typical triplet objective has the shape:

```text
loss(i, j, k) =
  omega_{ijk}
  log(1 + exp(s(y_i, y_k) - s(y_i, y_j)))
```

or an equivalent ranking-style loss.

Here:

- `j` is a neighbor of `i`,
- `k` is a farther point,
- `omega_{ijk}` weights the importance of the triplet,
- `s` is a similarity score in the embedding.

The important geometric rule is:

```text
distance(y_i, y_j) < distance(y_i, y_k)
```

### 6.3 Intuition

TriMap is like a judge comparing three objects:

```text
For anchor i, keep friend j closer than stranger k.
```

Instead of trying to match exact probabilities, it tries to preserve relative ordering information.

### 6.4 Why Researchers Care

Triplet comparisons can help with global structure because far points explicitly appear in the training signal.

TriMap is a strong candidate for a future module in this repo:

- animate triplets as "anchor, pull, push" arrows;
- let users sample triplets and see violated inequalities;
- show how global shape changes when far negatives are included.

## 7. PaCMAP

Primary source: Wang, Huang, Rudin, and Shaposhnik, "Understanding How Dimension Reduction Tools Work..." JMLR 2021.

### 7.1 Motivation

PaCMAP is built from an empirical study of what helps dimensionality reduction preserve both local and global structure.

Its central idea is to control three kinds of pairs:

1. near pairs,
2. mid-near pairs,
3. further pairs.

This is more explicit than t-SNE/UMAP, where the balance between local attraction and global repulsion is more implicit.

### 7.2 Pair Types

For an anchor point `i`:

- near pair: `j` is among nearest neighbors;
- mid-near pair: `j` is neither too close nor too far;
- further pair: `j` is far away.

### 7.3 Objective Shape

PaCMAP uses attractive terms for near and mid-near pairs and repulsive terms for further pairs.

A simplified conceptual version:

```text
C =
  w_N  sum_{(i,j) in near}     attraction(d_ij)
  + w_M sum_{(i,j) in midnear} attraction(d_ij)
  + w_F sum_{(i,j) in further} repulsion(d_ij)
```

where:

```text
d_ij = ||y_i - y_j||^2
```

The actual implementation uses specific rational functions and a staged weighting schedule.

### 7.4 Why Mid-Near Pairs Matter

Near pairs preserve local neighborhoods.

Further pairs prevent collapse.

Mid-near pairs are the interesting addition: they help the embedding preserve broader geometry without turning the problem into exact global distance preservation.

### 7.5 Future Visualization Module

PaCMAP would be fun to visualize because each pair type can be drawn differently:

- near pairs: strong green springs;
- mid-near pairs: amber structure-preserving tethers;
- further pairs: red repulsive guards.

The animation can show the schedule changing over optimization time.

## 8. Comparison of Objectives

| Method | Attraction | Repulsion | Global Signal |
| --- | --- | --- | --- |
| SNE | high `p_{j|i}` neighbors | implicit through `q` normalization | weak |
| t-SNE | high `p_ij` neighbors | Student-t normalization | moderate, often cluster-scale |
| UMAP | fuzzy graph edges | negative samples | moderate |
| TriMap | anchor-neighbor triplet relation | anchor-far triplet relation | strong via triplets |
| PaCMAP | near and mid-near pairs | further pairs | strong via pair design |

## 9. Interpretation Rules for Researchers

### Rule 1: Ask what the objective preserves

Do not ask:

```text
Is this plot true?
```

Ask:

```text
Which relationships did this objective try to preserve?
```

### Rule 2: Sweep parameters

Trust patterns that survive:

- multiple seeds,
- multiple perplexities or neighbor counts,
- multiple algorithms,
- meaningful baselines like PCA.

### Rule 3: Use metrics beside pictures

Useful checks:

- k-nearest-neighbor overlap,
- trustworthiness,
- continuity,
- class-neighborhood purity,
- downstream task performance.

### Rule 4: Separate visualization from proof

A beautiful map is a hypothesis generator. It is not a proof of clusters, density, causality, or physical state identity.

## 10. References

- Hinton, G. E. and Roweis, S. T. "Stochastic Neighbor Embedding." NeurIPS 2002. https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding
- van der Maaten, L. and Hinton, G. "Visualizing Data using t-SNE." JMLR 2008. https://www.jmlr.org/papers/v9/vandermaaten08a.html
- McInnes, L., Healy, J., and Melville, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv 2018. https://arxiv.org/abs/1802.03426
- Amid, E. and Warmuth, M. K. "TriMap: Large-scale Dimensionality Reduction Using Triplets." arXiv 2019. https://arxiv.org/abs/1910.00204
- Wang, Y., Huang, H., Rudin, C., and Shaposhnik, Y. "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization." JMLR 2021. https://www.jmlr.org/papers/v22/20-1061.html
