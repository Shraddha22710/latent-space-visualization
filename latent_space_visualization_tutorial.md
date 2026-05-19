# Latent Space Visualization Tutorial: PCA, t-SNE, and UMAP

This tutorial is a project guide for understanding and implementing latent-space visualization with PCA, t-SNE, and UMAP.

## 1. The Big Idea

Modern machine learning often represents data as vectors with many dimensions.

Examples:

- A grayscale MNIST digit image has 28 x 28 = 784 pixel values.
- A CIFAR-10 image has 32 x 32 x 3 = 3072 pixel values.
- A neural network embedding may have 128, 512, 768, or thousands of dimensions.

Humans cannot directly look at 784-dimensional or 3072-dimensional space. To visualize it, we reduce the data to 2D or 3D while trying to preserve meaningful structure.

That process is called dimensionality reduction.

This guide focuses on three popular methods:

- PCA: Principal Component Analysis
- t-SNE: t-distributed Stochastic Neighbor Embedding
- UMAP: Uniform Manifold Approximation and Projection

These methods are especially useful for visualizing latent spaces, such as the compressed representation learned by an autoencoder.

## 2. What Is a Latent Space?

A latent space is a compressed representation of data.

For example, an autoencoder learns to:

1. Encode an input image into a smaller vector.
2. Store the important information in that vector.
3. Decode the vector back into an image.

The compressed vector is called a latent representation.

If an autoencoder is trained well, similar inputs should often land near each other in latent space.

Example:

- Images of the digit `0` should cluster near other `0`s.
- Images of dogs may be closer to cats than to airplanes.
- Similar faces, words, sounds, or documents may form neighborhoods.

The problem is that latent vectors are often too high-dimensional to inspect directly. PCA, t-SNE, and UMAP help project them into 2D so we can plot them.

## 3. Why Not Just Plot the First Two Dimensions?

Suppose every image is represented by 784 numbers. You might think:

"Can we just use dimension 1 as x and dimension 2 as y?"

Usually, no.

The first two raw coordinates may not contain the most useful variation. Important structure may be spread across many dimensions. Dimensionality reduction tries to create better 2D coordinates by using information from the full high-dimensional dataset.

## 4. PCA: Principal Component Analysis

### 4.1 Intuition

PCA finds directions where the data varies the most.

Imagine a long cloud of points in 2D. If you wanted to compress it to 1D, the best line would usually be the line running along the longest direction of the cloud. PCA finds that direction.

In higher dimensions, PCA finds new axes called principal components:

- First principal component: direction of maximum variance.
- Second principal component: next most informative direction, perpendicular to the first.
- Third principal component: next direction, and so on.

For visualization, we usually keep the first two components.

### 4.2 What PCA Preserves

PCA tries to preserve global variance.

This means it is good at showing large-scale directions in the data. It is fast, deterministic, and easy to interpret.

### 4.3 PCA Algorithm

Given a data matrix `X` with shape:

```text
number_of_samples x number_of_features
```

PCA roughly does this:

1. Center the data by subtracting the mean of each feature.
2. Find directions of maximum variance.
3. Project the data onto the top components.

Mathematically, PCA can be computed using eigendecomposition of the covariance matrix or using singular value decomposition.

### 4.4 PCA Strengths

- Very fast compared with t-SNE and UMAP.
- Works well as a first baseline.
- Useful for noise reduction.
- Components are mathematically interpretable.
- Can transform new data after fitting.

### 4.5 PCA Weaknesses

- It is linear.
- It cannot unfold nonlinear manifolds.
- It may miss complex local structure.

If the data lies on a curved surface, PCA can only project it onto a flat linear plane. This can cause separate parts of the structure to overlap.

### 4.6 When to Use PCA

Use PCA when:

- You want a quick first visualization.
- You want interpretable components.
- You need a fast preprocessing step.
- Your data is approximately linear.
- You want to reduce dimensions before t-SNE or UMAP.

## 5. t-SNE

### 5.1 Intuition

t-SNE is designed for visualization.

Its main goal is:

Keep nearby points in high-dimensional space nearby in low-dimensional space.

Unlike PCA, t-SNE is nonlinear. It can reveal clusters that PCA may flatten or overlap.

### 5.2 Neighborhoods Instead of Axes

PCA asks:

"Which directions explain the most variance?"

t-SNE asks:

"Which points are neighbors of which other points?"

This is a very different question. t-SNE is less concerned with preserving exact global distances and more focused on preserving local neighborhoods.

### 5.3 High-Dimensional Similarities

t-SNE first converts distances in high-dimensional space into probabilities.

For each point, nearby points get high probability and faraway points get low probability.

The idea:

- If two points are close in high-dimensional space, the probability of choosing one as the neighbor of the other should be high.
- If two points are far apart, the probability should be low.

The neighborhood size is controlled by a hyperparameter called perplexity.

### 5.4 Low-Dimensional Similarities

t-SNE then creates random 2D coordinates and computes similarities between points in this low-dimensional map.

At first, the low-dimensional similarities are poor because the points are randomly placed.

t-SNE then moves the 2D points so that low-dimensional similarities become as close as possible to high-dimensional similarities.

### 5.5 Loss Function

t-SNE compares the high-dimensional probability distribution `P` and the low-dimensional probability distribution `Q`.

It minimizes KL divergence:

```text
KL(P || Q) = sum P(i, j) * log(P(i, j) / Q(i, j))
```

This penalizes cases where points that were neighbors in high-dimensional space are placed far apart in the low-dimensional map.

### 5.6 Why the "t" in t-SNE?

The low-dimensional map uses a Student t-distribution rather than a Gaussian distribution.

This gives the low-dimensional space heavier tails, which helps reduce crowding. Without this, too many moderately distant points would be squeezed into the center of the plot.

### 5.7 t-SNE Strengths

- Excellent for showing local clusters.
- Good for nonlinear data.
- Often produces visually clear separated groups.
- Useful for exploratory visualization.

### 5.8 t-SNE Weaknesses

- Can be slow on large datasets.
- Sensitive to hyperparameters.
- Different random seeds can produce different layouts.
- Distances between faraway clusters are often not meaningful.
- Cluster sizes in the plot may not reflect true cluster sizes.
- New points are not naturally embedded without rerunning or using special variants.

### 5.9 Important t-SNE Hyperparameters

`perplexity`

Controls the effective neighborhood size.

- Lower perplexity focuses on very local structure.
- Higher perplexity considers broader neighborhoods.
- Common values: 5 to 50.

`learning_rate`

Controls how large the optimization steps are.

`n_iter` or `max_iter`

Controls how long the optimization runs.

`init`

Many workflows initialize t-SNE using PCA for more stable results.

### 5.10 When to Use t-SNE

Use t-SNE when:

- You care about local cluster structure.
- You want a visual exploration tool.
- You are okay with slower runtime.
- You do not need exact global distances.

Avoid overinterpreting:

- The absolute distance between separated clusters.
- The size of clusters.
- The exact orientation of the plot.

## 6. UMAP

### 6.1 Intuition

UMAP is another nonlinear dimensionality reduction method.

Like t-SNE, it tries to keep nearby high-dimensional points nearby in low-dimensional space. But it uses a graph-based approach.

UMAP is often faster than t-SNE and can preserve more global structure.

### 6.2 Graph View of Data

UMAP builds a graph where:

- Each data point is a node.
- Edges connect nearby points.
- Edge weights represent how strongly two points are connected.

The main idea:

1. Build a weighted neighbor graph in high-dimensional space.
2. Build a weighted neighbor graph in low-dimensional space.
3. Move the low-dimensional points until the low-dimensional graph resembles the high-dimensional graph.

### 6.3 Step 1: Find Nearest Neighbors

For each point, UMAP finds its `k` nearest neighbors.

This `k` is controlled by:

```text
n_neighbors
```

Small `n_neighbors` values emphasize local detail.

Large `n_neighbors` values preserve broader global structure.

### 6.4 Step 2: Create a Weighted Graph

UMAP assigns stronger weights to closer neighbors and weaker weights to farther neighbors.

Each point gets its own local notion of distance. This matters because data density may vary across the dataset.

In dense regions, nearby points may be very close. In sparse regions, nearby points may be farther apart. UMAP adapts to this.

### 6.5 Step 3: Optimize the Low-Dimensional Map

UMAP initializes points in 2D and then adjusts them so the low-dimensional graph matches the high-dimensional graph.

It uses a cross-entropy style objective:

- Connected points are pulled together.
- Unconnected or weakly connected points are pushed apart.

### 6.6 UMAP Strengths

- Usually faster than t-SNE.
- Often preserves local and some global structure.
- Works well on large datasets.
- Can transform new points after fitting.
- Has useful hyperparameters for controlling the map.

### 6.7 UMAP Weaknesses

- Sensitive to hyperparameters.
- Can create misleading structure if overinterpreted.
- Requires choices about distance metric and neighborhood size.
- Theoretical details are more complex than PCA.

### 6.8 Important UMAP Hyperparameters

`n_neighbors`

Controls local versus global structure.

- Small values: more local detail, more separated clusters.
- Large values: more global continuity, less fragmented layout.

Common values:

```text
5, 10, 15, 30, 50, 100
```

`min_dist`

Controls how tightly points can pack together.

- Low `min_dist`: tighter clusters.
- High `min_dist`: more spread-out clusters.

Common values:

```text
0.0, 0.1, 0.25, 0.5
```

`metric`

Defines how distance is measured in high-dimensional space.

Common choices:

- `euclidean`
- `cosine`
- `manhattan`
- `correlation`

For neural embeddings, `cosine` is often worth trying.

### 6.9 When to Use UMAP

Use UMAP when:

- You want nonlinear visualization.
- You have a larger dataset.
- You want better runtime than t-SNE.
- You care about both local and broad structure.
- You may need to transform new points later.

## 7. PCA vs t-SNE vs UMAP

| Method | Type | Preserves | Speed | Best Use |
| --- | --- | --- | --- | --- |
| PCA | Linear | Global variance | Very fast | Baseline, preprocessing, interpretable compression |
| t-SNE | Nonlinear | Local neighborhoods | Slow to moderate | Cluster visualization |
| UMAP | Nonlinear | Local and some global structure | Fast to moderate | Scalable latent-space visualization |

## 8. Important Visualization Warnings

Dimensionality reduction plots are useful, but they can mislead.

### 8.1 Do Not Overinterpret Axes

In PCA, axes have meaning: PC1 and PC2.

In t-SNE and UMAP, the x-axis and y-axis usually do not have direct semantic meaning. Rotating or flipping the plot does not change the result.

### 8.2 Do Not Overinterpret Global Distances

In t-SNE especially, faraway cluster distances are not always meaningful.

If cluster A appears twice as far from cluster B as from cluster C, that does not necessarily mean A is semantically twice as different from B.

### 8.3 Do Not Treat Clusters as Guaranteed Truth

These methods can make visual clusters even when the underlying structure is less clean.

Always validate with labels, metrics, domain knowledge, or downstream tests.

### 8.4 Try Multiple Seeds and Parameters

If a conclusion only appears for one random seed or one parameter setting, be cautious.

Good practice:

- Run multiple random seeds.
- Try several perplexity or neighbor values.
- Compare PCA, t-SNE, and UMAP.
- Check whether patterns are stable.

## 9. Practical Python Tutorial

### 9.1 Install Packages

```bash
pip install numpy pandas matplotlib scikit-learn umap-learn
```

### 9.2 Load MNIST Digits

This example uses the small digits dataset from scikit-learn.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
y = digits.target

X_scaled = StandardScaler().fit_transform(X)

print(X.shape)
```

### 9.3 Plot Helper

```python
def plot_embedding(embedding, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.8
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()
```

### 9.4 PCA Example

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plot_embedding(X_pca, y, "PCA projection of digits")

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

What to look for:

- Are digit classes separated?
- Which digits overlap?
- How much variance do the first two components explain?

### 9.5 t-SNE Example

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plot_embedding(X_tsne, y, "t-SNE projection of digits")
```

What to look for:

- t-SNE will often create clearer local clusters than PCA.
- Some digit classes may split into subclusters because there are multiple writing styles.

### 9.6 UMAP Example

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

plot_embedding(X_umap, y, "UMAP projection of digits")
```

What to look for:

- UMAP often produces clear clusters quickly.
- Try changing `n_neighbors` and `min_dist`.

## 10. Parameter Experiments

### 10.1 t-SNE Perplexity Experiment

```python
perplexities = [5, 15, 30, 50]

for perplexity in perplexities:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    embedding = tsne.fit_transform(X_scaled)
    plot_embedding(embedding, y, f"t-SNE perplexity={perplexity}")
```

Expected behavior:

- Low perplexity may create many small islands.
- Higher perplexity may merge local neighborhoods into broader groups.

### 10.2 UMAP n_neighbors Experiment

```python
neighbors_list = [5, 15, 50, 100]

for n_neighbors in neighbors_list:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)
    plot_embedding(embedding, y, f"UMAP n_neighbors={n_neighbors}")
```

Expected behavior:

- Low `n_neighbors` emphasizes local clusters.
- High `n_neighbors` emphasizes broader continuity.

### 10.3 UMAP min_dist Experiment

```python
min_dist_values = [0.0, 0.1, 0.5, 0.9]

for min_dist in min_dist_values:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)
    plot_embedding(embedding, y, f"UMAP min_dist={min_dist}")
```

Expected behavior:

- Low `min_dist` makes clusters tighter.
- High `min_dist` spreads points out.

## 11. Visualizing an Autoencoder Latent Space

These methods connect naturally to latent-space visualization. Here is the workflow.

### 11.1 Train or Load an Autoencoder

An autoencoder has two parts:

```text
input -> encoder -> latent vector -> decoder -> reconstruction
```

After training, we use only the encoder to produce latent vectors.

### 11.2 Extract Latent Vectors

Pseudo-code:

```python
latent_vectors = encoder.predict(images)
```

If the latent vectors already have 2 dimensions, you can plot them directly.

If the latent vectors have many dimensions, use PCA, t-SNE, or UMAP:

```python
latent_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
).fit_transform(latent_vectors)
```

### 11.3 Plot Latent Space

```python
plot_embedding(latent_2d, labels, "Autoencoder latent space visualized with UMAP")
```

What to look for:

- Do samples from the same class cluster together?
- Are similar classes close?
- Are there outliers?
- Are there smooth transitions between classes?

## 12. Pixel Space vs Latent Space

A key practical idea is that dimensionality reduction on raw pixels can fail as data becomes more complex.

For simple datasets like MNIST, raw pixel vectors may still contain enough structure for PCA, t-SNE, and UMAP to reveal meaningful clusters.

For more complex datasets like CIFAR-10, raw pixel distances are often less meaningful. Two images can be semantically similar but have very different pixels due to pose, lighting, background, or color.

In that case, it is often better to:

1. Use a neural network to create embeddings.
2. Apply PCA, t-SNE, or UMAP to those embeddings.

The neural network learns more useful features than raw pixels.

## 13. Recommended Workflow

For a real project:

1. Start with PCA.
2. Try UMAP.
3. Try t-SNE if local clusters are especially important.
4. Tune parameters.
5. Run multiple seeds.
6. Compare results with labels or domain knowledge.
7. Avoid making claims from one plot alone.

## 14. Common Mistakes

### Mistake 1: Treating t-SNE/UMAP axes as meaningful

The axes usually do not directly correspond to human-readable features.

### Mistake 2: Assuming cluster distance equals semantic distance

Especially in t-SNE, faraway distances can be visually convenient rather than truly meaningful.

### Mistake 3: Using raw pixels for complex image datasets

Raw pixel distance is often a poor proxy for semantic similarity.

### Mistake 4: Trusting one parameter setting

Always test whether the pattern is stable.

### Mistake 5: Forgetting scaling

Many methods work better when features are standardized.

## 15. Quick Decision Guide

Use PCA if:

- You need speed.
- You need interpretability.
- You want a baseline.
- You want preprocessing before nonlinear methods.

Use t-SNE if:

- You want strong local cluster visualization.
- Dataset size is manageable.
- You mainly care about neighborhoods, not global geometry.

Use UMAP if:

- You want a strong default visualization method.
- You need better speed than t-SNE.
- You want a balance of local and global structure.
- You want to transform new points later.

## 16. Mini Glossary

Dimensionality reduction:

Reducing the number of features while preserving useful structure.

Latent space:

A learned compressed representation of data.

Embedding:

A vector representation of an object such as an image, word, document, or sound.

Manifold:

A lower-dimensional structure embedded inside a higher-dimensional space.

Perplexity:

A t-SNE parameter that controls effective neighborhood size.

Nearest-neighbor graph:

A graph connecting each point to its closest points.

KL divergence:

A measure of how different one probability distribution is from another.

Cross entropy:

A loss used to compare target relationships with predicted relationships.

## 17. Summary

PCA, t-SNE, and UMAP all help us see high-dimensional data in 2D.

PCA is fast and linear. It shows major variance directions.

t-SNE is nonlinear and excellent for local clusters, but it can be slow and easy to overinterpret.

UMAP is nonlinear, fast, graph-based, and often a strong practical choice for visualizing latent spaces.

For latent-space visualization, the most important lesson is this:

The quality of the representation matters as much as the visualization algorithm.

If raw pixels do not produce meaningful clusters, use learned embeddings from an autoencoder or another neural network, then visualize those embeddings with PCA, t-SNE, or UMAP.

## 18. Sources and Further Reading

- UMAP paper: https://arxiv.org/abs/1802.03426
- t-SNE paper: https://www.jmlr.org/papers/v9/vandermaaten08a.html
- scikit-learn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- UMAP documentation: https://umap-learn.readthedocs.io/
