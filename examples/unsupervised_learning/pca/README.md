# Principal Component Analysis (PCA)

Modern datasets are frequently **high-dimensional**: images have thousands of pixels, gene-expression profiles span tens of thousands of transcripts, and customer logs track dozens of behaviors per user. High dimensionality makes data hard to visualize, slow to model, and prone to overfitting. Dimensionality reduction is the unsupervised task of compressing data into a smaller number of informative features while retaining as much of its structure as possible.

Principal Component Analysis (PCA) is the classical linear technique for this job. Rather than picking a subset of the original features, PCA searches for **new axes — called principal components — that are linear combinations of the original features and are oriented along the directions of greatest variance**. Projecting onto the first few components keeps most of the "spread" of the data while discarding the rest.

At a high level, PCA works by:

1. **Centering** the data by subtracting the mean of each feature.
2. Computing the **covariance matrix** of the centered data.
3. **Eigen-decomposing** the covariance matrix into eigenvalues (how much variance each axis carries) and eigenvectors (the axes themselves).
4. Sorting the eigenvectors by their eigenvalues in descending order and keeping the top `k` as the **principal axes**.
5. **Projecting** the centered data onto those `k` axes to obtain a compressed representation.

Because the principal components are orthogonal and ordered by variance, PCA gives both a **compression** (the top-k projection) and a **story** (the relative importance of each direction, summarized by the variance ratio). The same decomposition is also closely related to the Singular Value Decomposition, which is why PCA can be computed either from the covariance matrix or directly from the data matrix via SVD.

PCA is used everywhere in practice: to visualize clusters in 2D or 3D, to denoise data by keeping only the top components, to speed up downstream models, and to build eigen-faces, eigen-digits, and similar latent representations.

In `pca_from_scratch.ipynb`, we:

- Derive the covariance-matrix formulation of PCA and explain what the eigenvalues and variance ratios mean.
- Implement PCA from scratch using `numpy.linalg.eigh` and project a real dataset (such as the iris or wine dataset) into 2D for visualization.
- Plot the **explained-variance curve** ("scree plot") and discuss how to choose the number of components.
- Show how PCA preserves reconstructions: project, inverse-project, and measure the reconstruction error.
- Compare our implementation against scikit-learn's `PCA` to confirm axes and variance ratios agree (up to sign flips).
