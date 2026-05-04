# Singular Value Decomposition (SVD)

The Singular Value Decomposition is arguably the most important matrix factorization in applied mathematics and machine learning. It generalizes the eigen-decomposition to arbitrary (even non-square, even rank-deficient) matrices and sits at the heart of dimensionality reduction, data compression, recommender systems, latent semantic analysis, low-rank approximation, and numerical stability of many linear-algebra routines.

For any real `m × n` matrix `A`, the SVD writes

    A = U Σ Vᵀ,

where

- `U` is an `m × m` orthogonal matrix whose columns are the **left singular vectors**,
- `V` is an `n × n` orthogonal matrix whose columns are the **right singular vectors**, and
- `Σ` is an `m × n` diagonal matrix of non-negative **singular values** `σ₁ ≥ σ₂ ≥ … ≥ 0`.

Intuitively, the SVD says that *any linear map can be understood as a rotation, followed by a non-negative rescaling along orthogonal axes, followed by another rotation.* The singular values tell you how much the transformation stretches along each axis; the singular vectors tell you which axes.

The property that makes the SVD so useful in machine learning is the **Eckart–Young theorem**: truncating the decomposition to the top `k` singular values gives the best rank-`k` approximation of `A` in both the Frobenius and spectral norms. That single fact underlies:

- **PCA**: the right singular vectors of the centered data matrix are exactly the principal components, and the squared singular values are proportional to the explained variance.
- **Low-rank compression** of images, matrices, and tensors — you keep only the top `k` singular triplets and discard the rest.
- **Latent Semantic Analysis** (LSA) in natural language processing, where SVD of a term-document matrix reveals topics.
- **Collaborative filtering** in recommender systems, where SVD factorizes a sparse user-item rating matrix into latent user and item embeddings.
- **Numerical linear algebra**: the SVD gives the pseudoinverse, the condition number, the numerical rank, and the most stable way to solve least-squares problems.

SVD does not require the matrix to be square or symmetric, handles rank-deficient matrices gracefully, and is numerically stable — which is why production libraries implement PCA by calling SVD rather than forming the covariance matrix explicitly.

In `svd_from_scratch.ipynb`, we:

- Derive the relationship between SVD, eigen-decomposition of `AᵀA`, and PCA.
- Compute the SVD of small illustrative matrices by hand and with `numpy.linalg.svd`.
- Demonstrate low-rank **image compression**: truncate an image matrix to the top `k` singular values and plot the reconstruction quality as `k` grows.
- Show the connection between the singular values and the explained-variance ratio from PCA.
- Build a **toy collaborative-filtering recommender** by factorizing a user-item rating matrix via SVD.
