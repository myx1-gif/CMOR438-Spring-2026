# DBSCAN Clustering

Clustering is a form of unsupervised learning used to uncover natural groupings within data. Unlike supervised learning, where models learn from labeled outcomes, clustering seeks to find structure in data without predefined categories. Real-world applications include identifying regions of anomalous behavior in credit-card transactions, grouping geo-tagged photos by location, and finding dense regions of activity in sensor networks.

DBSCAN — **D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise — takes a fundamentally different view than centroid-based methods like K-means. Instead of asking *"which center is each point closest to?"*, it asks *"where is the data dense enough to form a cluster, and what sits out in the sparse empty space as noise?"* At a high level, DBSCAN works by:

1. Choosing a neighbourhood radius `ε` (epsilon) and a minimum number of neighbours `min_samples`.
2. Labeling a point as a **core point** if at least `min_samples` points lie within distance `ε` of it.
3. Connecting core points that are within `ε` of one another into the same cluster, and absorbing non-core points that fall inside a core point's neighbourhood as **border points**.
4. Marking any point that is neither a core nor a border point as **noise** (given the special label `-1`).

This density-based view gives DBSCAN three properties that K-means lacks:

- It discovers clusters of **arbitrary shape** (rings, crescents, filaments), not just spherical blobs.
- It does **not require you to specify the number of clusters**; the algorithm decides from the density of the data.
- It has a built-in notion of **noise / outliers**, so points in low-density regions are flagged rather than forced into a cluster.

The trade-off is that DBSCAN is sensitive to its two hyperparameters and to the scale of the features — you typically need to standardize the data and tune `ε` carefully.

In `dbscan_from_scratch.ipynb`, we:

- Walk through the core / border / noise terminology and the expansion procedure that grows each cluster.
- Visualize DBSCAN on synthetic datasets where K-means fails (e.g., concentric rings and non-convex shapes).
- Explore how `ε` and `min_samples` together control cluster granularity and the noise ratio.
- Compare our from-scratch implementation against scikit-learn's `DBSCAN` to confirm the two agree on the same labelings.
