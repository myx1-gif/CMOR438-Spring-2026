# K-means Clustering

Clustering is a form of unsupervised learning used to uncover natural groupings within data. Unlike supervised learning, where models learn from labeled outcomes, clustering seeks to find structure in data without predefined categories. Real-world applications include grouping social media posts for targeted advertising, organizing grocery items in a store layout, or segmenting customers by purchasing behavior.

K-means clustering is one of the most widely used and foundational algorithms in this space. At a high level, K-means works by:

1. Choosing a number of clusters (K).
2. Assigning each observation to the cluster with the nearest centroid (the "center" of a cluster).
3. Updating centroids based on the mean of the assigned points.
4. Repeating this process until the assignments stabilize.

This iterative refinement results in groupings that minimize the distance of points to their respective centroids. While simple and computationally efficient, K-means has important limitations: the number of clusters must be chosen in advance, it assumes clusters are roughly spherical and evenly sized, and results can depend on initial centroid placement.

In `k_means_from_scratch.ipynb`, we:

- Explain the mathematics and iterative update scheme of K-means
- Highlight practical use cases across industries
- Explore and cluster different wines, imagining a potential wine recommendation system
- Analyze how the choice of K impacts the algorithm's performance
- Compare results from our implementation against scikit-learn's built-in K-means
