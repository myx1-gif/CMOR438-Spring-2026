# Label Propagation

Label propagation is a form of **semi-supervised learning** that bridges the gap between fully supervised and fully unsupervised settings. In many real-world problems — detecting fake accounts on a social network, categorizing scientific papers, or labeling protein functions — a tiny fraction of the data is labeled and the vast majority is not. Throwing away the unlabeled points would waste information; assigning labels by hand is expensive. Label propagation uses the **geometry of the data itself** to spread a handful of known labels to all the neighbouring unlabeled points.

At a high level, label propagation works by:

1. Constructing a **similarity graph** over all points (labeled and unlabeled), where edge weights measure how close two points are — most commonly through an RBF (Gaussian) kernel on Euclidean distance.
2. Row-normalizing that graph into a **transition matrix**, so each row sums to 1 and encodes "how much of my label should come from each of my neighbours."
3. Initializing a soft label distribution where labeled points are one-hot and unlabeled points are all zero.
4. Iteratively updating each point's distribution as a weighted average of its neighbours' distributions, **clamping** the labeled points back to their known class after every sweep so the ground truth is never overwritten.
5. Stopping when the distributions stop changing (or after a fixed number of iterations). The predicted class for each unlabeled point is the argmax of its final distribution.

The intuition is simple: a point surrounded by mostly "class 0" neighbours should itself be called "class 0," and that decision carries outward through the graph. This is the classical **manifold assumption** — points that lie close on a data manifold usually share the same label — and it lets you get surprisingly accurate predictions from only a handful of labeled seeds.

In `label_propagation_from_scratch.ipynb`, we:

- Derive the transition-matrix update and explain why clamping labeled points is necessary.
- Build an RBF similarity graph from scratch and visualize it as a heat-map / network.
- Demonstrate label propagation on a dataset where only a small fraction of points are labeled, and show how the labels "bleed" out to the rest of the graph.
- Study the effect of the kernel bandwidth and the `spread` parameter on convergence and accuracy.
- Compare our from-scratch propagation against scikit-learn's `LabelPropagation` as a sanity check.
