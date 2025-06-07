from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
cluster = AgglomerativeClustering(n_clusters=3)
cluster.fit(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
