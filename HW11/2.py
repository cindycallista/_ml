import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
