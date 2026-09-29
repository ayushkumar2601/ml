import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
X = df.iloc[:, 1:3].values
K = 3

np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], K, replace=False)]

while True:
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(K)])
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

print("Final centroids:\n", centroids)
print("Labels:", labels)
