import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
X = df.iloc[:, 1:3].values
epsilon = 2
min_pts = 2

labels = np.full(X.shape[0], -1)
cluster_id = 0

def region_query(p_idx):
    return np.where(np.linalg.norm(X - X[p_idx], axis=1) <= epsilon)[0]

for p in range(X.shape[0]):
    if labels[p] != -1:
        continue
    neighbors = region_query(p)
    if len(neighbors) < min_pts:
        labels[p] = -1 # Noise
    else:
        labels[p] = cluster_id
        i = 0
        while i < len(neighbors):
            pn = neighbors[i]
            if labels[pn] == -1:
                labels[pn] = cluster_id
            elif labels[pn] < 0:
                labels[pn] = cluster_id
                pn_neighbors = region_query(pn)
                if len(pn_neighbors) >= min_pts:
                    neighbors = np.append(neighbors, pn_neighbors)
            i += 1
        cluster_id += 1

print("DBSCAN Labels (-1 is noise):", labels)
