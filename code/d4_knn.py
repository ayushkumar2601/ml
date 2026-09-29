import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
# Assuming features and then label as last column
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

def knn(X_train, y_train, x_query, k=3):
    distances = np.sqrt(np.sum((X_train - x_query)**2, axis=1))
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    values, counts = np.unique(nearest_labels, return_counts=True)
    return values[np.argmax(counts)]

print("KNN Predictions on the training set (k=3):")
for i, x_q in enumerate(X):
    print(f"Sample {i} Prediction:", knn(X, y, x_q, k=3))
