import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

w = np.zeros(X.shape[1])
b = 0
lr = 1

while True:
    errors = 0
    for i in range(len(X)):
        if y[i] * (np.dot(X[i], w) + b) <= 0:
            w += lr * y[i] * X[i]
            b += lr * y[i]
            errors += 1
    if errors == 0:
        break

print(f"Final weights: {w}, bias: {b}")
