import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
X = df.iloc[:, 1].values
y = df.iloc[:, 2].values
y = np.where(y == np.min(y), 0, 1)

mean_X = np.mean(X)
std_X = np.std(X)
X = (X - mean_X) / std_X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    m = len(y)
    h = sigmoid(X * w + b)
    return -1/m * np.sum(y * np.log(h + 1e-15) + (1-y) * np.log(1-h + 1e-15))

w, b = 0.0, 0.0
lr, epochs = 0.1, 10000

for _ in range(epochs):
    h = sigmoid(X * w + b)
    w -= lr * np.mean((h - y) * X)
    b -= lr * np.mean(h - y)

x_test = (20 - mean_X) / std_X
prob = sigmoid(w * x_test + b)
print(f"Weights: w={w}, b={b}")
print(f"Probability of passing for 20 marks: {prob}")
print("Pass" if prob >= 0.5 else "Fail")
