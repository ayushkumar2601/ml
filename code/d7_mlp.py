import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labels, y_int = np.unique(y, return_inverse=True)
y_onehot = np.zeros((y_int.size, y_int.max() + 1))
y_onehot[np.arange(y_int.size), y_int] = 1

X = (X - X.mean(axis=0)) / X.std(axis=0)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return z * (1 - z)

np.random.seed(42)
input_dim = X.shape[1]
hidden_dim = 5
output_dim = y_onehot.shape[1]

W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

lr = 0.1
epochs = 100

for epoch in range(1, epochs + 1):
    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    loss = np.mean((y_onehot - a2)**2)
    
    # Backprop
    d2 = (a2 - y_onehot) * sigmoid_deriv(a2)
    dW2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0, keepdims=True)
    
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)
    dW1 = np.dot(X.T, d1)
    db1 = np.sum(d1, axis=0, keepdims=True)
    
    W1 -= lr * dW1 / X.shape[0]
    b1 -= lr * db1 / X.shape[0]
    W2 -= lr * dW2 / X.shape[0]
    b2 -= lr * db2 / X.shape[0]
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

predictions = np.argmax(a2, axis=1)
accuracy = np.mean(predictions == y_int)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
