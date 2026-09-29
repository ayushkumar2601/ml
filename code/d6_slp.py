import numpy as np

def train_slp():
    # Logical AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    w = np.zeros(2)
    b = 0
    lr = 0.1
    
    while True:
        errors = 0
        for i in range(4):
            y_pred = 1 if (np.dot(X[i], w) + b) >= 0 else 0
            if y[i] != y_pred:
                w += lr * (y[i] - y_pred) * X[i]
                b += lr * (y[i] - y_pred)
                errors += 1
        if errors == 0:
            break
    return w, b

w, b = train_slp()
print(f"Trained SLP for AND gate. Weights: {w}, Bias: {b}")

print("Testing the trained perceptron:")
for x in [[0,0], [0,1], [1,0], [1,1]]:
    pred = 1 if (np.dot(x, w) + b) >= 0 else 0
    print(f"Input {x} -> Prediction {pred}")
