import numpy as np

y_actual = np.array([0]*900 + [1]*100)
np.random.shuffle(y_actual)

y_pred = np.array([0]*900 + [1]*100)
np.random.shuffle(y_pred)

TP = np.sum((y_actual == 1) & (y_pred == 1))
TN = np.sum((y_actual == 0) & (y_pred == 0))
FP = np.sum((y_actual == 0) & (y_pred == 1))
FN = np.sum((y_actual == 1) & (y_pred == 0))

print(f"Confusion Matrix:\n[[{TN}, {FP}]\n [{FN}, {TP}]]")

print("Decision: Precision is used when FP is costly, Recall is used when FN is costly.")

priority = np.random.choice(['FP', 'FN'])
beta = 0.5 if priority == 'FP' else 2
print(f"Randomly assigned priority to: {priority} => Beta = {beta}")

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0

f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) else 0
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
print(f"F-beta Score: {f_beta:.4f}")
