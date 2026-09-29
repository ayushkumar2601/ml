import numpy as np
import pandas as pd

df = pd.read_csv('demo.csv')
# Assuming format from image: Sl No, Marks in Class test, Marks in Semester
X = df.iloc[:, 1].values  
Y = df.iloc[:, 2].values  

n = len(X)
m = (n * np.sum(X*Y) - np.sum(X)*np.sum(Y)) / (n * np.sum(X**2) - np.sum(X)**2)
c = (np.sum(Y) - m * np.sum(X)) / n

pred_20 = m * 20 + c
print(f"Estimated marks for 20 in class test: {pred_20}")

Y_pred = m * X + c
mae = np.mean(np.abs(Y - Y_pred))
mse = np.mean((Y - Y_pred)**2)
rmse = np.sqrt(mse)
ss_tot = np.sum((Y - np.mean(Y))**2)
ss_res = np.sum((Y - Y_pred)**2)
r2 = 1 - (ss_res / ss_tot)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
