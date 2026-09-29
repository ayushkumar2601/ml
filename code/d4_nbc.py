import pandas as pd
import numpy as np

df = pd.read_csv('demo.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])
            
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
        
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            num = np.exp(-((x - self.mean[idx])**2) / (2 * self.var[idx]))
            den = np.sqrt(2 * np.pi * self.var[idx])
            posterior = np.sum(np.log(num / den))
            posteriors.append(prior + posterior)
        return self.classes[np.argmax(posteriors)]

nb = NaiveBayes()
nb.fit(X, y)
print("Naive Bayes predictions:", nb.predict(X))
