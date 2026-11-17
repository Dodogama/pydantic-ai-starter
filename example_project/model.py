import math


class LogisticRegression:
    def __init__(self, n_features, lr=0.1):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr

    def predict_proba(self, x):
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        e = math.exp(z)
        return e / (1 + e)

    def predict(self, X):
        return [int(self.predict_proba(x) >= 0.5) for x in X]

    def loss(self, X, y):
        """Mean binary cross-entropy over the batch."""
        raise NotImplementedError

    def step(self, X, y):
        """One full-batch gradient-descent step. Returns the loss before the update."""
        raise NotImplementedError
