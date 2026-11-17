import random


def make_data(n=400, seed=0):
    """Linearly separable-ish 2D data: label is sign(a + 2b + noise). Returns (train, val)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        X.append([a, b])
        y.append(1 if a + 2 * b + rng.gauss(0, 0.5) > 0 else 0)
    split = int(n * 0.8)
    return (X[:split], y[:split]), (X[split:], y[split:])
