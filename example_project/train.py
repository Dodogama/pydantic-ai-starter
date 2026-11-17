import argparse
import math
import sys

from data import make_data
from model import LogisticRegression


def train(steps, X, y):
    m = LogisticRegression(len(X[0]))
    losses = [m.step(X, y) for _ in range(steps)]
    return m, losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--smoke", action="store_true", help="tiny subset; fail unless loss decreases")
    a = p.parse_args()
    (X, y), _ = make_data()
    if a.smoke:
        X, y = X[:32], y[:32]
    _, losses = train(a.steps, X, y)
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    if any(math.isnan(v) for v in losses):
        sys.exit("loss is NaN")
    if a.smoke and not losses[-1] < losses[0]:
        sys.exit("smoke: loss did not decrease")
