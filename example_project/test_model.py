import math
import unittest

from model import LogisticRegression


class TestModel(unittest.TestCase):
    def test_untrained_loss_is_log2(self):
        self.assertAlmostEqual(LogisticRegression(2).loss([[1.0, 2.0]], [1]), math.log(2))

    def test_step_reduces_loss(self):
        m, X, y = LogisticRegression(2), [[1.0, 0.0], [-1.0, 0.0]], [1, 0]
        m.step(X, y)
        self.assertLess(m.loss(X, y), math.log(2))


if __name__ == "__main__":
    unittest.main()
