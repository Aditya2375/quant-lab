"""Tests for the pairs backtester."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "pairs-backtester"))
from pairs_backtester import (ols, adf_test, half_life, backtest, sharpe,
                              max_drawdown, load_close, align, run_pair,
                              mat_inverse, solve_ls)

import random


class TestStats(unittest.TestCase):
    def test_ols_recovers_line(self):
        random.seed(7)
        x = [random.gauss(0, 1) for _ in range(500)]
        y = [1.3 + 2.1 * v + random.gauss(0, 0.01) for v in x]
        alpha, beta = ols(x, y)
        self.assertAlmostEqual(alpha, 1.3, places=1)
        self.assertAlmostEqual(beta, 2.1, places=1)

    def test_mat_inverse(self):
        m = [[4.0, 1.0], [2.0, 3.0]]
        inv = mat_inverse(m)
        for i in range(2):
            for j in range(2):
                ident = sum(m[i][k] * inv[k][j] for k in range(2))
                self.assertAlmostEqual(ident, 1.0 if i == j else 0.0, places=9)

    def test_adf_flags_random_walk(self):
        random.seed(11)
        rw = [0.0]
        for _ in range(1500):
            rw.append(rw[-1] + random.gauss(0, 1))
        t, p = adf_test(rw)
        self.assertGreater(t, -2.57, "random walk must not reject unit root")

    def test_adf_flags_stationary(self):
        random.seed(13)
        ar = [0.0]
        for _ in range(1500):
            ar.append(0.9 * ar[-1] + random.gauss(0, 1))
        t, p = adf_test(ar)
        self.assertLess(t, -3.43, "AR(1) with 0.9 must reject unit root")

    def test_half_life(self):
        random.seed(17)
        ar = [0.0]
        for _ in range(3000):
            ar.append(0.95 * ar[-1] + random.gauss(0, 0.1))
        hl = half_life(ar)
        self.assertTrue(10 < hl < 18, f"expected ~13.5 days, got {hl}")


class TestBacktest(unittest.TestCase):
    def test_no_lookahead_flat_when_never_triggered(self):
        # spread pinned at mu: z always 0, no trades, equity flat
        dates = [f"2024-01-{d:02d}" for d in range(1, 29)]
        px_a = [100.0] * len(dates)
        px_b = [50.0] * len(dates)
        bt = backtest(dates, px_a, px_b, beta=1.0,
                      mu=math.log(2.0), sigma=0.01)
        self.assertEqual(bt["trades"], [])
        self.assertAlmostEqual(bt["equity"][-1], 1.0, places=12)

    def test_costs_reduce_equity(self):
        random.seed(19)
        n = 400
        dates = [str(i) for i in range(n)]
        px_b = [100.0] * n
        # oscillating spread forces repeated entries and exits
        px_a = [100.0 * math.exp(0.20 * math.sin(i / 5.0)) for i in range(n)]
        bt = backtest(dates, px_a, px_b, beta=1.0, mu=0.0, sigma=0.10,
                      entry=1.0, exit_z=0.2, cost_bps=5.0)
        self.assertLess(bt["equity"][-1], bt["equity_nocost"][-1])
        self.assertGreater(len(bt["trades"]), 5)

    def test_metrics(self):
        self.assertAlmostEqual(max_drawdown([1, 1.2, 0.9, 1.1]), -0.25)
        self.assertEqual(sharpe([1.0] * 50), 0.0)


class TestLivePairs(unittest.TestCase):
    def test_vma_is_cointegrated(self):
        r = run_pair("V", "MA")
        self.assertLess(r["adf_t"], -2.86)
        self.assertEqual(r["formation"]["days"], 1260)
        self.assertGreater(r["trading"]["days"], 1000)

    def test_ups_fdx_is_not(self):
        r = run_pair("UPS", "FDX")
        self.assertGreater(r["adf_t"], -2.57)


if __name__ == "__main__":
    unittest.main()
