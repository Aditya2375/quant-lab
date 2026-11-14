"""Tests for the volatility forecaster."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "vol-forecaster"))
from vol_forecaster import (garch_variances, garch_fit, garch_negloglik,
                            qlike, mse, dm_test, build_features,
                            load_returns, FEATURE_NAMES)


class TestGarch(unittest.TestCase):
    def test_recursion_constant_when_no_shocks(self):
        # with zero returns, variance decays geometrically to omega/(1-b)
        w, a, b = 0.0001, 0.1, 0.8
        r = np.zeros(50)
        s2 = garch_variances((w, a, b), r)
        self.assertAlmostEqual(s2[-1], w / (1 - b), places=6)

    def test_recursion_spike_raises_variance(self):
        w, a, b = 1e-6, 0.1, 0.85
        r = np.zeros(30)
        r[10] = 0.05
        s2 = garch_variances((w, a, b), r)
        self.assertGreater(s2[11], s2[9])
        self.assertGreater(s2[12], s2[20])  # decay after the shock

    def test_fit_recovers_simulated_params(self):
        rng = np.random.default_rng(7)
        w, a, b = 1e-6, 0.09, 0.88
        n = 6000
        r = np.empty(n)
        s2 = 1e-4
        for t in range(n):
            s2 = w + a * r[t - 1] ** 2 + b * s2 if t else 1e-4
            r[t] = rng.normal() * math.sqrt(s2)
        fit = garch_fit(r)
        self.assertAlmostEqual(fit["alpha"], a, delta=0.04)
        self.assertAlmostEqual(fit["beta"], b, delta=0.04)
        self.assertGreater(fit["persistence"], 0.9)

    def test_mle_beats_arbitrary_params(self):
        dates, r = load_returns()
        r = r[:2000]
        fit = garch_fit(r)
        nll_fit = garch_negloglik((fit["omega"], fit["alpha"], fit["beta"]), r)
        nll_other = garch_negloglik((1e-5, 0.2, 0.5), r)
        self.assertLess(nll_fit, nll_other)


class TestLosses(unittest.TestCase):
    def test_qlike_prefers_better_forecast(self):
        rng = np.random.default_rng(3)
        true_var = 1e-4 * (1 + 0.5 * np.sin(np.arange(500) / 20.0))
        realized = true_var * rng.chisquare(1, 500)
        good = np.full(500, np.mean(true_var))
        bad = np.full(500, 5 * np.mean(true_var))
        self.assertLess(qlike(realized, good), qlike(realized, bad))

    def test_dm_detects_identical_and_different(self):
        rng = np.random.default_rng(5)
        a = np.abs(rng.normal(size=400))
        stat_same, p_same, _ = dm_test(a, a)
        self.assertEqual(stat_same, 0.0)
        b = a + 0.5
        stat, p, md = dm_test(a, b)
        self.assertLess(stat, -2.0)
        self.assertLess(p, 0.05)


class TestFeatures(unittest.TestCase):
    def test_no_lookahead_in_features(self):
        rng = np.random.default_rng(9)
        r = rng.normal(scale=0.01, size=300)
        idx = np.arange(63, 300)
        X = build_features(r, idx)
        # perturb a return AFTER the last feature day: features must not move
        r2 = r.copy()
        r2[299] = 0.5
        X2 = build_features(r2, idx[:-1])
        np.testing.assert_allclose(X[:-1], X2, rtol=0, atol=0)

    def test_feature_shape(self):
        r = np.random.default_rng(1).normal(scale=0.01, size=200)
        X = build_features(r, np.arange(63, 200))
        self.assertEqual(X.shape, (200 - 63, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
