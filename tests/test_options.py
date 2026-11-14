import unittest, sys, importlib.util, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('op', ROOT / 'options-pricer' / 'options_pricer.py')
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)

S, K, T, R, SIG, Q = 100.0, 105.0, 0.5, 0.03, 0.25, 0.01

class ClosedForm(unittest.TestCase):
    def test_known_call(self):
        # independent reference: scipy quadrature of the risk-neutral
        # expectation gives 5.349804578880892 +/- 1.5e-12
        p = op.bs_price(S, K, T, R, SIG, Q, 'call')
        self.assertAlmostEqual(p, 5.349804578880892, places=10)
    def test_put_call_parity(self):
        c = op.bs_price(S, K, T, R, SIG, Q, 'call'); p = op.bs_price(S, K, T, R, SIG, Q, 'put')
        self.assertAlmostEqual(c - p, S*math.exp(-Q*T) - K*math.exp(-R*T), places=12)
    def test_zero_vol(self):
        self.assertAlmostEqual(op.bs_price(S, K, T, R, 0.0, Q, 'call'),
                               max(S*math.exp(-Q*T) - K*math.exp(-R*T), 0.0), places=12)
    def test_bad_inputs(self):
        for bad in [(-1, K, T, R, SIG), (S, 0, T, R, SIG), (S, K, -1, R, SIG), (S, K, T, R, -0.1)]:
            with self.assertRaises(ValueError): op.bs_price(*bad)
    def test_greeks_finite_diff(self):
        g = op.bs_greeks(S, K, T, R, SIG, Q, 'call')
        h = 1e-4
        fd_delta = (op.bs_price(S+h, K, T, R, SIG, Q) - op.bs_price(S-h, K, T, R, SIG, Q)) / (2*h)
        fd_vega = (op.bs_price(S, K, T, R, SIG+h, Q) - op.bs_price(S, K, T, R, SIG-h, Q)) / (2*h)
        self.assertAlmostEqual(g['delta'], fd_delta, places=4)
        self.assertAlmostEqual(g['vega'], fd_vega, places=2)
    def test_implied_vol_roundtrip(self):
        p = op.bs_price(S, K, T, R, SIG, Q, 'put')
        self.assertAlmostEqual(op.implied_vol(p, S, K, T, R, Q, 'put'), SIG, places=8)
    def test_implied_vol_rejects_arbitrage_price(self):
        with self.assertRaises(ValueError): op.implied_vol(1e9, S, K, T, R, Q, 'call')

class MonteCarlo(unittest.TestCase):
    def test_converges_to_closed_form(self):
        exact = op.bs_price(S, K, T, R, SIG, Q, 'call')
        est = op.mc_price(S, K, T, R, SIG, Q, 'call', n=200_000, seed=42)
        self.assertLess(abs(est['price'] - exact), 4 * est['se'])
    def test_antithetic_plus_control_not_worse(self):
        exact = op.bs_price(S, K, T, R, SIG, Q, 'call')
        plain = op.mc_price(S, K, T, R, SIG, Q, 'call', n=50_000, seed=1)
        avcv = op.mc_price(S, K, T, R, SIG, Q, 'call', n=50_000, seed=1, antithetic=True, control=True)
        self.assertLessEqual(avcv['se'], plain['se'])
    def test_qmc_beats_plain(self):
        exact = op.bs_price(S, K, T, R, SIG, Q, 'call')
        plain = op.mc_price(S, K, T, R, SIG, Q, 'call', n=65_536, seed=5)
        sob = op.qmc_price(S, K, T, R, SIG, Q, 'call', n=65_536, seed=5)
        self.assertLess(abs(sob['price'] - exact), abs(plain['price'] - exact))
    def test_deterministic(self):
        a = op.mc_price(S, K, T, R, SIG, Q, 'call', n=10_000, seed=9)
        b = op.mc_price(S, K, T, R, SIG, Q, 'call', n=10_000, seed=9)
        self.assertEqual(a['price'], b['price'])
    def test_put(self):
        exact = op.bs_price(S, K, T, R, SIG, Q, 'put')
        est = op.qmc_price(S, K, T, R, SIG, Q, 'put', n=65_536, seed=3)
        self.assertLess(abs(est['price'] - exact), 5 * est['se'] + 1e-4)

if __name__ == '__main__': unittest.main()
