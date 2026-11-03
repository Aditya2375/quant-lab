import unittest,sys,importlib.util,pathlib
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from common import perf_stats,load_prices
spec=importlib.util.spec_from_file_location('mc',ROOT/'monte-carlo-risk/monte_carlo.py');mc=importlib.util.module_from_spec(spec);spec.loader.exec_module(mc)
class Methods(unittest.TestCase):
 def test_initial_drawdown(self):self.assertEqual(perf_stats(pd.Series([-.1,0]))['max_drawdown'],-.1)
 def test_bad_returns(self):
  with self.assertRaises(ValueError):perf_stats(pd.Series([-1.1]))
 def test_log_drift(self):
  paths=mc.simulate(np.array([.1]),np.array([[0.]]),np.array([1.]),2,252,100.,1)
  self.assertAlmostEqual(paths[0,-1],100*np.exp(.1),places=8)
 def test_reproducible(self):
  args=(np.array([.05]),np.array([[.04]]),np.array([1.]),20,10,100.,42)
  np.testing.assert_equal(mc.simulate(*args),mc.simulate(*args))
 def test_invalid_bounds(self):
  with self.assertRaises(ValueError):mc.simulate(np.array([.1]),np.array([[.1]]),np.array([1.]),0,10,100.,1)
 def test_ticker_traversal(self):
  with self.assertRaises(ValueError):load_prices(['../../secret'])
 def test_tail(self):
  x=mc.risk_metrics(np.array([50.,80.,100.,120.]),100.)
  self.assertEqual(x['prob_loss'],.5);self.assertGreaterEqual(x['es_95'],x['var_95'])
if __name__=='__main__':unittest.main()

class PairBoundary(unittest.TestCase):
 def test_failed_screen_stops(self):
  from unittest.mock import patch
  spec=importlib.util.spec_from_file_location('pair',ROOT/'pairs-trading/pairs_trading.py');pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
  idx=pd.date_range('2020-01-01',periods=700)
  data=pd.DataFrame({'Y':np.exp(np.linspace(4,5,700)+np.sin(np.arange(700))*.01),'X':np.exp(np.linspace(3,4,700))},index=idx)
  with patch.object(pair,'load_prices',return_value=data),patch.object(pair,'coint',return_value=(0,.5,None)):
   with self.assertRaisesRegex(ValueError,'failed cointegration'):pair.run(['Y','X'],'5y',1.,20,2.,.001)
