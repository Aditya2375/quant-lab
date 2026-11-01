# quant-lab

Three core quant-finance techniques, implemented from scratch in Python and
tested on real market data.

**Interactive results dashboard: https://aditya2375.github.io/quant-lab/** -
the charts below, explorable in the browser (zoom, hover, toggle series). No
setup needed to view it.

| # | Project | Technique | Headline result |
|---|---------|-----------|-----------------|
| 1 | [monte-carlo-risk](monte-carlo-risk/) | Correlated Monte Carlo simulation | 1y VaR 95% of -$2,517 and expected shortfall of -$3,296 on a $10k equal-weight AAPL/MSFT/GOOGL/NVDA portfolio; 27.8% chance of a losing year |
| 2 | [pairs-trading](pairs-trading/) | Engle-Granger cointegration + Z-score mean reversion | V/MA is genuinely cointegrated (p = 0.01); the strategy returned +7.1% over 2 out-of-sample years at a -7.1% max drawdown, vs +44.8% buy-and-hold |
| 3 | [ma-crossover](ma-crossover/) | 50/200 golden-cross backtest, honest testing | On 15y of SPY the rule trails buy-and-hold (10.1% vs 14.6% CAGR) - most simple strategies fail exactly here |

## Layout

```
monte-carlo-risk/    Monte Carlo portfolio risk (VaR / expected shortfall)
pairs-trading/       cointegration screen + pairs-trading backtest
ma-crossover/        moving-average crossover backtest vs buy-and-hold
docs/                the dashboard site (GitHub Pages) + generated JSON
scripts/common.py    shared data download + performance metrics
```

## Quick start

```sh
pip install -r requirements.txt
python monte-carlo-risk/monte_carlo.py     # add --sims 50000 for a tighter tail
python pairs-trading/pairs_trading.py      # --pair KO PEP to try another pair
python ma-crossover/ma_crossover.py        # --ticker QQQ, --fast 20 --slow 100 ...
```

Each script downloads its own data (cached in `data/`), prints a JSON summary,
saves charts to its `output/` folder, and refreshes the JSON the dashboard
renders. Every script is deterministic (`--seed` for the simulation), so the
published numbers are reproducible.

## Notes on honesty

* The Monte Carlo model assumes log-normal returns and stable correlations;
  real tails are fatter and crash correlations worse.
* The pairs strategy estimates its hedge ratio on a separate formation
  window and pays costs on every position change; most candidate pairs fail
  the cointegration screen (KO/PEP included) and are rejected before
  backtesting.
* The crossover backtest delays signals one day (no lookahead), charges 10
  bps per switch, and reports against buy-and-hold.

Market data: daily adjusted closes via
[yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance).
