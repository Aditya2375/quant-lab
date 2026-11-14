# quant-lab

Live: https://aditya2375.github.io/quant-lab/

Three quantitative finance studies, each with a live interactive demo, pinned input data,
fixed seeds, and a test suite that fails if a published number stops matching its inputs.

| # | study | question it answers | live demo |
|---|-------|---------------------|-----------|
| 01 | `options-pricer/` | Does Monte Carlo actually converge to Black-Scholes, how fast, and where does the model itself break? | [options.html](https://aditya2375.github.io/quant-lab/options.html) |
| 02 | `pairs-backtester/` | Can a cointegration spread strategy survive honest out-of-sample testing with transaction costs? | [pairs.html](https://aditya2375.github.io/quant-lab/pairs.html) |
| 03 | `vol-forecaster/` | Can machine learning beat GARCH(1,1) at forecasting realized volatility? | [vol.html](https://aditya2375.github.io/quant-lab/vol.html) |

## The rules every study follows

- **Pinned inputs.** Every market series is a CSV in `data/raw/` with row counts, date ranges
  and sha256 hashes in `data/manifest.json` (fetched with `scripts/fetch_data.py`, free Yahoo
  Finance data via yfinance). No number depends on a provider serving the same bytes twice.
- **No lookahead.** Signals use yesterday's information or less; formation and trading windows
  are disjoint; walk-forward splits are explicit.
- **Costs are charged.** Every backtest states its transaction-cost assumption in basis points
  and applies it to every executed leg.
- **Errors come with bars.** Simulation results carry standard errors and confidence intervals.
  Forecasts are scored out-of-sample with named loss functions.
- **Limits are stated.** Each study ships an honest-limits note: what the model assumes, where
  it breaks, and what the results do not show.

## Testing

```
pip install -r requirements.txt
python3 -m unittest discover -s tests   # engine correctness + cross-checks
node tests/test_engine_js.mjs           # browser engine vs quadrature references
```

The JavaScript engine in `docs/assets/engine.js` (which powers the live demos) agrees with the
closed form to ~1e-13 and is pinned against scipy-quadrature reference values.

## Earlier work

`monte-carlo-risk/`, `pairs-trading/` (v1) and `ma-crossover/` were the first pass. An audit
found method bugs (double-counted drift drag, level/log spread mismatch, drawdown without the
initial peak); the old metrics were withdrawn and the corrected code is kept for provenance.
Study 02 is a ground-up rewrite of the pairs idea under the rules above.

Educational only. Not investment advice.
Contact: adityakulkarnicoding@gmail.com
