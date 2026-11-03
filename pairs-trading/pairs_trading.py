"""Pairs trading (statistical arbitrage) on a cointegrated pair.

Method, following the classic Engle-Granger recipe:

1. Formation window: confirm the two price series are cointegrated
   (Engle-Granger two-step test) and estimate the hedge ratio with OLS.
2. Trading window: track the spread as a rolling Z-score. Go long the
   spread at -2 standard deviations, short it at +2, and flatten when the
   spread mean-reverts through zero.
3. Score it honestly: the hedge ratio comes only from the formation
   window, and every position change pays transaction costs.

Usage:
    python pairs_trading.py                      # V vs MA, 5y of history
    python pairs_trading.py --pair KO PEP --entry 2.0 --cost 0.0005
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import load_prices, perf_stats, style_ax

OUT = Path(__file__).resolve().parent / "output"
DASH = Path(__file__).resolve().parent.parent / "docs" / "data"


def run(pair: list[str], period: str, formation_years: float, z_window: int,
        entry: float, cost: float) -> dict:
    if len(pair)!=2 or pair[0]==pair[1] or formation_years<=0 or z_window<2 or entry<=0 or not 0<=cost<1: raise ValueError("Invalid pair/parameters")
    y, x = pair
    prices = load_prices(pair, period)
    split = prices.index[0] + pd.Timedelta(days=int(formation_years * 365))
    form, trade = prices.loc[prices.index<split], prices.loc[prices.index>=split]
    if len(form)<60 or len(trade)<z_window+2: raise ValueError("Insufficient disjoint formation/trading data")

    # --- formation: cointegration test + hedge ratio --------------------------
    score, pvalue, _ = coint(np.log(form[y]), np.log(form[x]))
    X = sm.add_constant(np.log(form[x]))
    fit = sm.OLS(np.log(form[y]), X).fit()
    beta = float(fit.params.iloc[1])

    if not np.isfinite(pvalue) or pvalue>0.05: raise ValueError("Formation log-price pair failed cointegration screen; no strategy result emitted")
    if not np.isfinite(beta) or beta<=0: raise ValueError("Positive finite hedge ratio required")
    # --- trading: spread, z-score, positions ----------------------------------
    spread = np.log(trade[y]) - float(fit.params.iloc[0]) - beta * np.log(trade[x])
    mu = spread.rolling(z_window).mean()
    sigma = spread.rolling(z_window).std()
    z = (spread - mu) / sigma

    pos = pd.Series(0.0, index=trade.index)
    current = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = current
            continue
        if current == 0:
            if zi > entry:
                current = -1.0
            elif zi < -entry:
                current = 1.0
        elif current > 0 and zi >= 0:
            current = 0.0
        elif current < 0 and zi <= 0:
            current = 0.0
        pos.iloc[i] = current

    # --- P&L: dollar-neutral spread returns, costs on position changes --------
    ret_y = trade[y].pct_change()
    ret_x = trade[x].pct_change()
    spread_ret = (ret_y - beta * ret_x)/(1+abs(beta))
    executed=pos.shift(1).fillna(0)
    trades=executed.diff().abs().fillna(executed.abs())
    strategy=executed*spread_ret-cost*trades
    strategy = strategy.dropna()

    if pvalue > 0.05:
        print(f"WARNING: {y}/{x} fails the Engle-Granger test on the formation "
              f"window (p = {pvalue:.3f}). Trading a non-cointegrated pair is a "
              f"coin flip - the backtest below is shown for completeness only.",
              file=sys.stderr)

    stats = perf_stats(strategy)
    stats["n_trades"] = int((trades > 0).sum())
    stats["time_in_market"] = round(float((pos != 0).mean()), 3)
    stats["coint_pvalue"] = round(float(pvalue), 4)
    stats["hedge_ratio"] = round(beta, 4)
    stats["buy_hold_" + y.lower()] = perf_stats(ret_y.loc[strategy.index])["total_return"]

    # --- charts ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2]})
    axes[0].plot(trade.index, trade[y], label=y, lw=1.4)
    axes[0].plot(trade.index, trade[x], label=x, lw=1.4)
    style_ax(axes[0], f"{y} vs {x} (trading window)", ylabel="price ($)")
    axes[0].legend()

    axes[1].plot(z.index, z, color="#4C78A8", lw=1.2, label="spread Z-score")
    axes[1].axhline(entry, color="#E45756", ls="--", lw=1)
    axes[1].axhline(-entry, color="#E45756", ls="--", lw=1)
    axes[1].axhline(0, color="gray", lw=1, ls=":")
    in_long = pos > 0
    in_short = pos < 0
    axes[1].fill_between(z.index, -entry, -4, where=in_long, color="#54A24B",
                         alpha=0.25, label="long spread", step="mid")
    axes[1].fill_between(z.index, entry, 4, where=in_short, color="#F58518",
                         alpha=0.25, label="short spread", step="mid")
    axes[1].set_ylim(-4, 4)
    style_ax(axes[1], f"spread Z-score ({z_window}-day rolling), entry at ±{entry}")
    axes[1].legend(loc="lower left", ncols=3, fontsize=9)

    equity = (1 + strategy).cumprod()
    bh = (1 + ret_y.loc[strategy.index]).cumprod()
    axes[2].plot(equity.index, equity, lw=1.6, label="pairs strategy")
    axes[2].plot(bh.index, bh, lw=1.4, alpha=0.8, label=f"buy & hold {y}")
    style_ax(axes[2], "equity curves (growth of $1)", ylabel="value ($)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "pairs.png", dpi=140)
    plt.close(fig)

    # --- dashboard payload ------------------------------------------------------
    eq_bench = bh
    dash = {
        "summary": {"method_version":2,"data_start":str(prices.index[0].date()),"data_end":str(prices.index[-1].date()),"pair": pair, "formation_years": formation_years,
                    "z_window": z_window, "entry_z": entry, "cost_per_trade": cost,
                    **stats},
        "dates": [d.strftime("%Y-%m-%d") for d in strategy.index],
        "zscore": [None if pd.isna(v) else round(float(v), 3)
                   for v in z.loc[strategy.index]],
        "position": pos.loc[strategy.index].tolist(),
        "equity_strategy": equity.round(4).tolist(),
        "equity_bench": eq_bench.round(4).tolist(),
        "entry_z": entry,
    }
    (DASH / "pairs-trading.json").write_text(json.dumps(dash, allow_nan=False))
    return {"pair": pair, **stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, default=["V", "MA"], metavar=("Y", "X"))
    ap.add_argument("--period", default="5y")
    ap.add_argument("--formation-years", type=float, default=3.0)
    ap.add_argument("--z-window", type=int, default=60)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--cost", type=float, default=0.0005, help="per position change")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    DASH.mkdir(parents=True, exist_ok=True)
    summary = run(args.pair, args.period, args.formation_years,
                  args.z_window, args.entry, args.cost)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
