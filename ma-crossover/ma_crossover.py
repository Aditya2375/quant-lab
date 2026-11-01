"""50/200-day moving average crossover ("golden cross") backtest.

The simplest trend-following rule there is: hold the index while the
50-day average sits above the 200-day average, sit in cash otherwise.

The interesting part is not the rule, it is testing it honestly:

* signals are delayed one day - you act on yesterday's cross, so there is
  no lookahead bias;
* every switch pays transaction costs;
* the result is judged against simply buying and holding.

Usage:
    python ma_crossover.py                     # SPY, 15 years
    python ma_crossover.py --ticker QQQ --fast 50 --slow 200 --cost 0.001
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import load_prices, perf_stats, style_ax

OUT = Path(__file__).resolve().parent / "output"
DASH = Path(__file__).resolve().parent.parent / "docs" / "data"


def run(ticker: str, period: str, fast: int, slow: int, cost: float) -> dict:
    px = load_prices([ticker], period)[ticker]

    sma_fast = px.rolling(fast).mean()
    sma_slow = px.rolling(slow).mean()
    signal = (sma_fast > sma_slow).astype(float)
    position = signal.shift(1).fillna(0)  # act the day after the cross

    ret = px.pct_change()
    switches = position.diff().abs().fillna(0)
    strategy = position * ret - cost * switches
    valid = sma_slow.notna()
    strategy, ret = strategy[valid], ret[valid]
    position, switches = position[valid], switches[valid]

    stats = perf_stats(strategy)
    bench = perf_stats(ret)
    stats["benchmark"] = bench
    stats["n_switches"] = int((switches > 0).sum())
    stats["time_in_market"] = round(float(position.mean()), 3)

    # --- charts -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})
    p = px[valid]
    axes[0].plot(p.index, p, color="#333", lw=1, label=ticker, alpha=0.8)
    axes[0].plot(p.index, sma_fast[valid], lw=1.2, label=f"SMA {fast}")
    axes[0].plot(p.index, sma_slow[valid], lw=1.2, label=f"SMA {slow}")
    buys = switches[(switches > 0) & (position > 0)].index
    sells = switches[(switches > 0) & (position == 0)].index
    axes[0].scatter(buys, p.loc[buys], marker="^", color="#54A24B", s=60,
                    zorder=5, label="golden cross (buy)")
    axes[0].scatter(sells, p.loc[sells], marker="v", color="#E45756", s=60,
                    zorder=5, label="death cross (sell)")
    style_ax(axes[0], f"{ticker} with {fast}/{slow}-day moving averages",
             ylabel="price ($)")
    axes[0].legend(fontsize=9)

    eq_s = (1 + strategy).cumprod()
    eq_b = (1 + ret).cumprod()
    axes[1].plot(eq_s.index, eq_s, lw=1.6, label="crossover strategy")
    axes[1].plot(eq_b.index, eq_b, lw=1.6, label="buy & hold")
    axes[1].set_yscale("log")
    style_ax(axes[1], "equity curves, log scale (growth of $1)", ylabel="value ($)")
    axes[1].legend()

    dd_s = eq_s / eq_s.cummax() - 1
    dd_b = eq_b / eq_b.cummax() - 1
    axes[2].fill_between(dd_s.index, dd_s, 0, alpha=0.6, label="strategy")
    axes[2].fill_between(dd_b.index, dd_b, 0, alpha=0.4, label="buy & hold")
    style_ax(axes[2], "drawdown", ylabel="from peak")
    axes[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "crossover.png", dpi=140)
    plt.close(fig)

    dash = {
        "summary": {"ticker": ticker, "fast": fast, "slow": slow,
                    "cost_per_switch": cost, **stats},
        "dates": [d.strftime("%Y-%m-%d") for d in eq_s.index],
        "equity_strategy": eq_s.round(4).tolist(),
        "equity_bench": eq_b.round(4).tolist(),
        "dd_strategy": dd_s.round(4).tolist(),
        "dd_bench": dd_b.round(4).tolist(),
    }
    (DASH / "ma-crossover.json").write_text(json.dumps(dash))
    return {"ticker": ticker, **{k: v for k, v in stats.items() if k != "benchmark"},
            "buy_and_hold": bench}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--period", default="15y")
    ap.add_argument("--fast", type=int, default=50)
    ap.add_argument("--slow", type=int, default=200)
    ap.add_argument("--cost", type=float, default=0.001, help="10 bps per switch")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    DASH.mkdir(parents=True, exist_ok=True)
    summary = run(args.ticker, args.period, args.fast, args.slow, args.cost)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
