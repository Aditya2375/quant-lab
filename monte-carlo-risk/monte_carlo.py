"""Monte Carlo simulation of portfolio risk.

Estimates drift and the covariance structure of a small portfolio from
historical daily log returns, then simulates thousands of correlated
geometric Brownian motion paths one year forward. From the distribution of
outcomes we read off the risk numbers a risk manager actually quotes:
Value at Risk (VaR) and Expected Shortfall (CVaR) at 95% and 99%.

Usage:
    python monte_carlo.py                      # default portfolio, 10k paths
    python monte_carlo.py --sims 20000 --tickers AAPL MSFT TSLA
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
from common import TRADING_DAYS, load_prices, style_ax

OUT = Path(__file__).resolve().parent / "output"
DASH = Path(__file__).resolve().parent.parent / "docs" / "data"


def simulate(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray,
             n_sims: int, days: int, start_value: float, seed: int) -> np.ndarray:
    """Correlated GBM paths, daily rebalanced to `weights`. Returns (n_sims, days+1)."""
    if not (1<=n_sims<=100000 and 1<=days<=2520 and start_value>0 and np.isfinite(start_value)): raise ValueError("Invalid simulation bounds")
    if len(mu)<1 or len(mu)>20 or cov.shape!=(len(mu),len(mu)) or weights.shape!=(len(mu),): raise ValueError("Dimension mismatch")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all() or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1): raise ValueError("Finite long-only unit weights required")
    if n_sims*days*len(mu)>20000000: raise ValueError("Simulation exceeds memory budget")
    if not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<-1e-10: raise ValueError("Covariance must be positive semidefinite")
    rng = np.random.default_rng(seed)
    eigen,vec=np.linalg.eigh(cov)
    chol=vec @ np.diag(np.sqrt(np.maximum(eigen,0)))
    drift = mu / TRADING_DAYS
    n_assets = len(mu)

    z = rng.standard_normal((n_sims, days, n_assets))
    # daily log returns per asset: drift - 0.5*var + correlated shock
    var_daily = np.diag(cov) / TRADING_DAYS
    log_ret = drift + z @ chol.T / np.sqrt(TRADING_DAYS)
    simple_ret = np.exp(log_ret) - 1.0
    port_ret = simple_ret @ weights  # (n_sims, days)
    paths = start_value * np.cumprod(1 + port_ret, axis=1)
    return np.hstack([np.full((n_sims, 1), start_value), paths])


def risk_metrics(final: np.ndarray, start_value: float) -> dict:
    if len(final)<2 or not np.isfinite(final).all() or start_value<=0: raise ValueError("Invalid final values")
    pnl = final - start_value
    out = {}
    for conf in (0.95, 0.99):
        tail = pnl <= np.percentile(pnl, (1 - conf) * 100)
        out[f"var_{int(conf*100)}"] = round(float(-np.percentile(pnl, (1 - conf) * 100)), 2)
        out[f"es_{int(conf*100)}"] = round(float(-pnl[tail].mean()), 2)
    out["prob_loss"] = round(float((pnl < 0).mean()), 4)
    out["expected_final"] = round(float(final.mean()), 2)
    out["median_final"] = round(float(np.median(final)), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "NVDA"])
    ap.add_argument("--period", default="5y", help="history used for estimation")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--days", type=int, default=TRADING_DAYS, help="horizon in trading days")
    ap.add_argument("--start-value", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    DASH.mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.tickers, args.period)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu = log_ret.mean().values * TRADING_DAYS
    cov = log_ret.cov().values * TRADING_DAYS
    weights = np.full(len(args.tickers), 1 / len(args.tickers))

    paths = simulate(mu, cov, weights, args.sims, args.days, args.start_value, args.seed)
    final = paths[:, -1]
    stats = risk_metrics(final, args.start_value)

    # ---- chart 1: distribution of final values --------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(final, bins=100, color="#4C78A8", alpha=0.85)
    var95 = args.start_value - stats["var_95"]
    es95 = args.start_value - stats["es_95"]
    ax.axvline(var95, color="#F58518", lw=2, label=f"VaR 95%  (${var95:,.0f})")
    ax.axvline(es95, color="#E45756", lw=2, label=f"ES 95%  (${es95:,.0f})")
    ax.axvline(final.mean(), color="#54A24B", lw=2, ls="--",
               label=f"mean  (${final.mean():,.0f})")
    style_ax(ax, f"Simulated portfolio value after {args.days} trading days "
                 f"({args.sims:,} paths)", "portfolio value ($)", "frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "distribution.png", dpi=140)
    plt.close(fig)

    # ---- chart 2: percentile fan over the horizon -----------------------------
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(args.days + 1)
    ax.fill_between(x, pct[0], pct[4], color="#4C78A8", alpha=0.25, label="5th-95th pct")
    ax.fill_between(x, pct[1], pct[3], color="#4C78A8", alpha=0.45, label="25th-75th pct")
    ax.plot(x, pct[2], color="#1F3B63", lw=2, label="median")
    ax.axhline(args.start_value, color="gray", lw=1, ls=":")
    style_ax(ax, "Monte Carlo cone of outcomes", "trading days", "portfolio value ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cone.png", dpi=140)
    plt.close(fig)

    summary = {
        "method_version": 2,
        "data_start": str(prices.index[0].date()), "data_end": str(prices.index[-1].date()),
        "tickers": args.tickers,
        "weights": list(weights.round(4)),
        "estimation_period": args.period,
        "sims": args.sims,
        "horizon_days": args.days,
        "start_value": args.start_value,
        **stats,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # plotly-ready payload for the web dashboard
    counts, edges = np.histogram(final, bins=80)
    dash = {
        "summary": summary,
        "hist": {"centers": ((edges[:-1] + edges[1:]) / 2).round(2).tolist(),
                 "counts": counts.tolist()},
        "cone": {"days": x.tolist(), **{
            k: pct[i].round(2).tolist()
            for i, k in enumerate(["p5", "p25", "p50", "p75", "p95"])}},
        "markers": {"var95": round(float(var95), 2), "es95": round(float(es95), 2),
                    "mean": round(float(final.mean()), 2)},
    }
    (DASH / "monte-carlo.json").write_text(json.dumps(dash))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
