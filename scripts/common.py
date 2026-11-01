"""Shared helpers: price downloads (cached), performance metrics, plotting style."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRADING_DAYS = 252


def load_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download adjusted daily closes with yfinance, cached as CSVs in data/."""
    DATA_DIR.mkdir(exist_ok=True)
    frames = {}
    for t in tickers:
        cache = DATA_DIR / f"{t.replace('^', '_')}.csv"
        if cache.exists():
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            import yfinance as yf

            df = yf.download(t, period=period, interval="1d", auto_adjust=True,
                             progress=False)
            if df.empty:
                raise RuntimeError(f"no data returned for {t}")
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.to_csv(cache, header=True)
        frames[t] = s
    prices = pd.concat(frames, axis=1).dropna()
    prices.columns = tickers
    return prices


def perf_stats(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Standard backtest statistics from a daily simple-return series."""
    returns = returns.dropna()
    if returns.empty:
        return {}
    equity = (1 + returns).cumprod()
    n = len(returns)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (periods_per_year / n) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1
    return {
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd.min()), 4),
        "days": n,
    }


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
