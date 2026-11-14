"""Pairs trading, backtested honestly.

Formation period: fit the hedge ratio by OLS on log prices, test the
residual spread for a unit root (ADF) and estimate its OU half-life.
Trading period: out-of-sample, trade z-score crossings with real costs.

Every number in the study comes out of run_study() on the pinned CSVs.
No lookahead: the hedge ratio and z-score bands are frozen at the end of
the formation window and never refit on trading-period data.
"""
import csv
import json
import math
import os

PAIRS = [
    ("V", "MA"), ("KO", "PEP"), ("XOM", "CVX"), ("JPM", "BAC"),
    ("HD", "LOW"), ("MS", "GS"), ("UPS", "FDX"), ("NEE", "SO"),
    ("PG", "KMB"), ("LIN", "APD"),
]

FORMATION_DAYS = 1260          # ~5 trading years to fit on
ENTRY_Z = 2.0
EXIT_Z = 0.5
COST_BPS = 5.0                 # per side, per leg: commission + half-spread
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_close(ticker):
    out = {}
    with open(os.path.join(DATA, f"{ticker}.csv")) as fh:
        for row in csv.DictReader(fh):
            out[row["Date"]] = float(row["Close"])
    return out


def align(a, b):
    dates = sorted(set(a) & set(b))
    return dates, [a[d] for d in dates], [b[d] for d in dates]


def ols(x, y):
    """y = alpha + beta*x, closed form."""
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n
    return alpha, beta


def adf_test(series, max_lag=10):
    """ADF with constant, lag chosen by AIC. Returns (t_stat, p_bound).

    p bound from MacKinnon (1994) critical values for the constant-only
    case: 1% -3.43, 5% -2.86, 10% -2.57.
    """
    y = series
    dy = [y[i] - y[i - 1] for i in range(1, len(y))]
    best = None
    for lag in range(0, max_lag + 1):
        # common sample across lag candidates (standard AIC lag selection):
        # drop the first max_lag points regardless of the candidate lag
        Y = dy[max_lag:]
        X = []
        for t in range(max_lag, len(dy)):
            row = [1.0, y[t]]
            row += [dy[t - i] for i in range(1, lag + 1)]
            X.append(row)
        n = len(Y)
        beta = solve_ls(X, Y)
        resid = [yt - sum(b * x for b, x in zip(beta, xr))
                 for yt, xr in zip(Y, X)]
        sigma2 = sum(r * r for r in resid) / n
        k = lag + 2
        aic = n * math.log(sigma2) + 2 * k
        if best is None or aic < best[0]:
            best = (aic, beta[1], X, resid, n)
    _, b, X, resid, n = best
    # SE from the (X'X)^-1 diagonal - matters once lagged terms are in X
    k = len(X[0])
    ata = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    inv = mat_inverse(ata)
    sigma2 = sum(r * r for r in resid) / (n - k)
    se = math.sqrt(sigma2 * inv[1][1])
    t_stat = b / se
    if t_stat < -3.43:
        p = 0.01
    elif t_stat < -2.86:
        p = 0.05
    elif t_stat < -2.57:
        p = 0.10
    else:
        p = 1.0
    return t_stat, p


def solve_ls(X, y):
    """Least squares via normal equations + Gaussian elimination."""
    k = len(X[0])
    ata = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(ata[r][c]))
        ata[c], ata[piv] = ata[piv], ata[c]
        aty[c], aty[piv] = aty[piv], aty[c]
        for r in range(c + 1, k):
            f = ata[r][c] / ata[c][c]
            for j in range(c, k):
                ata[r][j] -= f * ata[c][j]
            aty[r] -= f * aty[c]
    out = [0.0] * k
    for c in range(k - 1, -1, -1):
        out[c] = (aty[c] - sum(ata[c][j] * out[j]
                  for j in range(c + 1, k))) / ata[c][c]
    return out


def mat_inverse(m):
    """Gauss-Jordan inverse of a small square matrix."""
    k = len(m)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
           for i, row in enumerate(m)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        aug[c], aug[piv] = aug[piv], aug[c]
        d = aug[c][c]
        aug[c] = [v / d for v in aug[c]]
        for r in range(k):
            if r != c and aug[r][c] != 0.0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    return [row[k:] for row in aug]


def half_life(spread):
    """OU half-life from AR(1) on the spread."""
    y = spread[1:]
    x = spread[:-1]
    alpha, beta = ols(x, y)
    if beta <= 0 or beta >= 1:
        return float("inf")
    return -math.log(2) / math.log(beta)


def backtest(dates, px_a, px_b, beta, mu, sigma,
             entry=ENTRY_Z, exit_z=EXIT_Z, cost_bps=COST_BPS):
    """Dollar-neutral z-score strategy, frozen parameters.

    Position +1 = long A / short beta*B, -1 = the reverse, 0 = flat.
    Signals from day i-1's close are acted on at day i's close (no
    same-bar fill). Costs charged at every position change, and charged
    twice when flipping direction (exit + entry).
    """
    pos = 0
    entry_px = None
    equity = [1.0]
    eq_nocost = [1.0]
    trades = []
    z_series = []
    for i in range(len(dates)):
        z = (math.log(px_a[i]) - beta * math.log(px_b[i]) - mu) / sigma
        z_series.append(z)
        if i == 0:
            continue
        ret_a = px_a[i] / px_a[i - 1] - 1.0
        ret_b = px_b[i] / px_b[i - 1] - 1.0
        spread_ret = (ret_a - beta * ret_b) / (1.0 + abs(beta))
        eq_nocost.append(eq_nocost[-1] * (1.0 + pos * spread_ret))
        prev_z = z_series[i - 1]
        new_pos = pos
        if pos == 0 and abs(prev_z) > entry:
            new_pos = -1 if prev_z > 0 else 1
        elif pos != 0 and abs(prev_z) < exit_z:
            new_pos = 0
        cost = 0.0
        if new_pos != pos:
            flips = 1.0 if (new_pos == 0 or pos == 0) else 2.0
            cost = (cost_bps / 1e4) * flips
            if new_pos != 0:
                entry_px = (px_a[i], px_b[i])
            else:
                pnl = pos * (math.log(px_a[i] / entry_px[0])
                             - beta * math.log(px_b[i] / entry_px[1]))
                trades.append({"date": dates[i], "dir": pos, "log_pnl": pnl})
                entry_px = None
        equity.append(equity[-1] * (1.0 + pos * spread_ret - cost))
        pos = new_pos
    # an unclosed position at the end still counts: mark it at the last close
    if entry_px is not None and pos != 0:
        pnl = pos * (math.log(px_a[-1] / entry_px[0])
                     - beta * math.log(px_b[-1] / entry_px[1]))
        trades.append({"date": dates[-1], "dir": pos, "log_pnl": pnl,
                       "open": True})
    return {
        "equity": equity, "equity_nocost": eq_nocost,
        "trades": trades, "z": z_series,
    }


def max_drawdown(curve):
    peak, mdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return mdd


def sharpe(curve, periods=252):
    rets = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve))]
    if len(rets) < 2:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return 0.0 if sd == 0 else (m / sd) * math.sqrt(periods)


def run_pair(ta, tb, formation=FORMATION_DAYS):
    a, b = load_close(ta), load_close(tb)
    dates, pa, pb = align(a, b)
    form_dates, form_a, form_b = dates[:formation], pa[:formation], pb[:formation]
    test_dates, test_a, test_b = dates[formation:], pa[formation:], pb[formation:]

    la = [math.log(v) for v in form_a]
    lb = [math.log(v) for v in form_b]
    alpha, beta = ols(lb, la)          # log A = alpha + beta log B + spread
    spread = [x - beta * y for x, y in zip(la, lb)]
    t_stat, p = adf_test(spread)
    hl = half_life(spread)
    mu = sum(spread) / len(spread)
    sigma = math.sqrt(sum((s - mu) ** 2 for s in spread) / (len(spread) - 1))

    bt = backtest(test_dates, test_a, test_b, beta, mu, sigma)
    ins = backtest(form_dates, form_a, form_b, beta, mu, sigma)
    wins = sum(1 for t in bt["trades"] if t["log_pnl"] > 0)
    return {
        "in_sample": {
            "total_return": ins["equity"][-1] - 1.0,
            "sharpe": sharpe(ins["equity"]),
            "n_trades": len(ins["trades"]),
        },
        "pair": f"{ta}/{tb}",
        "formation": {"start": form_dates[0], "end": form_dates[-1],
                      "days": len(form_dates)},
        "trading": {"start": test_dates[0], "end": test_dates[-1],
                    "days": len(test_dates)},
        "hedge_ratio": beta,
        "adf_t": t_stat, "adf_p_bound": p,
        "half_life_days": hl,
        "n_trades": len(bt["trades"]),
        "hit_rate": (wins / len(bt["trades"])) if bt["trades"] else None,
        "total_return": bt["equity"][-1] - 1.0,
        "total_return_nocost": bt["equity_nocost"][-1] - 1.0,
        "sharpe": sharpe(bt["equity"]),
        "sharpe_nocost": sharpe(bt["equity_nocost"]),
        "max_drawdown": max_drawdown(bt["equity"]),
        "equity": bt["equity"], "equity_nocost": bt["equity_nocost"],
        "z": bt["z"], "dates": test_dates,
    }


def sensitivity(ta, tb, entries=(1.5, 2.0, 2.5), costs=(0.0, 5.0, 10.0)):
    """Entry-band x cost grid for one pair: is any result robust?"""
    a, b = load_close(ta), load_close(tb)
    dates, pa, pb = align(a, b)
    la = [math.log(v) for v in pa[:FORMATION_DAYS]]
    lb = [math.log(v) for v in pb[:FORMATION_DAYS]]
    alpha, beta = ols(lb, la)
    spread = [x - beta * y for x, y in zip(la, lb)]
    mu = sum(spread) / len(spread)
    sigma = math.sqrt(sum((s - mu) ** 2 for s in spread) / (len(spread) - 1))
    td, ta_, tb_ = dates[FORMATION_DAYS:], pa[FORMATION_DAYS:], pb[FORMATION_DAYS:]
    grid = []
    for e in entries:
        for c in costs:
            bt = backtest(td, ta_, tb_, beta, mu, sigma, entry=e, cost_bps=c)
            grid.append({"entry_z": e, "cost_bps": c,
                         "sharpe": sharpe(bt["equity"]),
                         "total_return": bt["equity"][-1] - 1.0,
                         "n_trades": len(bt["trades"])})
    return grid


def run_study():
    results = [run_pair(a, b) for a, b in PAIRS]
    for r in results:
        hl = r['half_life_days']
        ins = r['in_sample']
        print(f"{r['pair']:9s} adf_t={r['adf_t']:6.2f} "
              f"hl={hl if hl != float('inf') else -1:6.1f} "
              f"in:sharpe={ins['sharpe']:5.2f}({ins['n_trades']:2d}t) "
              f"out:trades={r['n_trades']:3d} sharpe={r['sharpe']:5.2f} "
              f"ret={r['total_return']:6.1%} mdd={r['max_drawdown']:6.1%}")
    return results


if __name__ == "__main__":
    run_study()
