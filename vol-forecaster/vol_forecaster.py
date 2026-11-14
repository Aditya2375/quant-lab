"""Volatility forecasting: GARCH(1,1) vs machine learning, same data, honest scoring.

GARCH(1,1) is fitted by exact Gaussian maximum likelihood (own recursion +
scipy optimizer, no arch package). The ML side is gradient boosting on
lagged-volatility features. Both are scored out-of-sample in a walk-forward
loop against the realized variance proxy, with QLIKE (the robust loss for
volatility comparison, Patton 2011) and a Diebold-Mariano test on the loss
differential. Nothing is tuned on the test window: ML hyperparameters are
fixed a priori and both models refit only on past data at each step.
"""
import csv
import math
import os

import numpy as np
from scipy.optimize import minimize

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "SPY.csv")
TRAIN_DAYS = 5040            # ~20 trading years initial fit window
REFIT_EVERY = 252            # refit once a year in the walk-forward loop


def load_returns(path=DATA):
    dates, closes = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            dates.append(row["Date"])
            closes.append(float(row["Close"]))
    c = np.asarray(closes)
    r = np.diff(np.log(c))
    return dates[1:], r


# ---------------------------------------------------------------- GARCH(1,1)

def garch_variances(params, r):
    """Variance recursion: s2_t = w + a r_{t-1}^2 + b s2_{t-1}."""
    w, a, b = params
    n = len(r)
    s2 = np.empty(n)
    s2[0] = np.var(r)
    r2 = r * r
    for t in range(1, n):
        s2[t] = w + a * r2[t - 1] + b * s2[t - 1]
    return s2


def garch_negloglik(params, r):
    w, a, b = params
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    s2 = garch_variances(params, r)
    if np.any(s2 <= 0):
        return 1e10
    return float(0.5 * np.sum(np.log(2 * np.pi * s2) + r * r / s2))


def garch_fit(r, start=None):
    """MLE fit. Stationary variance anchor for the start; L-BFGS-B polish."""
    v0 = np.var(r)
    if start is None:
        start = np.array([v0 * 0.05, 0.08, 0.87])
    best = None
    for s0 in (start, np.array([v0 * 0.02, 0.05, 0.93]),
               np.array([v0 * 0.1, 0.12, 0.80])):
        res = minimize(garch_negloglik, s0, args=(r,), method="L-BFGS-B",
                       bounds=[(1e-12, None), (1e-8, 0.9999), (1e-8, 0.9999)])
        if best is None or res.fun < best.fun:
            best = res
    w, a, b = best.x
    return {"omega": float(w), "alpha": float(a), "beta": float(b),
            "persistence": float(a + b),
            "uncond_var": float(w / (1 - a - b)),
            "loglik": float(-best.fun)}


# ------------------------------------------------------------------ features

def build_features(r, idx):
    """Features for the ML model at day t (forecasting t+1 variance).

    All features are measurable at the close of day t. idx selects days.
    """
    r2 = r * r
    cols = []
    for lag in (1, 2, 3, 5, 10, 21):
        cols.append(np.array([r2[t - lag + 1] for t in idx]))
    for win in (5, 10, 21, 63):
        cols.append(np.array([np.mean(r2[t - win + 1:t + 1]) for t in idx]))
    # Parkinson range vol would need OHLC alignment; log-price range over
    # 21 days is a cheap long-memory feature instead.
    cols.append(np.array([np.max(r2[t - 20:t + 1]) for t in idx]))
    cols.append(np.array([abs(r[t]) for t in idx]))
    X = np.column_stack(cols)
    return np.log(X + 1e-12)


FEATURE_NAMES = ["r2_lag1", "r2_lag2", "r2_lag3", "r2_lag5", "r2_lag10",
                 "r2_lag21", "rv_5d", "rv_10d", "rv_21d", "rv_63d",
                 "max_r2_21d", "abs_r"]


# -------------------------------------------------------------------- losses

def qlike(realized, forecast):
    """QLIKE = log(f) + y/f. Robust to noisy realized proxies (Patton 2011)."""
    return float(np.mean(np.log(forecast) + realized / forecast))


def mse(realized, forecast):
    return float(np.mean((realized - forecast) ** 2))


def dm_test(loss_a, loss_b, h=1):
    """Diebold-Mariano on the loss differential d = a - b.

    Returns (dm_stat, approx p, mean_d). Positive stat: a loses more than b.
    Newey-West style long-run variance with h-1 lags.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    md = np.mean(d)
    gamma0 = np.mean((d - md) ** 2)
    lr = gamma0
    for k in range(1, h + 1):
        w = 1 - k / (h + 1)
        cov = np.mean((d[k:] - md) * (d[:-k] - md))
        lr += 2 * w * cov
    if lr <= 0:
        return 0.0, 1.0, float(md)  # identical loss series
    stat = md / math.sqrt(lr / n)
    # normal approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2))))
    return float(stat), float(p), float(md)


# ------------------------------------------------------------- walk-forward

def walk_forward(dates, r, train_days=TRAIN_DAYS, refit_every=REFIT_EVERY):
    """Expanding-window walk-forward from train_days to the end.

    At each refit point both models see only data strictly before the
    forecast block. Forecasts: one-day-ahead variance for every day of the
    block (GARCH: recursive filter through the block without refitting;
    ML: features from realized returns only).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    n = len(r)
    out = {"dates": [], "realized": [], "garch": [], "ml": [],
           "naive": [], "rv21": [], "refits": []}
    t0 = train_days
    while t0 < n - 1:
        t1 = min(t0 + refit_every, n - 1)
        idx_all = np.arange(0, t0)
        r_train = r[idx_all]

        gp = garch_fit(r_train)
        s2 = garch_variances((gp["omega"], gp["alpha"], gp["beta"]), r_train)
        # roll the filter through the block on realized returns, forecasting
        # day t+1 from info through day t - same horizon as the ML model
        g_fore = []
        last_s2 = s2[-1]
        for t in range(t0, t1):
            last_s2 = (gp["omega"] + gp["alpha"] * r[t - 1] ** 2
                       + gp["beta"] * last_s2)           # variance of day t
            g_fore.append(gp["omega"] + gp["alpha"] * r[t] ** 2
                          + gp["beta"] * last_s2)          # forecast for t+1

        # target y at feature day t is r[t+1]^2; cap t at t0-2 so no test
        # block return ever enters a training target
        feat_idx = np.arange(63, t0 - 1)
        X_train = build_features(r, feat_idx)
        y_train = np.log(r[feat_idx + 1] ** 2 + 1e-12)
        ml = HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                           learning_rate=0.06,
                                           l2_regularization=1.0,
                                           random_state=42)
        ml.fit(X_train, y_train)
        # Duan smearing: the model is trained on log(variance), so exp(pred)
        # estimates the median, not the mean. Rescale by mean(exp(resid))
        # on the training window - without this the comparison tilts against
        # ML for a trivial reason.
        smear = float(np.mean(np.exp(y_train - ml.predict(X_train))))
        test_idx = np.arange(t0, t1)
        X_test = build_features(r, test_idx)
        m_fore = np.exp(ml.predict(X_test)) * smear - 1e-12

        out["dates"].extend(dates[t] for t in test_idx)
        out["realized"].extend(float(r[t + 1] ** 2) for t in test_idx)
        out["garch"].extend(float(v) for v in g_fore)
        out["ml"].extend(float(v) for v in m_fore)
        # baselines: tomorrow's variance = today's squared return (naive),
        # or the trailing 21-day realized variance (rv21)
        out["naive"].extend(float(max(r[t] ** 2, 1e-8)) for t in test_idx)
        out["rv21"].extend(float(np.mean(r[t - 20:t + 1] ** 2))
                           for t in test_idx)
        out["refits"].append({
            "fit_until": dates[t0 - 1], "block_start": dates[t0],
            "garch": gp,
        })
        t0 = t1
    return out


def summarize(wf):
    y = np.asarray(wf["realized"])
    g = np.asarray(wf["garch"])
    m = np.asarray(wf["ml"])
    res = {}
    na = np.asarray(wf["naive"])
    rv = np.asarray(wf["rv21"])
    for name, f in (("garch", g), ("ml", m), ("naive", na), ("rv21", rv)):
        res[name] = {"qlike": qlike(y, f), "mse": mse(y, f),
                     "corr": float(np.corrcoef(np.log(y + 1e-12),
                                               np.log(f + 1e-12))[0, 1])}
    # per-day loss differential series for the DM test
    dg = np.log(g) + y / g
    dm = np.log(m) + y / m
    stat, p, md = dm_test(dg, dm, h=10)
    res["dm"] = {"stat": stat, "p_approx": p,
                 "mean_qlike_diff_g_minus_ml": md}
    # rolling 1y mean QLIKE difference (GARCH - ML): negative = GARCH ahead
    win = 252
    diff = dg - dm
    roll = []
    for i in range(win, len(diff)):
        roll.append((wf["dates"][i], float(np.mean(diff[i - win:i]))))
    res["rolling_qlike_diff"] = roll
    res["n_forecast_days"] = len(y)
    res["window"] = [wf["dates"][0], wf["dates"][-1]]
    res["last_fit"] = wf["refits"][-1]
    return res


if __name__ == "__main__":
    dates, r = load_returns()
    print(f"{len(r)} daily returns, {dates[0]} .. {dates[-1]}")
    wf = walk_forward(dates, r)
    s = summarize(wf)
    for k in ("garch", "ml"):
        print(f"{k:6s} QLIKE={s[k]['qlike']:.5f} MSE={s[k]['mse']:.3e} "
              f"corr(log)={s[k]['corr']:.3f}")
    print(f"DM stat={s['dm']['stat']:.2f} p~{s['dm']['p_approx']:.3f} "
          f"(positive: ML better) over {s['n_forecast_days']} days")
    lf = s["last_fit"]["garch"]
    print(f"last GARCH fit: alpha={lf['alpha']:.4f} beta={lf['beta']:.4f} "
          f"persistence={lf['persistence']:.4f}")
