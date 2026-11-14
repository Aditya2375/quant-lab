"""Export study 03: pinned JSON for the site + charts."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from vol_forecaster import (load_returns, walk_forward, summarize,
                            garch_fit, garch_variances)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "output")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

BG, FG, GRID = "#0b0e14", "#e6e9ef", "#232a36"
ACC, ACC2, WARN, PURP = "#5ea1ff", "#8be0a4", "#ff7a7a", "#c9a0ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.5,
})

dates, r = load_returns()
print("running walk-forward...")
wf = walk_forward(dates, r)
s = summarize(wf)

# --- chart 1: the stylized fact - vol clusters ----------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
n = len(r)
step = 1
x = dates
ax.plot(x, np.abs(r) * 100, color=ACC, lw=0.4, alpha=0.8)
roll21 = np.sqrt(np.convolve(r * r, np.ones(21) / 21, "same")) * 100 * np.sqrt(252)
ax.plot(x, roll21, color=WARN, lw=1.2,
        label="21d realized vol, annualized")
ax.set_yscale("log")
ticks = list(range(0, n, 1260))
ax.set_xticks(ticks)
ax.set_xticklabels([dates[i][:4] for i in ticks])
ax.set_ylabel("|daily return| %, log scale")
ax.set_title("The stylized fact: volatility arrives in clusters")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "clusters.png"), dpi=140)
plt.close(fig)

# --- chart 2: crisis window, forecast vs realized --------------------------
y = np.asarray(wf["realized"])
g = np.asarray(wf["garch"])
m = np.asarray(wf["ml"])
wd = wf["dates"]
i0 = next(i for i, d in enumerate(wd) if d >= "2020-01-01")
i1 = next(i for i, d in enumerate(wd) if d >= "2020-12-31")
fig, ax = plt.subplots(figsize=(10, 4.2))
seg = slice(i0, i1)
xi = list(range(i0, i1))
ax.plot(xi, np.sqrt(y[seg] * 252) * 100, color=FG, lw=0.7, alpha=0.55,
        label="realized (noisy proxy)")
ax.plot(xi, np.sqrt(g[seg] * 252) * 100, color=ACC2, lw=1.6,
        label="GARCH(1,1) forecast")
ax.plot(xi, np.sqrt(m[seg] * 252) * 100, color=PURP, lw=1.2,
        label="gradient boosting forecast")
ticks = list(range(i0, i1, 42))
ax.set_xticks(ticks)
ax.set_xticklabels([wd[i][:7] for i in ticks])
ax.set_ylabel("annualized vol %")
ax.set_title("2020: both models chase the spike; GARCH sits closer to it")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "crisis2020.png"), dpi=140)
plt.close(fig)

# --- chart 3: rolling 1y QLIKE advantage -----------------------------------
roll = s["rolling_qlike_diff"]
fig, ax = plt.subplots(figsize=(10, 3.6))
rx = [d for d, v in roll]
rv = [v for d, v in roll]
ax.plot(rx, rv, color=ACC, lw=1.3)
ax.axhline(0, color=WARN, lw=1, ls="--")
ax.fill_between(rx, rv, 0, where=[v < 0 for v in rv], color=ACC2, alpha=0.25)
ax.fill_between(rx, rv, 0, where=[v >= 0 for v in rv], color=WARN, alpha=0.25)
ticks = list(range(0, len(rx), 504))
ax.set_xticks(ticks)
ax.set_xticklabels([rx[i][:4] for i in ticks])
ax.set_ylabel("1y mean QLIKE diff")
ax.set_title("GARCH minus ML, rolling one year: below zero = GARCH ahead")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rolling_diff.png"), dpi=140)
plt.close(fig)

# --- chart 4: fitted params across the 13 refits ---------------------------
fig, ax = plt.subplots(figsize=(10, 3.6))
fx = [rf["block_start"] for rf in wf["refits"]]
ax.plot(fx, [rf["garch"]["alpha"] for rf in wf["refits"]], color=WARN,
        marker="o", ms=3, label="alpha (shock weight)")
ax.plot(fx, [rf["garch"]["beta"] for rf in wf["refits"]], color=ACC,
        marker="o", ms=3, label="beta (memory)")
ax.plot(fx, [rf["garch"]["persistence"] for rf in wf["refits"]], color=ACC2,
        marker="o", ms=3, label="alpha + beta (persistence)")
ticks = list(range(0, len(fx), 2))
ax.set_xticks(ticks)
ax.set_xticklabels([fx[i][:4] for i in ticks])
ax.set_ylabel("parameter value")
ax.set_title("GARCH refits are boringly stable - that is a feature")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "params.png"), dpi=140)
plt.close(fig)

# --- pinned JSON ------------------------------------------------------------
def slim(v):
    return {"qlike": round(v["qlike"], 5), "mse": v["mse"],
            "corr": round(v["corr"], 3)}

i2 = next(i for i, d in enumerate(wd) if d >= "2020-01-01")
crisis = slice(i0, i1, 5)
payload = {
    "generated_from": "data/raw/SPY.csv (pinned, sha256 in data/manifest.json)",
    "method": {
        "train_days": 5040, "refit_every_days": 252,
        "ml": "HistGradientBoostingRegressor(max_iter=300, depth=6, lr=0.06, l2=1.0), Duan smearing correction",
        "features": ["r2 lags 1/2/3/5/10/21", "rolling mean r2 5/10/21/63d",
                      "max r2 21d", "|r|"],
        "loss": "QLIKE (Patton 2011); MSE and log-corr as secondary",
        "target": "next-day variance, proxy r^2",
    },
    "window": s["window"],
    "n_forecast_days": s["n_forecast_days"],
    "scoreboard": {k: slim(s[k]) for k in ("garch", "ml", "rv21", "naive")},
    "dm": s["dm"],
    "last_fit": s["last_fit"]["garch"],
    "refits": [{"date": rf["block_start"],
                "alpha": round(rf["garch"]["alpha"], 4),
                "beta": round(rf["garch"]["beta"], 4),
                "persistence": round(rf["garch"]["persistence"], 4)}
               for rf in wf["refits"]],
    "rolling_qlike_diff": [[d, round(v, 4)] for d, v in roll[::5]],
    "crisis2020": {
        "dates": wd[crisis],
        "realized": [round(float(np.sqrt(v * 252) * 100), 2)
                     for v in y[crisis]],
        "garch": [round(float(np.sqrt(v * 252) * 100), 2) for v in g[crisis]],
        "ml": [round(float(np.sqrt(v * 252) * 100), 2) for v in m[crisis]],
    },
}
with open(os.path.join(DOCS, "vol-study.json"), "w") as fh:
    json.dump(payload, fh)
print("written. scoreboard:")
for k in ("garch", "ml", "rv21", "naive"):
    print(f"  {k:6s} QLIKE={s[k]['qlike']:.5f} MSE={s[k]['mse']:.3e} corr={s[k]['corr']:.3f}")
print(f"  DM(garch-ml)={s['dm']['stat']:.2f} p~{s['dm']['p_approx']:.4f}")
