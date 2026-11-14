"""Export study 02: pinned JSON for the site + charts."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pairs_backtester import (PAIRS, run_pair, sensitivity, load_close,
                              align, ols, FORMATION_DAYS)
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "output")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

BG, FG, GRID = "#0b0e14", "#e6e9ef", "#232a36"
ACC, ACC2, WARN = "#5ea1ff", "#8be0a4", "#ff7a7a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.5,
})

results = {f"{a}/{b}": run_pair(a, b) for a, b in PAIRS}
order = sorted(results, key=lambda k: results[k]["adf_t"])

# --- chart 1: in-sample vs out-of-sample sharpe, every pair ---------------
fig, ax = plt.subplots(figsize=(10, 4.6))
x = range(len(order))
ax.bar([i - 0.2 for i in x], [results[k]["in_sample"]["sharpe"] for k in order],
       width=0.4, color=ACC, label="in-sample (2016-2021)")
ax.bar([i + 0.2 for i in x], [results[k]["sharpe"] for k in order],
       width=0.4, color=ACC2, label="out-of-sample (2021-2026)")
ax.set_xticks(list(x))
ax.set_xticklabels(order, rotation=45, ha="right")
ax.axhline(0, color=FG, lw=0.8)
ax.set_ylabel("Sharpe ratio")
ax.set_title("The strategy works where it was fitted. Then it mostly stops.")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "in_vs_out.png"), dpi=140)
plt.close(fig)

# --- chart 2: HD/LOW detail - z-score bands + equity ----------------------
hd = results["HD/LOW"]
dates = hd["dates"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                               height_ratios=[1.2, 1])
step = 5
xi = list(range(0, len(dates), step))
xd = xi
ax1.plot(xd, hd["z"][::step], color=ACC, lw=1.1, label="spread z-score")
ax1.axhline(2.0, color=WARN, ls="--", lw=1, label="entry |z| = 2")
ax1.axhline(-2.0, color=WARN, ls="--", lw=1)
ax1.axhline(0.5, color=FG, ls=":", lw=0.9, label="exit |z| = 0.5")
ax1.axhline(-0.5, color=FG, ls=":", lw=0.9)
ax1.set_ylabel("z-score of log spread")
ax1.set_title("HD/LOW out-of-sample: frozen 2021 parameters, never refit")
ax1.legend(frameon=False, loc="upper right")
n = len(dates)
tick_idx = list(range(0, n, max(1, n // 6)))
ax2.plot(xd, hd["equity"][::step], color=ACC2, lw=1.4,
         label="with costs (5 bps/side/leg)")
ax2.plot(xd, hd["equity_nocost"][::step], color=ACC, lw=1.1, ls="--",
         label="frictionless")
ax2.set_ylabel("equity (start = 1.0)")
ax2.legend(frameon=False)
for a in (ax1, ax2):
    a.set_xticks([i for i in tick_idx])
    a.set_xticklabels([dates[i][:7] for i in tick_idx])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hd_low_detail.png"), dpi=140)
plt.close(fig)

# --- pinned JSON for the site ----------------------------------------------
def slim(r):
    return {k: r[k] for k in (
        "pair", "formation", "trading", "hedge_ratio", "adf_t",
        "adf_p_bound", "half_life_days", "n_trades", "hit_rate",
        "total_return", "total_return_nocost", "sharpe", "sharpe_nocost",
        "max_drawdown", "in_sample")}

payload = {
    "generated_from": "pinned csvs, see data/manifest.json",
    "params": {"formation_days": FORMATION_DAYS, "entry_z": 2.0,
               "exit_z": 0.5, "cost_bps_per_side_leg": 5.0},
    "pairs": [slim(results[k]) for k in order],
    "detail": {
        "pair": "HD/LOW",
        "dates": dates[::step],
        "z": [round(v, 4) for v in hd["z"][::step]],
        "equity": [round(v, 5) for v in hd["equity"][::step]],
        "equity_nocost": [round(v, 5) for v in hd["equity_nocost"][::step]],
        "sensitivity": sensitivity("HD", "LOW"),
    },
}
with open(os.path.join(DOCS, "pairs-study.json"), "w") as fh:
    json.dump(payload, fh)
print("charts + json written; pairs order:", order)
