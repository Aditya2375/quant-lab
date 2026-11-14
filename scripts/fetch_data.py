"""Fetch and pin market data for quant-lab studies.

Every CSV in data/raw is committed to the repo with a sha256 in
data/manifest.json so every published number is reproducible forever.
Source: yfinance (Yahoo Finance daily bars, auto-adjusted). Free, no key.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# (ticker, period, need_ohlc)
TARGETS = [("SPY", "max", True)]
PAIRS_UNIVERSE = ["V","MA","KO","PEP","XOM","CVX","JPM","BAC","HD","LOW",
                  "MS","GS","UPS","FDX","NEE","SO","PG","KMB","LIN","APD"]
TARGETS += [(t, "10y", True) for t in PAIRS_UNIVERSE]

manifest = {"source": "yfinance (Yahoo Finance daily bars, auto_adjust=True)",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "series": {}}
for ticker, period, ohlc in TARGETS:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                     progress=False, timeout=30)
    if df.empty:
        print(f"FAILED {ticker}: no data"); sys.exit(1)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index.name = "Date"
    out = RAW / f"{ticker}.csv"
    df.to_csv(out, float_format="%.6f")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest["series"][ticker] = {
        "rows": len(df), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
        "period_requested": period, "sha256": digest, "file": f"data/raw/{ticker}.csv"}
    print(f"{ticker:5s} {len(df):5d} rows  {df.index[0].date()} -> {df.index[-1].date()}  {digest[:12]}")

(ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("manifest written")

# To also pin the option chain used by the options study:
#   python3 -c "import yfinance as yf; yf.Ticker('SPY').option_chain('2026-12-11').calls.to_csv(
#       'data/raw/SPY-options-2026-12-11-calls.csv', index=False)"
# then add its sha256 to data/manifest.json (see commit history for the exact entry).
