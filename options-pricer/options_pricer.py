"""Black-Scholes closed form vs Monte Carlo, with variance reduction and QMC.

European options under GBM with continuous dividend yield. Three estimators:
plain Monte Carlo, antithetic + control-variate MC, and Sobol quasi-MC.
Every published number is computed from a fixed seed and written to
docs/data/options-pricer.json; unit tests cross-check everything against
the closed form.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy.stats import norm, qmc

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

# ---------------------------------------------------------------- closed form

def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price for a European option."""
    if S <= 0 or K <= 0: raise ValueError("S and K must be positive")
    if T <= 0: raise ValueError("T must be positive")
    if sigma < 0: raise ValueError("sigma must be non-negative")
    if kind not in ("call", "put"): raise ValueError("kind must be call or put")
    if sigma == 0:
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(fwd, 0.0) if kind == "call" else max(-fwd, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic Greeks. Vega/rho per 1.00 move; theta per calendar day."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = norm.pdf(d1)
    gamma = math.exp(-q * T) * pdf / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * pdf * math.sqrt(T)
    if kind == "call":
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)
                 + q * S * math.exp(-q * T) * norm.cdf(d1))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * math.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta_per_day": theta / 365.0, "rho": rho}

def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=200):
    """Invert Black-Scholes by bisection on sigma in [1e-6, 5]."""
    lo, hi = 1e-6, 5.0
    p_lo, p_hi = bs_price(S, K, T, r, lo, q, kind), bs_price(S, K, T, r, hi, q, kind)
    if not (p_lo - tol <= price <= p_hi + tol):
        raise ValueError("price outside attainable range (arbitrage or bad input)")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, q, kind) > price: hi = mid
        else: lo = mid
        if hi - lo < tol: break
    return 0.5 * (lo + hi)

# ------------------------------------------------------------- monte carlo

def _terminal_prices(S, T, r, sigma, q, z):
    return S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * z)

def mc_price(S, K, T, r, sigma, q=0.0, kind="call", n=100_000, seed=7,
             antithetic=False, control=False):
    """Monte Carlo estimate with standard error.

    antithetic: pair each z with -z (halves variance for monotone payoffs).
    control: discounted S_T as control variate with exact mean S0*e^{-qT};
    coefficient estimated on the same sample (standard practice).
    Returns dict(price, se, ci95_halfwidth, n).
    """
    if n < 2: raise ValueError("need at least 2 paths")
    rng = np.random.default_rng(seed)
    m = n // 2 if antithetic else n
    z = rng.standard_normal(m)
    if antithetic:
        z = np.concatenate([z, -z])
    st = _terminal_prices(S, T, r, sigma, q, z)
    disc = math.exp(-r * T)
    pay = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
    y = disc * pay
    if control:
        x = disc * st
        x_mean = S * math.exp(-q * T)
        beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        y = y - beta * (x - x_mean)
    est = float(np.mean(y))
    se = float(np.std(y, ddof=1) / math.sqrt(len(y)))
    return {"price": est, "se": se, "ci95": 1.959963984540054 * se, "n": len(y)}

def qmc_price(S, K, T, r, sigma, q=0.0, kind="call", n=100_000, seed=7,
              scramble=True):
    """Sobol quasi-Monte Carlo estimate. n rounded up to a power of two.

    Scrambled Sobol with 16 independent scrambles gives an error estimate.
    """
    m = 2 ** math.ceil(math.log2(max(n, 16)))
    disc = math.exp(-r * T)
    estimates = []
    rng = np.random.default_rng(seed)
    for _ in range(16):
        eng = qmc.Sobol(d=1, scramble=scramble, seed=rng)
        u = eng.random_base2(int(math.log2(m))).ravel()
        u = np.clip(u, 1e-12, 1 - 1e-12)
        z = norm.ppf(u)
        st = _terminal_prices(S, T, r, sigma, q, z)
        pay = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
        estimates.append(float(disc * np.mean(pay)))
    est = float(np.mean(estimates))
    se = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
    return {"price": est, "se": se, "ci95": 2.131 * se, "n": m}  # t_15,95%

# ------------------------------------------------------- convergence study

def convergence_study(S, K, T, r, sigma, q=0.0, kind="call", seed=7,
                      powers=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)):
    """Absolute error of each estimator vs the closed form as N grows."""
    exact = bs_price(S, K, T, r, sigma, q, kind)
    rows = []
    for p in powers:
        n = 2 ** p
        base = seed * 1_000_003 + p
        plain = mc_price(S, K, T, r, sigma, q, kind, n, base)
        avcv = mc_price(S, K, T, r, sigma, q, kind, n, base,
                        antithetic=True, control=True)
        sob = qmc_price(S, K, T, r, sigma, q, kind, n, base)
        rows.append({
            "n": n,
            "plain": {"price": plain["price"], "se": plain["se"],
                      "abs_err": abs(plain["price"] - exact)},
            "antithetic_control": {"price": avcv["price"], "se": avcv["se"],
                                   "abs_err": abs(avcv["price"] - exact)},
            "sobol": {"price": sob["price"], "se": sob["se"],
                      "abs_err": abs(sob["price"] - exact)},
        })
    return {"exact": exact, "rows": rows}

# ------------------------------------------------------------------- main

def spy_anchor():
    """Realistic demo parameters estimated from the pinned SPY series."""
    import pandas as pd
    spy = pd.read_csv(ROOT / "data" / "raw" / "SPY.csv", index_col=0, parse_dates=True)
    close = spy["Close"].dropna()
    logret = np.log(close / close.shift(1)).dropna()
    recent = logret.iloc[-21:]
    sigma = float(recent.std() * math.sqrt(TRADING_DAYS))
    s0 = float(close.iloc[-1])
    as_of = str(close.index[-1].date())
    return s0, sigma, as_of

def run():
    s0, sigma_real, as_of = spy_anchor()
    # Demo contract: ~1-month ATM SPY call. Rate is an assumption, not data.
    S, K = s0, round(s0 / 5) * 5.0
    T, r, q = 30 / 365, 0.04, 0.012
    sigma = sigma_real
    params = {"S": S, "K": K, "T": T, "T_days": 30, "r": r, "q": q,
              "sigma": sigma, "sigma_source":
              f"SPY 21-day realized close-to-close vol, annualized, as of {as_of}",
              "kind": "call", "as_of": as_of}

    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    parity_lhs = call - put
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    greeks = bs_greeks(S, K, T, r, sigma, q, "call")

    conv = convergence_study(S, K, T, r, sigma, q, "call")
    # headline run: 1M paths (2^20), matching the reel's "100k paths" claim x10
    head = mc_price(S, K, T, r, sigma, q, "call", n=2**20, seed=1234567,
                    antithetic=True, control=True)

    iv_recovered = implied_vol(call, S, K, T, r, q, "call")

    out = {"params": params, "closed_form": {
               "call": call, "put": put,
               "put_call_parity_residual": abs(parity_lhs - parity_rhs),
               "greeks": greeks},
           "headline_mc": {**head, "abs_err": abs(head["price"] - call)},
           "convergence": conv,
           "implied_vol_roundtrip": {"recovered": iv_recovered,
                                     "abs_err": abs(iv_recovered - sigma)}}
    (ROOT / "docs" / "data").mkdir(exist_ok=True)
    dest = ROOT / "docs" / "data" / "options-pricer.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"bs_call": call, "mc_1m": head["price"],
                      "mc_err": out["headline_mc"]["abs_err"],
                      "parity_resid": out["closed_form"]["put_call_parity_residual"],
                      "iv_err": out["implied_vol_roundtrip"]["abs_err"]}, indent=2))
    # small provenance copy for the study folder
    (Path(__file__).parent / "output" / "summary.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    run()

# ------------------------------------------------------- market cross-check

def market_crosscheck(chain_csv, S, r, q, expiry, as_of):
    """Implied-vol smile from a pinned real SPY option chain.

    Inverts our own Black-Scholes on each strike's mid price and compares
    with the provider's impliedVolatility field, then prices every strike
    at one flat sigma to show what the smile does to flat-model prices.
    """
    import pandas as pd
    from datetime import date
    df = __import__("pandas").read_csv(chain_csv)
    T = (date.fromisoformat(expiry) - date.fromisoformat(as_of)).days / 365.0
    points = []
    for _, row in df.iterrows():
        K_, bid, ask = float(row["strike"]), float(row["bid"]), float(row["ask"])
        mid = 0.5 * (bid + ask)
        y_iv = float(row["impliedVolatility"])
        try:
            my_iv = implied_vol(mid, S, K_, T, r, q, "call")
        except ValueError:
            continue
        points.append({"strike": K_, "mid": round(mid, 2), "bid": bid, "ask": ask,
                       "my_iv": round(my_iv, 5), "yahoo_iv": round(y_iv, 5),
                       "open_interest": int(row["openInterest"]),
                       "moneyness": round(K_ / S, 4)})
    points.sort(key=lambda p: p["strike"])
    atm = min(points, key=lambda p: abs(p["strike"] - S))
    flat = atm["my_iv"]
    for p in points:
        p["flat_price"] = round(bs_price(S, p["strike"], T, r, flat, q, "call"), 2)
        p["flat_minus_mid"] = round(p["flat_price"] - p["mid"], 2)
    iv_diffs = [abs(p["my_iv"] - p["yahoo_iv"]) for p in points]
    return {"expiry": expiry, "T": round(T, 5), "n_strikes": len(points),
            "atm_iv": flat, "max_abs_iv_diff_vs_provider": round(max(iv_diffs), 5),
            "points": points}
