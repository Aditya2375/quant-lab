/* quant-lab numerical engine: Black-Scholes, Monte Carlo (plain, antithetic +
   control variate), scrambled 1-D Sobol QMC. Mirrors options_pricer.py;
   node tests pin agreement with the Python reference. */
"use strict";
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.QLE = factory();
})(typeof self !== "undefined" ? self : this, function () {

  // ---- special functions -------------------------------------------------
  // normCdf: power series near zero, Laplace continued fraction (Lentz) in
  // the tails. No memorized coefficient tables; verified against scipy to <1e-14.
  const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
  function normCdf(x) {
    if (x === 0) return 0.5;
    const ax = Math.abs(x);
    if (ax <= 1.5) {
      // Phi(x) = 0.5 + phi(0) * sum_k (-1)^k x^(2k+1) / ((2k+1) 2^k k!)
      let term = ax, sum = ax, k = 1;
      while (Math.abs(term) > 1e-18) {
        term *= -(ax * ax) * (2 * k - 1) / (2 * k * (2 * k + 1));
        sum += term; k++;
        if (k > 200) break;
      }
      const c = 0.5 + INV_SQRT_2PI * sum;
      return x > 0 ? c : 1 - c;
    }
    // tail Q(ax) = phi(ax) * R(ax), R = 1/(ax + 1/(ax + 2/(ax + 3/...)))
    // Lentz algorithm evaluates f = ax + 1/(ax + 2/...) = 1/R, so Q = phi/f.
    let f = ax + 1e-300, C = f, D = 0;
    for (let n = 1; n <= 500; n++) {
      const a = n, b = ax;
      D = b + a * D; if (D === 0) D = 1e-300; D = 1 / D;
      C = b + a / C; if (C === 0) C = 1e-300;
      const delta = C * D;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    const Q = INV_SQRT_2PI * Math.exp(-0.5 * ax * ax) / f;
    return x > 0 ? 1 - Q : Q;
  }
  function erf(x) { return 2 * normCdf(x * Math.SQRT2) - 1; }
  function erfc(x) { return 2 * normCdf(-x * Math.SQRT2); }
  function normPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  // inverse normal CDF: Acklam's algorithm
  function normPpf(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const plow = 0.02425, phigh = 1 - plow;
    let q, r, x;
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= phigh) {
      q = p - 0.5; r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
          (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    // one Halley refinement step
    const e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
    return x - u / (1 + 0.5 * x * u);
  }

  // ---- PRNG: sfc32 (fast, seedable, good quality) ------------------------
  function sfc32(a, b, c, d) {
    return function () {
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      let t = (a + b | 0) + d | 0;
      d = d + 1 | 0;
      a = b ^ b >>> 9;
      b = c + (c << 3) | 0;
      c = (c << 21 | c >>> 11) + t | 0;
      return (t >>> 0) / 4294967296;
    };
  }
  function gaussianGen(seed) {
    const rand = sfc32(seed ^ 0x9e3779b9, seed * 2654435761 >>> 0, seed * 2246822519 >>> 0, seed * 3266489917 >>> 0);
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0, s = 0;
      do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const mul = Math.sqrt(-2 * Math.log(s) / s);
      spare = v * mul;
      return u * mul;
    };
  }
  // van der Corput radical inverse base 2 with a digital (XOR) shift scramble
  function vdcScrambled(i, shift) {
    let x = i >>> 0, r = 0, bit = 0x80000000;
    while (bit) { if (x & 1) r = (r ^ bit) >>> 0; x >>>= 1; bit >>>= 1; }
    r = (r ^ (shift >>> 0)) >>> 0;
    const u = r / 4294967296;
    return u > 0 ? u : 1e-12;
  }

  // ---- Black-Scholes ------------------------------------------------------
  function bsPrice(S, K, T, r, sigma, q, kind) {
    if (S <= 0 || K <= 0) throw new Error("S and K must be positive");
    if (T <= 0) throw new Error("T must be positive");
    if (sigma < 0) throw new Error("sigma must be non-negative");
    if (sigma === 0) {
      const fwd = S * Math.exp(-q * T) - K * Math.exp(-r * T);
      return kind === "call" ? Math.max(fwd, 0) : Math.max(-fwd, 0);
    }
    const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    return kind === "call"
      ? S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2)
      : K * Math.exp(-r * T) * normCdf(-d2) - S * Math.exp(-q * T) * normCdf(-d1);
  }
  function bsGreeks(S, K, T, r, sigma, q, kind) {
    const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    const pdf = normPdf(d1);
    const gamma = Math.exp(-q * T) * pdf / (S * sigma * Math.sqrt(T));
    const vega = S * Math.exp(-q * T) * pdf * Math.sqrt(T);
    let delta, theta, rho;
    if (kind === "call") {
      delta = Math.exp(-q * T) * normCdf(d1);
      theta = -S * Math.exp(-q * T) * pdf * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCdf(d2) + q * S * Math.exp(-q * T) * normCdf(d1);
      rho = K * T * Math.exp(-r * T) * normCdf(d2);
    } else {
      delta = -Math.exp(-q * T) * normCdf(-d1);
      theta = -S * Math.exp(-q * T) * pdf * sigma / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normCdf(-d2) - q * S * Math.exp(-q * T) * normCdf(-d1);
      rho = -K * T * Math.exp(-r * T) * normCdf(-d2);
    }
    return { delta, gamma, vega, thetaPerDay: theta / 365, rho };
  }
  function impliedVol(price, S, K, T, r, q, kind, tol = 1e-10, maxIter = 200) {
    let lo = 1e-6, hi = 5.0;
    if (price < bsPrice(S, K, T, r, lo, q, kind) - tol || price > bsPrice(S, K, T, r, hi, q, kind) + tol)
      throw new Error("price outside attainable range");
    for (let i = 0; i < maxIter; i++) {
      const mid = 0.5 * (lo + hi);
      if (bsPrice(S, K, T, r, mid, q, kind) > price) hi = mid; else lo = mid;
      if (hi - lo < tol) break;
    }
    return 0.5 * (lo + hi);
  }

  // ---- Monte Carlo ---------------------------------------------------------
  function terminalPrices(S, T, r, sigma, q, z, out) {
    const drift = (r - q - 0.5 * sigma * sigma) * T, vol = sigma * Math.sqrt(T);
    out = out || new Float64Array(z.length);
    for (let i = 0; i < z.length; i++) out[i] = S * Math.exp(drift + vol * z[i]);
    return out;
  }
  function mcPrice(S, K, T, r, sigma, q, kind, n, seed, antithetic, control) {
    if (n < 2) throw new Error("need at least 2 paths");
    const gen = gaussianGen(seed);
    const m = antithetic ? n >> 1 : n;
    let z = new Float64Array(antithetic ? 2 * m : m);
    for (let i = 0; i < m; i++) z[i] = gen();
    if (antithetic) for (let i = 0; i < m; i++) z[m + i] = -z[i];
    const st = terminalPrices(S, T, r, sigma, q, z);
    const disc = Math.exp(-r * T), N = z.length;
    const y = new Float64Array(N);
    for (let i = 0; i < N; i++) y[i] = disc * (kind === "call" ? Math.max(st[i] - K, 0) : Math.max(K - st[i], 0));
    if (control) {
      const x = new Float64Array(N); const xMean = S * Math.exp(-q * T);
      let my = 0, mx = 0;
      for (let i = 0; i < N; i++) { x[i] = disc * st[i]; my += y[i]; mx += x[i]; }
      my /= N; mx /= N;
      let cov = 0, vx = 0;
      for (let i = 0; i < N; i++) { cov += (y[i] - my) * (x[i] - mx); vx += (x[i] - mx) * (x[i] - mx); }
      const beta = cov / vx;
      for (let i = 0; i < N; i++) y[i] -= beta * (x[i] - xMean);
    }
    let mean = 0; for (let i = 0; i < N; i++) mean += y[i]; mean /= N;
    let v = 0; for (let i = 0; i < N; i++) v += (y[i] - mean) * (y[i] - mean);
    v /= (N - 1);
    const se = Math.sqrt(v / N);
    return { price: mean, se, ci95: 1.959963984540054 * se, n: N };
  }
  function qmcPrice(S, K, T, r, sigma, q, kind, n, seed, scrambles = 16) {
    const m = Math.pow(2, Math.ceil(Math.log2(Math.max(n, 16))));
    const disc = Math.exp(-r * T);
    const rand = sfc32(seed, seed * 2 + 1, seed * 3 + 2, seed * 5 + 4);
    const ests = [];
    const z = new Float64Array(m);
    for (let s = 0; s < scrambles; s++) {
      const shift = (rand() * 4294967296) >>> 0;
      for (let i = 0; i < m; i++) z[i] = normPpf(vdcScrambled(i + 1, shift));
      const st = terminalPrices(S, T, r, sigma, q, z);
      let acc = 0;
      for (let i = 0; i < m; i++) acc += kind === "call" ? Math.max(st[i] - K, 0) : Math.max(K - st[i], 0);
      ests.push(disc * acc / m);
    }
    let mean = 0; for (const e of ests) mean += e; mean /= ests.length;
    let v = 0; for (const e of ests) v += (e - mean) * (e - mean); v /= (ests.length - 1);
    const se = Math.sqrt(v / ests.length);
    return { price: mean, se, ci95: 2.131 * se, n: m };
  }

  return { erf, normCdf, normPdf, normPpf, gaussianGen, sfc32, vdcScrambled,
           bsPrice, bsGreeks, impliedVol, mcPrice, qmcPrice, terminalPrices };
});
