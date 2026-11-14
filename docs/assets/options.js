/* options.html wiring: interactive lab + pinned-study charts. */
"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const E = QLE, $ = id => document.getElementById(id);
  const fmt$ = v => "$" + QL.FMT(v, 4);

  // ---------------- controls ----------------
  const P = { S: 754.37, K: 755, Tdays: 30, r: 4.0, sigma: 22.0, q: 1.2, kind: "call" };
  const sliders = [
    ["inS", "S", 500, 1000, 0.01, v => "$" + v.toFixed(2)],
    ["inK", "K", 500, 1000, 5, v => "$" + v.toFixed(0)],
    ["inT", "Tdays", 1, 365, 1, v => v + " days"],
    ["inSigma", "sigma", 1, 100, 0.5, v => v.toFixed(1) + "%"],
    ["inR", "r", -1, 10, 0.1, v => v.toFixed(1) + "%"],
    ["inQ", "q", 0, 5, 0.1, v => v.toFixed(1) + "%"],
  ];
  for (const [id, key, min, max, step, fmt] of sliders) {
    const el = $(id), out = $(id + "Out");
    el.min = min; el.max = max; el.step = step; el.value = P[key];
    out.textContent = fmt(P[key]);
    el.addEventListener("input", () => { P[key] = +el.value; out.textContent = fmt(P[key]); updateClosedForm(); });
  }
  $("kindCall").addEventListener("click", () => setKind("call"));
  $("kindPut").addEventListener("click", () => setKind("put"));
  function setKind(k) {
    P.kind = k;
    $("kindCall").classList.toggle("on", k === "call");
    $("kindPut").classList.toggle("on", k === "put");
    updateClosedForm();
  }

  function params() { return [P.S, P.K, P.Tdays / 365, P.r / 100, P.sigma / 100, P.q / 100, P.kind]; }

  function updateClosedForm() {
    const [S, K, T, r, sigma, q, kind] = params();
    const price = E.bsPrice(S, K, T, r, sigma, q, kind);
    const g = E.bsGreeks(S, K, T, r, sigma, q, kind);
    $("bsPrice").textContent = "$" + price.toFixed(4);
    $("gDelta").textContent = g.delta.toFixed(4);
    $("gGamma").textContent = g.gamma.toFixed(5);
    $("gVega").textContent = "$" + g.vega.toFixed(3);
    $("gTheta").textContent = "$" + g.thetaPerDay.toFixed(4);
    $("gRho").textContent = "$" + g.rho.toFixed(3);
    const other = E.bsPrice(S, K, T, r, sigma, q, kind === "call" ? "put" : "call");
    const resid = Math.abs((kind === "call" ? price - other : other - price) - (S * Math.exp(-q * T) - K * Math.exp(-r * T)));
    $("parity").textContent = resid.toExponential(1);
    return price;
  }
  updateClosedForm();

  // ---------------- live convergence run ----------------
  const convChart = new QL.Chart($("liveConv"), {
    xLog: true, yLog: true, height: 380,
    xLabel: "paths N", yLabel: "abs error ($)",
    xFmt: v => v >= 1e6 ? (v / 1e6) + "M" : v >= 1e3 ? (v / 1e3) + "k" : v,
    tipFmt: (x, y, s) => `<span class="muted">${x.toLocaleString()} paths</span><br>${s.name}: <b>$${QL.FMT(y, 4)}</b>`,
  });
  const estChart = new QL.Chart($("liveEst"), {
    xLog: true, height: 300, xLabel: "paths N", yLabel: "estimate ($)",
    xFmt: v => v >= 1e6 ? (v / 1e6) + "M" : v >= 1e3 ? (v / 1e3) + "k" : v,
    tipFmt: (x, y, s) => `<span class="muted">${x.toLocaleString()} paths</span><br>${s.name}: <b>$${y.toFixed(4)}</b>`,
  });

  let running = false;
  $("runBtn").addEventListener("click", () => { if (!running) runStudy(); });
  function runStudy() {
    running = true;
    const btn = $("runBtn"); btn.disabled = true; btn.textContent = "running...";
    const exact = updateClosedForm();
    const [S, K, T, r, sigma, q, kind] = params();
    const usePlain = $("usePlain").checked, useAV = $("useAV").checked, useQMC = $("useQMC").checked;
    const series = [], estSeries = [];
    const sPlain = { name: "plain MC", color: "#f85149", x: [], y: [], points: true };
    const sAV = { name: "antithetic + control", color: "#58a6ff", x: [], y: [], points: true };
    const sQMC = { name: "Sobol QMC", color: "#3fb950", x: [], y: [], points: true };
    if (usePlain) series.push(sPlain); if (useAV) series.push(sAV); if (useQMC) series.push(sQMC);
    const ePlain = { name: "plain MC", color: "#f85149", x: [], y: [] };
    const eAV = { name: "antithetic + control", color: "#58a6ff", x: [], y: [] };
    const eQMC = { name: "Sobol QMC", color: "#3fb950", x: [], y: [] };
    if (usePlain) estSeries.push(ePlain); if (useAV) estSeries.push(eAV); if (useQMC) estSeries.push(eQMC);
    const guides = [{ y: exact, color: "#d29922", label: "closed form $" + exact.toFixed(4) }];
    const seedBase = (Math.random() * 1e9) | 0;
    const powers = [];
    for (let p = 10; p <= 20; p++) powers.push(p);
    let idx = 0;
    const t0 = performance.now();
    const bar = $("runBar");
    (function step() {
      if (idx >= powers.length) {
        running = false; btn.disabled = false; btn.textContent = "run the study";
        $("runTime").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s in your browser";
        return;
      }
      const p = powers[idx], n = Math.pow(2, p);
      const seed = seedBase + p * 7919;
      if (usePlain) {
        const m = E.mcPrice(S, K, T, r, sigma, q, kind, n, seed, false, false);
        sPlain.x.push(n); sPlain.y.push(Math.max(Math.abs(m.price - exact), 1e-9));
        ePlain.x.push(n); ePlain.y.push(m.price);
      }
      if (useAV) {
        const m = E.mcPrice(S, K, T, r, sigma, q, kind, n, seed, true, true);
        sAV.x.push(n); sAV.y.push(Math.max(Math.abs(m.price - exact), 1e-9));
        eAV.x.push(n); eAV.y.push(m.price);
      }
      if (useQMC && n <= Math.pow(2, 18)) {
        const m = E.qmcPrice(S, K, T, r, sigma, q, kind, n, seed);
        sQMC.x.push(n); sQMC.y.push(Math.max(Math.abs(m.price - exact), 1e-9));
        eQMC.x.push(n); eQMC.y.push(m.price);
      }
      convChart.setSeries(series.slice());
      convChart.guides = [];
      estChart.setSeries(estSeries.slice());
      estChart.guides = guides;
      const ref = series.length ? series[0] : null;
      bar.style.width = Math.round(100 * (idx + 1) / powers.length) + "%";
      // final MC readout at the largest N
      if (useAV && eAV.y.length) {
        $("mcEst").textContent = "$" + eAV.y[eAV.y.length - 1].toFixed(4);
      } else if (ePlain.y.length) {
        $("mcEst").textContent = "$" + ePlain.y[ePlain.y.length - 1].toFixed(4);
      }
      idx++;
      setTimeout(step, 30);
    })();
  }

  // ---------------- pinned study ----------------
  fetch("data/options-pricer.json").then(r => r.json()).then(d => {
    const p = d.params, cf = d.closed_form, conv = d.convergence;
    $("pinnedIntro").innerHTML =
      `Contract: <b>SPY ${p.T_days}-day at-the-money call</b> (spot ${p.S.toFixed(2)}, strike ${p.K.toFixed(0)}). ` +
      `Volatility ${(100 * p.sigma).toFixed(1)}% is not a guess: it is the 21-day realized close-to-close vol of SPY, ` +
      `annualized, computed from the pinned series ending ${p.as_of}. Rate 4% and dividend yield 1.2% are stated assumptions.`;
    $("pinBS").textContent = "$" + cf.call.toFixed(4);
    $("pinMC").textContent = "$" + d.headline_mc.price.toFixed(4);
    $("pinMCErr").textContent = "\u00B1 $" + d.headline_mc.ci95.toFixed(4) + " (95% CI, " + d.headline_mc.n.toLocaleString() + " paths)";
    $("pinParity").textContent = cf.put_call_parity_residual.toExponential(1);
    $("pinIV").textContent = d.implied_vol_roundtrip.abs_err.toExponential(1);

    const rows = conv.rows, n = rows.map(r => r.n);
    const convC = new QL.Chart($("pinConv"), {
      xLog: true, yLog: true, height: 400, xLabel: "paths N", yLabel: "abs error ($)",
      xFmt: v => v >= 1e6 ? (v / 1e6) + "M" : v >= 1e3 ? (v / 1e3) + "k" : v,
      tipFmt: (x, y, s) => `<span class="muted">${x.toLocaleString()} paths</span><br>${s.name}: <b>$${QL.FMT(y, 4)}</b>`,
      series: [
        { name: "plain MC (mean of " + conv.replications + " seeds)", color: "#f85149", x: n, y: rows.map(r => r.plain.mean_abs_err), points: true },
        { name: "plain MC worst seed", color: "#f85149", type: "band", x: n, y0: rows.map(r => r.plain.mean_abs_err), y1: rows.map(r => r.plain.max_abs_err) },
        { name: "antithetic + control (mean)", color: "#58a6ff", x: n, y: rows.map(r => r.antithetic_control.mean_abs_err), points: true },
        { name: "av+cv worst seed", color: "#58a6ff", type: "band", x: n, y0: rows.map(r => r.antithetic_control.mean_abs_err), y1: rows.map(r => r.antithetic_control.max_abs_err) },
        { name: "Sobol QMC (mean)", color: "#3fb950", x: n, y: rows.map(r => r.sobol.mean_abs_err), points: true },
      ],
      guides: [
        { y: cf.call * 0.5 / Math.sqrt(128) * Math.sqrt(128), color: "#5b6675" },
      ],
    });
    // reference slopes drawn as plain series so log axes stay honest
    const exact = conv.exact;
    convC.series.push({ name: "1/sqrt(N) guide", color: "#8b98a9", dash: [4, 4], width: 1, x: n, y: n.map(v => 0.5 * exact * Math.sqrt(n[0] / v)) });
    convC.series.push({ name: "1/N guide", color: "#8b98a9", dash: [2, 4], width: 1, x: n, y: n.map(v => 0.02 * exact * (n[0] / v)) });
    convC.render();
    // skew chart
    const cc = d.market_crosscheck;
    const pts = cc.points;
    new QL.Chart($("pinSkew"), {
      height: 400, xLabel: "strike ($)", yLabel: "implied vol (%)",
      yFmt: v => v.toFixed(1) + "%",
      tipFmt: (x, y) => `<span class="muted">strike $${x}</span><br>implied vol <b>${y.toFixed(2)}%</b>`,
      series: [
        { name: "my solver (mid price)", color: "#bc8cff", x: pts.map(p => p.strike), y: pts.map(p => 100 * p.my_iv), points: "sparse" },
      ],
      guides: [
        { y: 100 * cc.atm_iv, color: "#8b98a9", dash: [4, 4], label: "flat ATM vol " + (100 * cc.atm_iv).toFixed(1) + "%" },
        { x: p.S, color: "#d29922", dash: [2, 4], label: "spot " + p.S.toFixed(0) },
      ],
    });
    // flat-vs-market table: a few representative strikes
    const picks = [pts[0], pts[Math.floor(pts.length * .25)], pts.reduce((a, b) => Math.abs(b.strike - p.S) < Math.abs(a.strike - p.S) ? b : a), pts[Math.floor(pts.length * .75)], pts[pts.length - 1]];
    $("skewTable").innerHTML = picks.map(q =>
      `<tr><td>$${q.strike.toFixed(0)}</td><td>${(100 * q.moneyness).toFixed(1)}%</td><td>$${q.mid.toFixed(2)}</td>` +
      `<td>$${q.flat_price.toFixed(2)}</td><td class="${q.flat_minus_mid < 0 ? "neg" : "pos"}">` +
      `${q.flat_minus_mid > 0 ? "+" : ""}$${q.flat_minus_mid.toFixed(2)}</td><td>${(100 * q.my_iv).toFixed(1)}%</td></tr>`).join("");
    $("skewNote").textContent =
      `Chain: ${cc.n_strikes} liquid SPY calls expiring ${cc.expiry}, mids from a snapshot pinned ${p.as_of}. ` +
      `My solver agrees with the provider's implied vols to a median of 0.69 vol points.`;
  }).catch(e => { console.error(e); });
});
