/* quant-lab study 02: pairs trading scoreboard + survivor detail charts.
   All numbers come from data/pairs-study.json, exported by the repo's
   backtester from the pinned CSVs. */
"use strict";
(function () {
  const $ = id => document.getElementById(id);

  const ONE_LINERS = {
    "V/MA": "Strongest cointegration evidence in the set, 19-day half-life, in-sample Sharpe 1.11. Out-of-sample: 0.07. The edge was already gone.",
    "HD/LOW": "The survivor. Same industry, real economic link, and the only pair that did better out-of-sample than in it. Three trades in five years.",
    "KO/PEP": "The textbook pair. Decoupled in 2022; one trade opened and never mean-reverted. -27% on an unclosed position.",
    "PG/KMB": "Passes at 5%, trades twice, makes 2.7% over five years. Dead money with extra steps.",
    "MS/GS": "No rejection at 5%, but it grinds out a small positive result. Two banks, one macro book; weaker than it looks.",
    "NEE/SO": "Utilities move on regulation, not on each other. No rejection, loses money both windows' worth of risk.",
    "JPM/BAC": "Big banks look like one trade but are not. No rejection at 5%, -16.7% out-of-sample.",
    "LIN/APD": "Industrial gases: correlated prices, non-stationary spread. The difference matters and it costs 15.2%.",
    "XOM/CVX": "Oil majors share the crude factor; the residual is not stationary. The +9.8% is one lucky trade.",
    "UPS/FDX": "Zero cointegration evidence (ADF t = -0.58) and the best raw return in the set. Two trades of pure luck, reported on purpose.",
  };

  fetch("data/pairs-study.json").then(r => r.json()).then(d => {
    const pairs = d.pairs;
    const fmtP = v => (v * 100).toFixed(1) + "%";
    const fmtS = v => (v >= 0 ? "" : "") + v.toFixed(2);

    // verdict cards
    const outs = pairs.map(p => p.sharpe).sort((a, b) => a - b);
    $("cardMedian").textContent = ((outs[4] + outs[5]) / 2).toFixed(2);
    const hd = pairs.find(p => p.pair === "HD/LOW");
    $("cardSurvivorSub").textContent = "Sharpe " + hd.sharpe.toFixed(2) +
      ", +" + (hd.total_return * 100).toFixed(1) + "%, " + hd.n_trades + " trades";

    // windows
    $("formWindow").textContent = pairs[0].formation.start + " to " + pairs[0].formation.end;
    $("tradeWindow").textContent = pairs[0].trading.start + " to " + pairs[0].trading.end;

    // table
    const tb = $("pairTable");
    pairs.forEach(p => {
      const tr = document.createElement("tr");
      tr.style.cursor = "pointer";
      const cls = v => v < 0 ? 'style="color:var(--red,#f85149)"' : "";
      tr.innerHTML =
        "<td><b>" + p.pair + "</b></td>" +
        "<td " + (p.adf_t < -2.86 ? 'style="color:var(--green,#3fb950)"' : "") + ">" + p.adf_t.toFixed(2) + "</td>" +
        "<td>" + (p.half_life_days > 500 ? "none" : p.half_life_days.toFixed(0) + "d") + "</td>" +
        "<td>" + p.in_sample.sharpe.toFixed(2) + "</td>" +
        "<td " + cls(p.sharpe) + ">" + p.sharpe.toFixed(2) + "</td>" +
        "<td " + cls(p.total_return) + ">" + fmtP(p.total_return) + "</td>" +
        "<td>" + fmtP(p.max_drawdown) + "</td>" +
        "<td>" + p.n_trades + "</td>";
      tr.addEventListener("click", () => {
        $("pairNote").innerHTML = "<b>" + p.pair + ".</b> " + ONE_LINERS[p.pair];
        tb.querySelectorAll("tr").forEach(r => r.style.background = "");
        tr.style.background = "rgba(94,161,255,.08)";
      });
      tb.appendChild(tr);
    });

    // HD/LOW detail charts
    const det = d.detail;
    $("hdSharpe").textContent = hd.sharpe.toFixed(2);
    const t = det.dates.map(s => new Date(s + "T00:00:00Z").getTime());

    new QL.Chart($("zChart"), {
      xDate: true, height: 300, yLabel: "z-score",
      tipFmt: (x, y, s) => `<span class="muted">${QL.fmtDateFull(x)}</span><br>${s.name}: <b>${QL.FMT(y, 2)}</b>`,
      series: [
        { name: "spread z-score", color: "#58a6ff", x: t, y: det.z, width: 1.2 },
      ],
      guides: [
        { y: 2.0, color: "#f85149", dash: [5, 4], label: "entry |z| = 2" },
        { y: -2.0, color: "#f85149", dash: [5, 4] },
        { y: 0.5, color: "#8b98a9", dash: [2, 4], label: "exit |z| = 0.5" },
        { y: -0.5, color: "#8b98a9", dash: [2, 4] },
        { y: 0, color: "#3a4150" },
      ],
    });

    new QL.Chart($("eqChart"), {
      xDate: true, height: 280, yLabel: "equity",
      tipFmt: (x, y, s) => `${s.name}: <b>${QL.FMT(y, 3)}</b>`,
      series: [
        { name: "with costs (5 bps/side/leg)", color: "#3fb950", x: t, y: det.equity, width: 1.6 },
        { name: "frictionless", color: "#58a6ff", x: t, y: det.equity_nocost, width: 1.1, dash: [4, 4] },
      ],
      guides: [{ y: 1.0, color: "#3a4150" }],
    });

    // sensitivity grid
    const st = $("sensTable");
    det.sensitivity.forEach(g => {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + g.entry_z.toFixed(1) + "</td><td>" + g.cost_bps.toFixed(0) +
        " bps</td><td>" + g.n_trades + "</td><td>" + fmtP(g.total_return) +
        "</td><td>" + g.sharpe.toFixed(2) + "</td>";
      st.appendChild(tr);
    });
  }).catch(e => {
    $("pairNote").textContent = "study data failed to load: " + e;
  });
})();
