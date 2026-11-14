/* quant-lab study 03: scoreboard render from the pinned vol-study.json. */
"use strict";
(function () {
  const $ = id => document.getElementById(id);

  const ROWS = [
    ["garch", "GARCH(1,1), own MLE", "winner on the robust loss"],
    ["ml", "gradient-boosted trees, 12 features", "loses on QLIKE, edges MSE"],
    ["rv21", "21-day rolling variance", "uncomfortably close to GARCH"],
    ["naive", "tomorrow = today", "catastrophic on QLIKE"],
  ];

  fetch("data/vol-study.json").then(r => r.json()).then(d => {
    $("wfWindow").textContent = d.window[0] + " to " + d.window[1];
    $("wfDays").textContent = d.n_forecast_days.toLocaleString();
    $("dmStat").textContent = d.dm.stat.toFixed(2) +
      " (negative = GARCH ahead)";

    const tb = $("scoreTable");
    ROWS.forEach(([key, label, verdict]) => {
      const s = d.scoreboard[key];
      const tr = document.createElement("tr");
      const best = key === "garch" ? 'style="color:var(--green,#3fb950)"' : "";
      tr.innerHTML = "<td><b>" + label + "</b></td>" +
        "<td " + best + ">" + s.qlike.toFixed(3) + "</td>" +
        "<td>" + s.mse.toExponential(2) + "</td>" +
        "<td>" + s.corr.toFixed(3) + "</td>" +
        "<td>" + verdict + "</td>";
      tb.appendChild(tr);
    });
  }).catch(e => {
    $("scoreTable").innerHTML = "<tr><td colspan=5>study data failed to load: " + e + "</td></tr>";
  });
})();
