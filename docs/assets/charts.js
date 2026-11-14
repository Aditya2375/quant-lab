/* quant-lab charts: dependency-free canvas line/scatter charts with log scales,
   bands, guides, crosshair tooltips and date axes. */
"use strict";
(function () {
  const FMT = (v, d) => {
    if (v === null || v === undefined || !isFinite(v)) return "-";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(1);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(d === undefined ? 2 : d);
    return v.toPrecision(2);
  };
  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const fmtDate = t => { const d = new Date(t); return MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); };
  const fmtDateFull = t => { const d = new Date(t); return d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); };

  function linTicks(min, max, target = 6) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min, step0 = span / target, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    let step = mag;
    for (const m of [1, 2, 2.5, 5, 10]) { if (step0 <= m * mag) { step = m * mag; break; } }
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-12; v += step) ticks.push(+v.toFixed(12));
    return ticks;
  }
  function logTicks(min, max) {
    const ticks = [];
    const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
    for (let e = lo; e <= hi; e++) for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= min && v <= max) ticks.push(v);
    }
    return ticks;
  }
  function dateTicks(min, max, target = 6) {
    const span = max - min, step = span / target;
    const year = 365.25 * 864e5, month = year / 12, day = 864e5;
    let unit = day;
    if (step > year * 1.5) unit = year; else if (step > month * 1.5) unit = Math.max(month, 3 * month * Math.round(step / (3 * month))); else if (step > 10 * day) unit = month; else unit = Math.max(day, Math.pow(2, Math.ceil(Math.log2(step / day))) * day);
    const ticks = [];
    let t = Math.ceil(min / unit) * unit;
    if (unit >= year) { const d0 = new Date(min); t = Date.UTC(d0.getUTCFullYear() + 1, 0, 1); for (; t <= max; t = Date.UTC(new Date(t).getUTCFullYear() + 1, 0, 1)) ticks.push(t); }
    else for (; t <= max; t += unit) ticks.push(t);
    return ticks;
  }

  class Chart {
    constructor(el, opts) {
      this.el = el; this.opts = opts = opts || {};
      this.canvas = document.createElement("canvas");
      this.tip = document.createElement("div"); this.tip.className = "tip";
      el.appendChild(this.canvas); el.appendChild(this.tip);
      this.ctx = this.canvas.getContext("2d");
      this.pad = { l: 64, r: 16, t: 14, b: 40 };
      this.series = opts.series || [];
      this.guides = opts.guides || [];
      this.height = opts.height || 380;
      this.xLog = !!opts.xLog; this.yLog = !!opts.yLog;
      this.xDate = !!opts.xDate;
      this.xLabel = opts.xLabel || ""; this.yLabel = opts.yLabel || "";
      this.yFmt = opts.yFmt || (v => FMT(v)); this.xFmt = opts.xFmt || (this.xDate ? fmtDate : (v => FMT(v)));
      this.tipFmt = opts.tipFmt || null;
      this._bind();
      this._ro = new ResizeObserver(() => this.render());
      this._ro.observe(el);
      this.render();
    }
    setSeries(s) { this.series = s; this.render(); }
    _xform(v) { return this.xLog ? Math.log10(v) : v; }
    _yform(v) { return this.yLog ? Math.log10(v) : v; }
    _ranges() {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const see = (x, y) => {
        if (!isFinite(x) || !isFinite(y)) return;
        const X = this._xform(x), Y = this._yform(y);
        if (X < x0) x0 = X; if (X > x1) x1 = X;
        if (Y < y0) y0 = Y; if (Y > y1) y1 = Y;
      };
      for (const s of this.series) {
        if (s.type === "band") { for (let i = 0; i < s.x.length; i++) { see(s.x[i], s.y0[i]); see(s.x[i], s.y1[i]); } }
        else if (s.x) for (let i = 0; i < s.x.length; i++) see(s.x[i], s.y[i]);
      }
      for (const g of this.guides) if (g.y !== undefined) see(x0 === Infinity ? 0 : Math.pow(10, x0) || 1, g.y);
      if (this.opts.xMin !== undefined) x0 = this._xform(this.opts.xMin);
      if (this.opts.xMax !== undefined) x1 = this._xform(this.opts.xMax);
      if (this.opts.yMin !== undefined) y0 = this._yform(this.opts.yMin);
      if (this.opts.yMax !== undefined) y1 = this._yform(this.opts.yMax);
      if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
      if (x0 === x1) { x0 -= .5; x1 += .5; } if (y0 === y1) { y0 -= .5; y1 += .5; }
      if (!this.yLog) { const m = (y1 - y0) * .06; y0 -= m; y1 += m; }
      if (this.opts.yPadBottomZero && y0 > 0) y0 = 0;
      return [x0, x1, y0, y1];
    }
    render() {
      const w = this.el.clientWidth || 600, h = this.height, dpr = window.devicePixelRatio || 1;
      this.canvas.width = w * dpr; this.canvas.height = h * dpr;
      this.canvas.style.height = h + "px";
      const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
      const [x0, x1, y0, y1] = this._ranges();
      const px = v => this.pad.l + (this._xform(v) - x0) / (x1 - x0) * (w - this.pad.l - this.pad.r);
      const py = v => this.pad.t + (1 - (this._yform(v) - y0) / (y1 - y0)) * (h - this.pad.t - this.pad.b);
      this._px = px; this._py = py; this._r = [x0, x1, y0, y1];
      const css = getComputedStyle(document.documentElement);
      const MUTED = css.getPropertyValue("--muted").trim() || "#8b98a9";
      const BORDER = css.getPropertyValue("--border").trim() || "#1f2a37";
      // grid + ticks
      c.font = "11px ui-monospace, Menlo, monospace"; c.fillStyle = MUTED; c.strokeStyle = BORDER; c.lineWidth = 1;
      const xu = this.xDate ? "date" : this.xLog ? "log" : "lin";
      const xt = this.opts.xMin !== undefined || this.series.length ?
        (xu === "date" ? dateTicks(Math.pow(10, 0) && (this.xLog ? Math.pow(10, x0) : x0), this.xLog ? Math.pow(10, x1) : x1)
         : xu === "log" ? logTicks(Math.pow(10, x0), Math.pow(10, x1)) : linTicks(x0, x1)) : [];
      c.textAlign = "center"; c.textBaseline = "top";
      for (const t of xt) {
        const X = px(t);
        c.globalAlpha = .35; c.beginPath(); c.moveTo(X, this.pad.t); c.lineTo(X, h - this.pad.b); c.stroke(); c.globalAlpha = 1;
        c.fillText(this.xFmt(t), X, h - this.pad.b + 8);
      }
      const yv0 = this.yLog ? Math.pow(10, y0) : y0, yv1 = this.yLog ? Math.pow(10, y1) : y1;
      const yt = this.yLog ? logTicks(yv0, yv1) : linTicks(yv0, yv1);
      c.textAlign = "right"; c.textBaseline = "middle";
      for (const t of yt) {
        const Y = py(t);
        c.globalAlpha = .35; c.beginPath(); c.moveTo(this.pad.l, Y); c.lineTo(w - this.pad.r, Y); c.stroke(); c.globalAlpha = 1;
        c.fillText(this.yFmt(t), this.pad.l - 8, Y);
      }
      // axis labels
      c.textAlign = "center"; c.fillText(this.xLabel, this.pad.l + (w - this.pad.l - this.pad.r) / 2, h - 12);
      c.save(); c.translate(13, this.pad.t + (h - this.pad.t - this.pad.b) / 2); c.rotate(-Math.PI / 2); c.fillText(this.yLabel, 0, 0); c.restore();
      // guides
      for (const g of this.guides) {
        c.strokeStyle = g.color || MUTED; c.setLineDash(g.dash || [4, 4]); c.lineWidth = 1;
        if (g.y !== undefined) { c.beginPath(); c.moveTo(this.pad.l, py(g.y)); c.lineTo(w - this.pad.r, py(g.y)); c.stroke(); }
        if (g.x !== undefined) { c.beginPath(); c.moveTo(px(g.x), this.pad.t); c.lineTo(px(g.x), h - this.pad.b); c.stroke(); }
        c.setLineDash([]);
        if (g.label) { c.fillStyle = g.color || MUTED; c.textAlign = "left"; c.textBaseline = "bottom";
          c.fillText(g.label, (g.x !== undefined ? px(g.x) + 5 : this.pad.l + 5), (g.y !== undefined ? py(g.y) - 4 : this.pad.t + 4)); }
      }
      // series
      for (const s of this.series) {
        const col = s.color || "#58a6ff";
        if (s.type === "band") {
          c.beginPath();
          for (let i = 0; i < s.x.length; i++) { const X = px(s.x[i]), Y = py(s.y1[i]); i ? c.lineTo(X, Y) : c.moveTo(X, Y); }
          for (let i = s.x.length - 1; i >= 0; i--) c.lineTo(px(s.x[i]), py(s.y0[i]));
          c.closePath(); c.globalAlpha = s.alpha === undefined ? .13 : s.alpha; c.fillStyle = col; c.fill(); c.globalAlpha = 1;
        } else {
          c.strokeStyle = col; c.lineWidth = s.width || 1.6; c.beginPath();
          let started = false;
          for (let i = 0; i < s.x.length; i++) {
            if (s.y[i] === null || !isFinite(s.y[i]) || (this.yLog && s.y[i] <= 0) || (this.xLog && s.x[i] <= 0)) { started = false; continue; }
            const X = px(s.x[i]), Y = py(s.y[i]);
            started ? c.lineTo(X, Y) : c.moveTo(X, Y); started = true;
          }
          if (s.dash) c.setLineDash(s.dash);
          c.stroke(); c.setLineDash([]);
          if (s.points) {
            c.fillStyle = col;
            const idxs = s.points === "sparse" ? sparseIdx(s.x.length, 40) : [...s.x.keys()];
            for (const i of idxs) { const X = px(s.x[i]), Y = py(s.y[i]); if (isFinite(X) && isFinite(Y)) { c.beginPath(); c.arc(X, Y, 2.4, 0, 7); c.fill(); } }
          }
        }
      }
      this._seriesPx = this.series.map(s => s.type === "band" ? null :
        ({ x: s.x, y: s.y, color: s.color, name: s.name }));
    }
    _bind() {
      const el = this.el;
      el.addEventListener("mousemove", e => {
        if (!this._px) return;
        const rect = el.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        if (mx < this.pad.l || mx > rect.width - this.pad.r) { this.tip.style.display = "none"; this._drawCross(null); return; }
        // nearest datum by x across series
        let best = null;
        for (const s of this._seriesPx || []) {
          if (!s) continue;
          for (let i = 0; i < s.x.length; i++) {
            const X = this._px(s.x[i]);
            if (!isFinite(X)) continue;
            const d = Math.abs(X - mx);
            if (!best || d < best.d) best = { d, s, i };
          }
        }
        if (!best || best.d > 40) { this.tip.style.display = "none"; this._drawCross(null); return; }
        const x = best.s.x[best.i], y = best.s.y[best.i];
        this.tip.innerHTML = this.tipFmt ? this.tipFmt(x, y, best.s) :
          `<span class="muted">${this.xDate ? fmtDateFull(x) : this.xLabel + " " + this.xFmt(x)}</span><br>${best.s.name ? best.s.name + ": " : ""}<b>${this.yFmt(y)}</b>`;
        this.tip.style.display = "block";
        const tw = this.tip.offsetWidth;
        this.tip.style.left = Math.min(mx + 14, rect.width - tw - 8) + "px";
        this.tip.style.top = Math.max(my - 44, 4) + "px";
        this._drawCross({ x, y });
      });
      el.addEventListener("mouseleave", () => { this.tip.style.display = "none"; this._drawCross(null); });
    }
    _drawCross(p) {
      // overlay: redraw then crosshair (cheap enough at our sizes)
      this.render();
      if (!p) return;
      const c = this.ctx, X = this._px(p.x), Y = this._py(p.y);
      c.strokeStyle = "rgba(139,152,169,.5)"; c.setLineDash([3, 3]); c.lineWidth = 1;
      c.beginPath(); c.moveTo(X, this.pad.t); c.lineTo(X, this.canvas.height / (window.devicePixelRatio || 1) - this.pad.b); c.stroke();
      c.beginPath(); c.moveTo(this.pad.l, Y); c.lineTo(this.canvas.width / (window.devicePixelRatio || 1) - this.pad.r, Y); c.stroke();
      c.setLineDash([]);
    }
  }
  function sparseIdx(n, k) { const out = []; const step = Math.max(1, Math.floor(n / k)); for (let i = 0; i < n; i += step) out.push(i); return out; }
  window.QL = { Chart, FMT, fmtDate, fmtDateFull, sparseIdx };
})();
