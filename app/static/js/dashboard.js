/* BIOPINN results dashboard -- vanilla JS + Plotly. No build step: this
   file is served as-is by FastAPI's StaticFiles. */

(() => {
  "use strict";

  const ZONE_COLORS = {
    proliferating_rim: "#c9623f",
    quiescent_zone: "#d9a441",
    necrotic_core: "#4c5b60",
  };
  const ZONE_LABELS = {
    proliferating_rim: "Proliferating rim",
    quiescent_zone: "Quiescent zone",
    necrotic_core: "Necrotic core",
  };

  const PLOT_FONT = { family: "Inter, system-ui, sans-serif", size: 12, color: "#12232e" };
  const PLOT_MARGIN = { l: 56, r: 16, t: 8, b: 44 };
  const PLOT_MARGIN_LEGEND = { l: 56, r: 16, t: 8, b: 80 };
  const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  let meta = null;
  let latestPredict = null;
  let latestOptimization = null;
  let debounceTimer = null;

  const $ = (id) => document.getElementById(id);

  function setStatus(state, text) {
    const pill = $("status-pill");
    pill.dataset.state = state;
    $("status-text").textContent = text;
  }

  function fmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    if (Math.abs(value) !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e5)) {
      return Number(value).toExponential(digits - 1);
    }
    return Number(value).toFixed(digits);
  }

  function badge(pass, textPass = "pass", textFail = "fail") {
    const cls = pass ? "pass" : "fail";
    const text = pass ? textPass : textFail;
    return `<span class="badge ${cls}">${text}</span>`;
  }

  // ------------------------------------------------------------------ //
  // Sidebar controls                                                    //
  // ------------------------------------------------------------------ //

  function setupControls() {
    const params = meta.parameters;
    for (const key of Object.keys(params)) {
      const p = params[key];
      const input = $(`in-${key}`);
      const step = (p.max - p.min) / 200;
      input.min = p.min;
      input.max = p.max;
      input.step = step > 0 ? step : 1;
      input.value = p.default;
      $(`val-${key}`).textContent = fmt(p.default, key === "k_d_per_hr" ? 4 : 1);
      input.addEventListener("input", () => {
        $(`val-${key}`).textContent = fmt(Number(input.value), key === "k_d_per_hr" ? 4 : 1);
        scheduleRunPredict();
      });
    }

    const zoneLegend = $("zone-legend");
    zoneLegend.innerHTML = meta.zones
      .map((z) => `<span><span class="zone-swatch" style="background:${ZONE_COLORS[z] || "#999"}"></span>${ZONE_LABELS[z] || z}</span>`)
      .join("");

    $("in-time").addEventListener("input", renderTimeDependentPanels);
  }

  function currentParams() {
    return {
      R_um: Number($("in-R_um").value),
      d_NP_nm: Number($("in-d_NP_nm").value),
      C0_uM: Number($("in-C0_uM").value),
      k_d_per_hr: Number($("in-k_d_per_hr").value),
      t_max_hr: Number($("in-t_max_hr").value),
    };
  }

  function scheduleRunPredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPredict, 180);
  }

  // ------------------------------------------------------------------ //
  // Live prediction                                                     //
  // ------------------------------------------------------------------ //

  async function runPredict() {
    const params = currentParams();
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      if (!res.ok) throw new Error(`predict failed (${res.status})`);
      latestPredict = await res.json();
      latestPredict.params = params;
      renderPredict();
    } catch (err) {
      console.error(err);
    }
  }

  function timeIndex() {
    if (!latestPredict) return 0;
    const frac = Number($("in-time").value);
    const n = latestPredict.t.length;
    return Math.min(n - 1, Math.max(0, Math.round(frac * (n - 1))));
  }

  function renderPredict() {
    if (!latestPredict) return;
    const d = latestPredict;

    $("val-time").textContent = fmt(d.t[timeIndex()], 1);
    $("stat-kill-fraction").textContent = fmt(d.kill_fraction * 100, 1) + "%";
    $("stat-resistance").textContent = fmt(d.resistance.resistant_volume_fraction * 100, 1) + "%";

    Plotly.react(
      "plot-concentration-heatmap",
      [{ z: d.C, x: d.r, y: d.t, type: "heatmap", colorscale: "Hot", reversescale: true, colorbar: { title: "μM" } }],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "Radial distance (μm)" }, yaxis: { title: "Time (hr)" } },
      PLOT_CONFIG
    );

    Plotly.react(
      "plot-penetration",
      [{ x: d.t, y: d.penetration_depth, type: "scatter", mode: "lines", line: { color: "#1c7c74", width: 2.5 }, fill: "tozeroy", fillcolor: "rgba(28,124,116,0.12)" }],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "Time (hr)" }, yaxis: { title: "Penetration depth (μm)" } },
      PLOT_CONFIG
    );

    Plotly.react(
      "plot-viability",
      [{ z: d.viability, x: d.r, y: d.t, type: "heatmap", colorscale: "RdYlGn", zmin: 0, zmax: 100, colorbar: { title: "%" } }],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "Radial distance (μm)" }, yaxis: { title: "Time (hr)" } },
      PLOT_CONFIG
    );

    const cytotoxTrace = { z: d.cytotoxicity.map((row) => row.map((v) => v * 100)), x: d.r, y: d.t, type: "heatmap", colorscale: "Reds", zmin: 0, zmax: 100, colorbar: { title: "%" } };
    Plotly.react(
      "plot-cytotoxicity",
      [cytotoxTrace],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "Radial distance (μm)" }, yaxis: { title: "Time (hr)" } },
      PLOT_CONFIG
    );

    renderTimeDependentPanels();
    if (latestOptimization) renderOptimization(latestOptimization);
  }

  function renderTimeDependentPanels() {
    if (!latestPredict) return;
    const d = latestPredict;
    const idx = timeIndex();
    $("val-time").textContent = fmt(d.t[idx], 1);
    $("stat-penetration").textContent = fmt(d.penetration_depth[idx], 1);

    const zoneColors = d.zones.map((z) => ZONE_COLORS[z] || "#999");
    Plotly.react(
      "plot-radial-profile",
      [{ x: d.r, y: d.C[idx], type: "scatter", mode: "lines", line: { color: "#12232e", width: 2 } },
       { x: d.r, y: d.r.map(() => d.resistance.threshold_uM), type: "scatter", mode: "lines", line: { color: "#b3462c", width: 1, dash: "dot" }, name: "sub-therapeutic threshold" }],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "Radial distance (μm)" }, yaxis: { title: "Concentration (μM)" }, showlegend: false },
      PLOT_CONFIG
    );
  }

  // ------------------------------------------------------------------ //
  // Optimization panel (polled once, cached server-side)                //
  // ------------------------------------------------------------------ //

  async function pollOptimization() {
    try {
      const res = await fetch("/api/optimization");
      const data = await res.json();
      if (res.status === 202) {
        setTimeout(pollOptimization, 1500);
        return;
      }
      latestOptimization = data;
      renderOptimization(data);
    } catch (err) {
      console.error(err);
      setTimeout(pollOptimization, 3000);
    }
  }

  function renderOptimization(data) {
    const el = $("optimization-content");
    if (data.status === "error") {
      el.innerHTML = `<p class="empty-state">Optimization failed: ${data.message}</p>`;
      return;
    }
    const currentR = latestPredict ? latestPredict.params.R_um : null;
    const currentEta = latestPredict ? latestPredict.kill_fraction : null;

    const rows = data.radii
      .map((r) => {
        const isCurrent = currentR !== null && Math.abs(r.R_um - currentR) < 1e-6;
        return `<tr class="${isCurrent ? "current-row" : ""}">
          <td>${fmt(r.R_um, 0)} μm</td>
          <td>${fmt(r.d_NP_star_nm, 1)} nm</td>
          <td>${fmt(r.C0_star_uM, 2)} μM</td>
          <td>${fmt(r.max_eta * 100, 1)}%</td>
          <td>${fmt(r.resistant_volume_fraction * 100, 1)}%</td>
          <td>${isCurrent && currentEta !== null ? fmt(currentEta * 100, 1) + "%" : "—"}</td>
        </tr>`;
      })
      .join("");

    el.innerHTML = `<table class="data-table">
      <thead><tr><th>Radius</th><th>d_NP*</th><th>C0*</th><th>Max η</th><th>Resistant volume</th><th>Your current η</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  // ------------------------------------------------------------------ //
  // Evaluation panel (six metrics, residual histogram, overlay)         //
  // ------------------------------------------------------------------ //

  async function pollEvaluation() {
    try {
      const res = await fetch("/api/evaluation");
      const data = await res.json();
      if (res.status === 202) {
        setTimeout(pollEvaluation, 1500);
        return;
      }
      renderEvaluation(data);
    } catch (err) {
      console.error(err);
      setTimeout(pollEvaluation, 3000);
    }
  }

  function renderEvaluation(data) {
    if (data.status === "error") {
      $("metrics-content").innerHTML = `<p class="empty-state">Evaluation failed: ${data.message}</p>`;
      $("stat-h1").textContent = "—";
      $("stat-h1-detail").textContent = "evaluation failed";
      return;
    }
    const g = data.global;
    const pf = data.threshold_pass_fail;
    const rows = [
      ["RMSE", g.rmse, pf.rmse, "μM"],
      ["MAE", g.mae, pf.mae, "μM"],
      ["R²", g.r2, pf.r2, ""],
      ["L2 relative error", g.l2_relative, pf.l2_relative_error, ""],
      ["Mean PDE residual", g.mean_pde_residual, pf.mean_pde_residual, ""],
      ["Penetration RMSE", g.penetration_rmse_um, pf.penetration_rmse_um, "μm"],
    ];
    $("metrics-content").innerHTML = `<table class="data-table">
      <thead><tr><th>Metric</th><th>Value</th><th></th></tr></thead>
      <tbody>${rows.map(([name, value, pass, unit]) => `<tr><td>${name}</td><td>${fmt(value)} ${unit}</td><td>${badge(pass)}</td></tr>`).join("")}</tbody>
    </table>`;

    const h1 = data.hypotheses.H1;
    $("stat-h1").innerHTML = badge(h1.pass, "H1 pass", "H1 fail");
    $("stat-h1-detail").textContent = `rmse=${fmt(h1.rmse_uM)}μM, r²=${fmt(h1.r2, 4)} over ${data.n_test_sims} sims`;

    const hist = data.residual_histogram;
    const centers = hist.bin_edges.slice(0, -1).map((e, i) => (e + hist.bin_edges[i + 1]) / 2);
    Plotly.react(
      "plot-residual-hist",
      [{ x: centers, y: hist.counts, type: "bar", marker: { color: "#1c7c74" } }],
      { font: PLOT_FONT, margin: PLOT_MARGIN, xaxis: { title: "|PDE residual|" }, yaxis: { title: "count" } },
      PLOT_CONFIG
    );

    const ov = data.overlay;
    const hourIdxs = [];
    [6, 24, 48, 72].forEach((h) => {
      if (h <= ov.t[ov.t.length - 1]) {
        let best = 0;
        let bestDiff = Infinity;
        ov.t.forEach((tv, i) => {
          const diff = Math.abs(tv - h);
          if (diff < bestDiff) { bestDiff = diff; best = i; }
        });
        hourIdxs.push([h, best]);
      }
    });
    const overlayTraces = [];
    hourIdxs.forEach(([h, idx], i) => {
      overlayTraces.push({ x: ov.r, y: ov.C_true[idx], type: "scatter", mode: "lines", line: { color: "#12232e", width: 2 }, name: `FDM t=${h}h`, showlegend: i === 0, legendgroup: "fdm" });
      overlayTraces.push({ x: ov.r, y: ov.C_pred[idx], type: "scatter", mode: "lines", line: { color: "#1c7c74", width: 2, dash: "dash" }, name: `PINN t=${h}h`, showlegend: i === 0, legendgroup: "pinn" });
    });
    $("overlay-caption").textContent = `Sim ${ov.sim_id} — R=${fmt(ov.R_um, 0)}μm, d_NP=${fmt(ov.d_NP_nm, 0)}nm, C0=${fmt(ov.C0_uM, 1)}μM`;
    Plotly.react(
      "plot-overlay",
      overlayTraces,
      { font: PLOT_FONT, margin: PLOT_MARGIN_LEGEND, xaxis: { title: "Radius (μm)" }, yaxis: { title: "Concentration (μM)" }, legend: { orientation: "h", y: -0.45 } },
      PLOT_CONFIG
    );
  }

  // ------------------------------------------------------------------ //
  // Ablation panel                                                      //
  // ------------------------------------------------------------------ //

  async function pollAblation() {
    try {
      const res = await fetch("/api/ablation");
      const data = await res.json();
      if (res.status === 202) {
        setTimeout(pollAblation, 1500);
        return;
      }
      renderAblation(data);
    } catch (err) {
      console.error(err);
      setTimeout(pollAblation, 3000);
    }
  }

  function renderAblation(data) {
    const el = $("ablation-content");
    if (data.status === "unavailable") {
      el.innerHTML = `<p class="empty-state">${data.message}</p>`;
      return;
    }
    if (data.status === "error") {
      el.innerHTML = `<p class="empty-state">Ablation failed: ${data.message}</p>`;
      return;
    }

    el.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <table class="data-table">
          <thead><tr><th>Metric</th><th>BIOPINN</th><th>Baseline (w_phys=0)</th></tr></thead>
          <tbody>
            <tr><td>Mean |PDE residual|</td><td>${fmt(data.biopinn.mean_abs_residual)}</td><td>${fmt(data.baseline.mean_abs_residual)}</td></tr>
            <tr><td>Max |PDE residual|</td><td>${fmt(data.biopinn.max_abs_residual)}</td><td>${fmt(data.baseline.max_abs_residual)}</td></tr>
            <tr><td>Physical consistency</td><td>${fmt(data.biopinn.physical_consistency_pct, 1)}%</td><td>${fmt(data.baseline.physical_consistency_pct, 1)}%</td></tr>
            <tr><td>Improvement factor (mean)</td><td colspan="2">${fmt(data.improvement_factor_mean, 2)}×</td></tr>
            <tr><td>Wilcoxon p-value</td><td colspan="2">${fmt(data.wilcoxon.p_value, 4)} (${data.wilcoxon.significant ? "significant" : "not significant"})</td></tr>
            <tr><td>H5 (≥${data.H5.target_improvement_factor}× and significant)</td><td colspan="2">${badge(data.H5.pass)}</td></tr>
          </tbody>
        </table>
        <div id="plot-ablation-hist" class="plot short"></div>
      </div>`;

    const bio = data.residuals_biopinn_hist;
    const base = data.residuals_baseline_hist;
    const bioCenters = bio.bin_edges.slice(0, -1).map((e, i) => (e + bio.bin_edges[i + 1]) / 2);
    const baseCenters = base.bin_edges.slice(0, -1).map((e, i) => (e + base.bin_edges[i + 1]) / 2);
    Plotly.react(
      "plot-ablation-hist",
      [
        { x: bioCenters, y: bio.counts, type: "bar", name: "BIOPINN", marker: { color: "#1c7c74" }, opacity: 0.85 },
        { x: baseCenters, y: base.counts, type: "bar", name: "baseline", marker: { color: "#b3462c" }, opacity: 0.6 },
      ],
      { font: PLOT_FONT, margin: PLOT_MARGIN_LEGEND, barmode: "overlay", xaxis: { title: "|PDE residual|" }, yaxis: { title: "count" }, legend: { orientation: "h", y: -0.45 } },
      PLOT_CONFIG
    );
  }

  // ------------------------------------------------------------------ //
  // Boot                                                                 //
  // ------------------------------------------------------------------ //

  async function boot() {
    try {
      const res = await fetch("/api/meta");
      if (res.status === 503) {
        const data = await res.json();
        setStatus("error", data.message || "Model failed to load.");
        setTimeout(boot, 2000);
        return;
      }
      meta = await res.json();
      setupControls();
      setStatus("ready", "Model loaded");
      await runPredict();
      pollOptimization();
      pollEvaluation();
      pollAblation();
    } catch (err) {
      console.error(err);
      setStatus("error", "Could not reach the server.");
      setTimeout(boot, 2000);
    }
  }

  $("btn-export-json").addEventListener("click", () => window.open("/api/export/metrics.json", "_blank"));
  $("btn-export-csv").addEventListener("click", () => window.open("/api/export/metrics.csv", "_blank"));
  $("btn-run-optimization").addEventListener("click", () => {
    setStatus("busy", "Refreshing optimization…");
    pollOptimization();
    setTimeout(() => setStatus("ready", "Model loaded"), 600);
  });

  boot();
})();
