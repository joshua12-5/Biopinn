"""Phase 14: manuscript Results & Discussion asset generation.

Computes every figure (Fig 4.1-4.10) and table (Table 4.1-4.9) in the
BIOPINN manuscript's Results & Discussion chapter from real pipeline
outputs -- a trained checkpoint, the held-out test set, and (for the
ablation table) a w_phys=0 baseline checkpoint. Every number is computed
here or in the `src/` modules it calls (`evaluate.py`, `ablation.py`,
`optimize.py`, `biology.py`, `fdm_solver.py`, `microenvironment.py`); this
module never reimplements the underlying physics, and never retrains.

`scripts/generate_results.py` is the CLI that calls these functions, saves
PNG+PDF figures and CSV tables to `results/paper/`, compiles the tables
into a Word document, and writes a reproducibility manifest.

A note on parameter sensitivity: the manuscript's baseline configuration
(R=400um, d_NP=100nm, C0=10uM, k_d=0.01/hr, t=72hr) is the same "typical
demo" regime that earlier phases (see tests/test_optimize.py and
tests/test_biology.py) found saturates the whole tumor well within the
simulation window -- which can wash out the heterogeneous-vs-homogeneous
contrast that Table 4.6/Fig 4.9 and Hypothesis H3 are checking for. This
module computes Table 4.6 and H3 honestly at the manuscript's literal
baseline rather than silently substituting different parameters to force
agreement with the manuscript's illustrative [SAMPLE] narrative -- if the
real computation disagrees with the sample text, that disagreement is
itself the correct output of replacing [SAMPLE] placeholders with actual
results, and is called out explicitly in Table 4.9 and the manifest.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.ablation import compare_residuals, wilcoxon_test
from src.biology import cytotoxicity_map, predict_concentration_field, viability_map
from src.config import resolve_path
from src.evaluate import (
    compute_metrics,
    compute_pde_residual_stats,
    evaluate_test_set,
    penetration_depth_from_concentration,
)
from src.fdm_solver import solve_fdm
from src.microenvironment import assign_zones, oxygen_gradient, radial_grid, stokes_einstein_diffusivity
from src.optimize import (
    _resistance_map,
    _summarize_field,
    grid_search_radius,
    kill_fraction,
    optimize_all_radii,
)

ZONE_LABELS = {"proliferating_rim": "Proliferating rim", "quiescent_zone": "Quiescent zone", "necrotic_core": "Necrotic core"}
ZONE_COLORS = {"proliferating_rim": "#c9623f", "quiescent_zone": "#d9a441", "necrotic_core": "#4c5b60"}


# --------------------------------------------------------------------------- #
# Shared inputs
# --------------------------------------------------------------------------- #


def baseline_params(config: dict) -> dict:
    """The manuscript's fixed baseline parameter set (R, d_NP, C0, k_d, t_max)."""
    return dict(config["paper"]["baseline"])


def load_training_history(config: dict) -> dict:
    """Per-component loss history from the Colab training run.

    Looks for `paths.training_history` (written directly by the current
    notebook) first, falling back to extracting "history" out of the
    older `artifacts/training_run.json` record for backward compatibility
    with a notebook run before that dedicated file existed.
    """
    history_path = resolve_path(config, "training_history")
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            return json.load(f)

    run_record_path = resolve_path(config, "artifacts") / "training_run.json"
    if run_record_path.exists():
        with open(run_record_path, encoding="utf-8") as f:
            return json.load(f)["history"]

    raise FileNotFoundError(
        f"No training history found at {history_path} or {run_record_path}. "
        "Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here."
    )


def solve_diameter_sweep(config: dict, params: dict | None = None) -> dict[float, dict]:
    """FDM-solve the baseline case at every nanoparticle diameter in the
    configured sweep, holding R/C0/k_d/t_max fixed. Shared by Table 4.1
    (baseline == the d_NP=100nm entry), Table 4.2, Table 4.4, and Fig 4.3,
    so each diameter is only solved once."""
    p = {**baseline_params(config), **(params or {})}
    return {
        d_NP: solve_fdm(p["R_um"], d_NP, p["C0_uM"], p["k_d_per_hr"], p["t_max_hr"], config)
        for d_NP in config["paper"]["sweeps"]["d_NP_nm"]
    }


def _baseline_fdm_from_sweep(sweep: dict[float, dict], params: dict, config: dict) -> dict:
    """Look up the baseline d_NP's solve in an existing diameter sweep,
    falling back to a fresh solve_fdm call if the sweep doesn't happen to
    include it (e.g. a config override that trims paper.sweeps.d_NP_nm
    without also including paper.baseline.d_NP_nm)."""
    if params["d_NP_nm"] in sweep:
        return sweep[params["d_NP_nm"]]
    return solve_fdm(params["R_um"], params["d_NP_nm"], params["C0_uM"], params["k_d_per_hr"], params["t_max_hr"], config)


def solve_hetero_and_homogeneous(config: dict, params: dict | None = None) -> dict:
    """Re-solves the FDM diffusion model twice at the given parameters (defaults
    to the manuscript baseline): once with the normal three-zone D_eff(r), once
    with a spatially homogeneous D_eff equal to the arithmetic mean of the three
    zone factors. Shared by Fig 4.9 and Table 4.6 so the physics is solved once."""
    p = {**baseline_params(config), **(params or {})}
    hetero = solve_fdm(p["R_um"], p["d_NP_nm"], p["C0_uM"], p["k_d_per_hr"], p["t_max_hr"], config)

    const = config["constants"]
    D_free_m2_s = stokes_einstein_diffusivity(p["d_NP_nm"], const["T"], const["eta"], const["k_B"])
    D_free_um2_hr = D_free_m2_s * 1e12 * 3600.0
    D_eff_homogeneous = D_free_um2_hr * float(np.mean(list(config["microenvironment"]["f_zone"].values())))
    homog = solve_fdm(p["R_um"], p["d_NP_nm"], p["C0_uM"], p["k_d_per_hr"], p["t_max_hr"], config, D_eff_override=D_eff_homogeneous)

    return {"params": p, "hetero": hetero, "homog": homog, "D_eff_homogeneous_um2_per_hr": D_eff_homogeneous}


def _volume_average(values: np.ndarray, r: np.ndarray) -> float:
    """Volume-weighted average over a sphere (dV proportional to r^2 dr) --
    a reporting statistic, not physics; matches src/optimize.py's convention."""
    weights = r**2
    return float(np.sum(values * weights) / np.sum(weights))


def _save_figure(fig: plt.Figure, output_dir, stem: str) -> dict:
    """Save a figure as both 300 DPI PNG and PDF, per the paper-asset spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_4_1_concentration_heatmap(model, config: dict, norm_stats: dict, n_r: int = 150, n_t: int = 100):
    """Fig 4.1 -- Spatiotemporal Concentration Heatmap. PINN C(r,t) at baseline."""
    params = baseline_params(config)
    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    t = np.linspace(0.0, params["t_max_hr"], n_t)
    C = predict_concentration_field(model, r, t, params, norm_stats)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.pcolormesh(r, t, C, cmap="viridis", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration C (μM)")
    ax.set_xlabel("Radial distance r (μm)")
    ax.set_ylabel("Time t (hr)")
    fig.tight_layout()
    return fig, {"params": params, "max_concentration_uM": float(C.max()), "caption": "Spatiotemporal Concentration Heatmap"}


def fig_4_2_radial_profiles(model, config: dict, norm_stats: dict, n_r: int = 200):
    """Fig 4.2 -- Radial Concentration Profiles at Selected Time Points."""
    params = baseline_params(config)
    time_points = config["paper"]["sweeps"]["time_points_hr"]
    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    t = np.array([float(x) for x in time_points])
    C = predict_concentration_field(model, r, t, params, norm_stats)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(time_points)))
    for i, hours in enumerate(time_points):
        ax.plot(r, C[i, :], color=colors[i], linewidth=2, label=f"t = {hours:g} hr")
    ax.set_xlabel("Radial distance r (μm)")
    ax.set_ylabel("Concentration C (μM)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, {"params": params, "time_points_hr": time_points, "caption": "Radial Concentration Profiles at Selected Time Points"}


def fig_4_3_pinn_vs_fdm_t24(model, config: dict, norm_stats: dict, diameter_sweep: dict | None = None):
    """Fig 4.3 -- PINN vs. FDM Concentration Profile Comparison at t=24hr, baseline, with a residual inset."""
    params = baseline_params(config)
    sweep = diameter_sweep or solve_diameter_sweep(config)
    baseline_fdm = _baseline_fdm_from_sweep(sweep, params, config)
    r, t_fdm, C_fdm = baseline_fdm["r"], baseline_fdm["t"], baseline_fdm["C"]
    t_idx = int(np.argmin(np.abs(t_fdm - 24.0)))

    C_pred = predict_concentration_field(model, r, np.array([24.0]), params, norm_stats)[0]
    C_true = C_fdm[t_idx, :]
    residual = C_pred - C_true

    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, height_ratios=[3, 1])
    ax_main.plot(r, C_true, "k-", linewidth=2, label="FDM reference")
    ax_main.plot(r, C_pred, color="#1c7c74", linestyle="--", linewidth=2, label="PINN prediction")
    ax_main.set_ylabel("Concentration C (μM)")
    ax_main.legend(fontsize=9)
    ax_main.grid(alpha=0.3)

    ax_res.axhline(0.0, color="gray", linewidth=1)
    ax_res.plot(r, residual, color="#b3462c", linewidth=1.5)
    ax_res.set_xlabel("Radial distance r (μm)")
    ax_res.set_ylabel("Residual (μM)")
    ax_res.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "params": params,
        "t_hr": float(t_fdm[t_idx]),
        "mean_abs_residual_uM": float(np.mean(np.abs(residual))),
        "max_abs_residual_uM": float(np.max(np.abs(residual))),
        "caption": f"PINN vs. FDM Concentration Profile Comparison at t = {t_fdm[t_idx]:.0f} hr",
    }


def fig_4_4_scatter_pred_vs_ref(model, config: dict, norm_stats: dict, sims: list[dict], max_points: int = 6000, seed: int = 0):
    """Fig 4.4 -- Scatter Plot: PINN Predicted vs. FDM Reference Concentrations,
    across the full held-out test set. R^2 is computed on every point; the
    scatter itself draws a random subsample of `max_points` for legibility."""
    all_pred, all_true = [], []
    for sim in sims:
        r, t, C_true = sim["r"], sim["t"], sim["C"]
        C_pred = predict_concentration_field(model, r, t, sim, norm_stats)
        all_pred.append(C_pred.ravel())
        all_true.append(C_true.ravel())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    metrics = compute_metrics(all_pred, all_true, config)

    rng = np.random.default_rng(seed)
    n_points = len(all_pred)
    if n_points > max_points:
        idx = rng.choice(n_points, size=max_points, replace=False)
        plot_pred, plot_true = all_pred[idx], all_true[idx]
    else:
        plot_pred, plot_true = all_pred, all_true

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(plot_true, plot_pred, s=4, alpha=0.25, color="#1c7c74", edgecolors="none")
    lo, hi = float(min(all_true.min(), all_pred.min())), float(max(all_true.max(), all_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y = x")
    ax.set_xlabel("FDM reference concentration (μM)")
    ax.set_ylabel("PINN predicted concentration (μM)")
    ax.text(0.05, 0.92, f"R² = {metrics['r2']:.4f}\nN = {n_points:,} points ({len(sims)} sims)", transform=ax.transAxes, fontsize=10, va="top")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    return fig, {
        "n_test_sims": len(sims),
        "n_points": n_points,
        "n_points_plotted": len(plot_pred),
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "caption": "PINN Predicted vs. FDM Reference Concentrations",
    }


def fig_4_5_training_loss(history: dict, config: dict):
    """Fig 4.5 -- Training Loss Convergence Curves, per component + total, log-y."""
    from src.train import HISTORY_KEYS

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, key in zip(axes.ravel(), HISTORY_KEYS):
        vals = np.asarray(history[key], dtype=float)
        if len(vals) and np.any(vals > 0):
            ax.semilogy(vals, color="#1c7c74", linewidth=1.5)
        else:
            ax.plot(vals, color="#1c7c74", linewidth=1.5)
        ax.set_title(f"{key} loss")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    fig.tight_layout()

    final_values = {key: float(history[key][-1]) if len(history[key]) else float("nan") for key in HISTORY_KEYS}
    return fig, {"n_logged_steps": len(history.get("total", [])), "final_losses": final_values, "caption": "Training Loss Convergence Curves"}


def fig_4_6_penetration_vs_time(model, config: dict, norm_stats: dict, n_r: int = 150, n_t: int = 60):
    """Fig 4.6 -- Drug Penetration Depth vs. Time for Five Nanoparticle Diameters."""
    params = baseline_params(config)
    diameters = config["paper"]["sweeps"]["d_NP_nm"]
    threshold_fraction = config["paper"]["penetration_threshold_fraction"]
    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    t = np.linspace(0.0, params["t_max_hr"], n_t)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(diameters)))
    final_depths = {}
    for i, d_NP in enumerate(diameters):
        sim_params = {**params, "d_NP_nm": d_NP}
        C = predict_concentration_field(model, r, t, sim_params, norm_stats)
        depth = penetration_depth_from_concentration(C, r, params["C0_uM"], threshold_fraction)
        ax.plot(t, depth, color=colors[i], linewidth=2, label=f"d_NP = {d_NP:g} nm")
        final_depths[d_NP] = float(depth[-1])

    ax.set_xlabel("Time t (hr)")
    ax.set_ylabel("Penetration depth (μm)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "params": params,
        "final_penetration_depth_um": final_depths,
        "caption": "Drug Penetration Depth vs. Time for Five Nanoparticle Diameters",
    }


def fig_4_7_viability_t72(model, config: dict, norm_stats: dict, n_r: int = 200, n_t: int = 60):
    """Fig 4.7 -- Spatial Cell Viability Distribution at t=72hr, baseline, three zones shaded."""
    params = baseline_params(config)
    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    t = np.linspace(0.0, params["t_max_hr"], n_t)
    C = predict_concentration_field(model, r, t, params, norm_stats)
    V = viability_map(C, t, config)[-1, :]

    oxygen = oxygen_gradient(r, config, R=params["R_um"])
    zones = assign_zones(r, oxygen, config)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for zone_name, color in ZONE_COLORS.items():
        mask = zones == zone_name
        if mask.any():
            ax.axvspan(r[mask].min(), r[mask].max(), color=color, alpha=0.12)
    ax.plot(r, V, color="#12232e", linewidth=2.5)
    for zone_name, color in ZONE_COLORS.items():
        mask = zones == zone_name
        if mask.any():
            mid = 0.5 * (r[mask].min() + r[mask].max())
            ax.text(mid, 102, ZONE_LABELS[zone_name], ha="center", fontsize=8, color=color)
    ax.set_xlabel("Radial distance r (μm)")
    ax.set_ylabel("Viability V (%)")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    zone_means = {z: float(V[zones == z].mean()) if (zones == z).any() else float("nan") for z in ZONE_COLORS}
    return fig, {
        "params": params,
        "zone_mean_viability_pct": zone_means,
        "caption": f"Spatial Cell Viability Distribution at t = {params['t_max_hr']:.0f} hr, Baseline Configuration",
    }


def fig_4_8_cytotoxicity_evolution(model, config: dict, norm_stats: dict, n_r: int = 150):
    """Fig 4.8 -- Cytotoxicity Evolution Maps at Five Time Points (small multiples
    of the radial cytotoxicity profile; the model is spherically-symmetric 1D-radial,
    so "map" here is the same r-only convention used throughout src/visualize.py)."""
    params = baseline_params(config)
    all_time_points = config["paper"]["sweeps"]["time_points_hr"]
    plot_time_points = [t for t in all_time_points if t > 0][:5]  # the five non-trivial points; t=0 is always 0

    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    t = np.array([0.0] + plot_time_points, dtype=float)  # t=0 included so the hazard integral starts correctly
    C = predict_concentration_field(model, r, t, params, norm_stats)
    Cyt = cytotoxicity_map(C, t, config) * 100.0

    fig, axes = plt.subplots(1, len(plot_time_points), figsize=(3.0 * len(plot_time_points), 4), sharey=True)
    for ax, hours in zip(axes, plot_time_points):
        idx = int(np.argmin(np.abs(t - hours)))
        ax.fill_between(r, 0, Cyt[idx, :], color="#b3462c", alpha=0.75)
        ax.set_title(f"t={hours:g}hr", fontsize=10)
        ax.set_xlabel("r (μm)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cytotoxicity (%)")
    fig.tight_layout()

    return fig, {"params": params, "time_points_hr": plot_time_points, "caption": "Cytotoxicity Evolution Maps at Five Time Points"}


def fig_4_9_hetero_vs_homog(config: dict, solved: dict | None = None):
    """Fig 4.9 -- Heterogeneous vs. Homogeneous D_eff Concentration Profile Comparison."""
    solved = solved or solve_hetero_and_homogeneous(config)
    p, hetero, homog = solved["params"], solved["hetero"], solved["homog"]
    r = hetero["r"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(r, hetero["C"][-1, :], color="#1c7c74", linewidth=2.5, label="Heterogeneous (3-zone) D_eff")
    ax.plot(r, homog["C"][-1, :], color="#b3462c", linestyle="--", linewidth=2.5, label="Homogeneous (mean) D_eff")
    ax.set_xlabel("Radial distance r (μm)")
    ax.set_ylabel("Concentration C (μM)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "params": p,
        "D_eff_homogeneous_um2_per_hr": solved["D_eff_homogeneous_um2_per_hr"],
        "caption": f"Heterogeneous vs. Homogeneous D_eff Concentration Profile Comparison (t = {p['t_max_hr']:.0f} hr)",
    }


def fig_4_10_effectiveness_surface(model, config: dict, norm_stats: dict, grid_result: dict | None = None):
    """Fig 4.10 -- Treatment Effectiveness Surface eta(d_NP, C0), baseline R, optimum marked."""
    params = baseline_params(config)
    grid_result = grid_result or grid_search_radius(model, params["R_um"], config, norm_stats)
    eta_grid, d_NP_grid, C0_grid = grid_result["eta_grid"], grid_result["d_NP_grid_nm"], grid_result["C0_grid_uM"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(C0_grid, d_NP_grid, eta_grid * 100.0, cmap="viridis", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("η — volume-averaged kill fraction (%)")
    ax.contour(C0_grid, d_NP_grid, eta_grid * 100.0, colors="white", linewidths=0.6, alpha=0.6)
    ax.scatter([grid_result["C0_star_uM"]], [grid_result["d_NP_star_nm"]], color="#b3462c", marker="*", s=260, edgecolors="white", linewidths=0.8, label="optimum", zorder=5)
    ax.set_xlabel("Surface concentration C0 (μM)")
    ax.set_ylabel("Nanoparticle diameter d_NP (nm)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    return fig, {
        "R_um": params["R_um"],
        "d_NP_star_nm": grid_result["d_NP_star_nm"],
        "C0_star_uM": grid_result["C0_star_uM"],
        "max_eta_pct": grid_result["max_eta"] * 100.0,
        "caption": f"Treatment Effectiveness Surface η(d_NP, C0) at R = {params['R_um']:.0f} μm",
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def table_4_1_fdm_summary(config: dict, diameter_sweep: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table 4.1 -- FDM Simulation Summary Statistics (Baseline)."""
    params = baseline_params(config)
    sweep = diameter_sweep or solve_diameter_sweep(config)
    baseline_fdm = _baseline_fdm_from_sweep(sweep, params, config)
    r, t, C, D_eff, dt_internal = baseline_fdm["r"], baseline_fdm["t"], baseline_fdm["C"], baseline_fdm["D_eff"], baseline_fdm["dt_internal"]
    dr = float(r[1] - r[0])
    cfl_number = float(D_eff.max() * dt_internal / dr**2)  # same ratio src/fdm_solver.py::check_cfl guards against

    depth_curve = penetration_depth_from_concentration(C, r, params["C0_uM"], config["paper"]["penetration_threshold_fraction"])
    rows = []
    for hours in config["paper"]["sweeps"]["time_points_hr"]:
        idx = int(np.argmin(np.abs(t - hours)))
        rows.append(
            {
                "Time (hr)": hours,
                "Max Penetration Depth (μm)": round(float(depth_curve[idx]), 1),
                "Center Conc. (μM)": round(float(C[idx, 0]), 4),
                "Mean Tumor Conc. (μM)": round(_volume_average(C[idx, :], r), 4),
                "CFL Number": round(cfl_number, 4),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"params": params, "cfl_number": cfl_number}


def table_4_2_penetration_vs_diameter(config: dict, diameter_sweep: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table 4.2 -- Maximum Penetration Depth at t=72hr Across Nanoparticle Diameters."""
    params = baseline_params(config)
    sweep = diameter_sweep or solve_diameter_sweep(config)
    const = config["constants"]
    f_rim = config["microenvironment"]["f_zone"]["proliferating_rim"]

    rows = []
    for d_NP, fdm in sorted(sweep.items()):
        r, t, C = fdm["r"], fdm["t"], fdm["C"]
        D_free_um2_hr = stokes_einstein_diffusivity(d_NP, const["T"], const["eta"], const["k_B"]) * 1e12 * 3600.0
        D_eff_rim = D_free_um2_hr * f_rim
        depth_final = penetration_depth_from_concentration(C, r, params["C0_uM"], config["paper"]["penetration_threshold_fraction"])[-1]
        rows.append(
            {
                "d_NP (nm)": d_NP,
                "D_free (μm²/hr)": round(D_free_um2_hr, 1),
                "D_eff,rim (μm²/hr)": round(D_eff_rim, 1),
                "Max Pen. Depth (μm)": round(float(depth_final), 1),
                "Fraction of Tumor Reached (%)": round(float(depth_final) / params["R_um"] * 100.0, 1),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"params": params}


def table_4_3_pinn_metrics(model, config: dict, norm_stats: dict, sims: list[dict], collocation_X: torch.Tensor) -> tuple[pd.DataFrame, dict]:
    """Table 4.3 -- PINN Test Set Evaluation Metrics (Mean +/- SD, N test simulations).

    RMSE/MAE/R^2/L2-relative-error/penetration-RMSE are computed per test
    simulation, then reported as mean +/- SD across sims (matching the
    manuscript's "N=300 test simulations" framing). The PDE residual is not
    tied to individual simulations by the saved collocation tensors (see
    module docstring in src/data_pipeline.py), so its mean +/- SD is instead
    taken over the population of per-point residual magnitudes across the
    test set's collocation batch -- an honest population statistic, noted
    here rather than silently presented as a per-simulation aggregate.
    """
    threshold_fraction = config["paper"]["penetration_threshold_fraction"]
    per_sim = []
    for sim in sims:
        r, t, C_true = sim["r"], sim["t"], sim["C"]
        C_pred = predict_concentration_field(model, r, t, sim, norm_stats)
        m = compute_metrics(C_pred, C_true, config)
        pen_pred = penetration_depth_from_concentration(C_pred, r, sim["C0_uM"], threshold_fraction)
        pen_true = penetration_depth_from_concentration(C_true, r, sim["C0_uM"], threshold_fraction)
        m["penetration_rmse_um"] = float(np.sqrt(np.mean((pen_pred - pen_true) ** 2)))
        per_sim.append(m)
    per_sim_df = pd.DataFrame(per_sim)

    residual_stats = compute_pde_residual_stats(model, collocation_X, config, norm_stats)
    pooled = evaluate_test_set(model, sims, norm_stats, config)["global"]

    thresholds = config["evaluation"]["thresholds"]
    metric_specs = [
        ("RMSE (μM)", per_sim_df["rmse"], thresholds["rmse_uM"], "less"),
        ("MAE (μM)", per_sim_df["mae"], thresholds["mae_uM"], "less"),
        ("R²", per_sim_df["r2"], thresholds["r2"], "greater"),
        ("L2 Relative Error", per_sim_df["l2_relative"], thresholds["l2_relative_error"], "less"),
        ("Mean PDE Residual", pd.Series(residual_stats["residuals"]), thresholds["mean_pde_residual"], "less"),
        ("Penetration RMSE (μm)", per_sim_df["penetration_rmse_um"], thresholds["penetration_rmse_um"], "less"),
    ]
    rows = []
    for name, series, threshold, direction in metric_specs:
        mean_val, std_val = float(series.mean()), float(series.std())
        passed = mean_val < threshold if direction == "less" else mean_val > threshold
        rows.append(
            {
                "Metric": name,
                "Value (mean ± SD)": f"{mean_val:.4g} ± {std_val:.4g}",
                "Mean": mean_val,
                "SD": std_val,
                "Publication Threshold": threshold,
                "Hypothesis Outcome": "Pass" if passed else "Fail",
            }
        )
    df = pd.DataFrame(rows)

    h1_pass = bool(pooled["r2"] > thresholds["r2"] and pooled["rmse"] < config["evaluation"]["hypotheses"]["H1_rmse_target_uM"])
    meta = {
        "n_test_sims": len(sims),
        "n_collocation_points": int(len(residual_stats["residuals"])),
        "pooled_global_metrics": pooled,
        "H1_pass_on_pooled_metrics": h1_pass,
    }
    return df, meta


def table_4_4_penetration_analysis(model, config: dict, norm_stats: dict, diameter_sweep: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table 4.4 -- Penetration Depth Analysis Across Nanoparticle Sizes.

    Pen. depth and tumor-volume-reached columns are PINN-predicted (the
    pharmacodynamic module's own input, per the Methodology chapter);
    "PINN RMSE" compares that PINN-predicted penetration-depth-vs-time curve
    against the FDM reference curve for the same configuration.
    """
    params = baseline_params(config)
    sweep = diameter_sweep or solve_diameter_sweep(config)
    threshold_fraction = config["paper"]["penetration_threshold_fraction"]

    rows = []
    for d_NP, fdm in sorted(sweep.items()):
        r, t, C_true = fdm["r"], fdm["t"], fdm["C"]
        sim_params = {**params, "d_NP_nm": d_NP}
        C_pred = predict_concentration_field(model, r, t, sim_params, norm_stats)

        pen_pred = penetration_depth_from_concentration(C_pred, r, params["C0_uM"], threshold_fraction)
        pen_true = penetration_depth_from_concentration(C_true, r, params["C0_uM"], threshold_fraction)
        rmse = float(np.sqrt(np.mean((pen_pred - pen_true) ** 2)))

        idx_24 = int(np.argmin(np.abs(t - 24.0)))
        idx_72 = int(np.argmin(np.abs(t - params["t_max_hr"])))
        rows.append(
            {
                "d_NP (nm)": d_NP,
                "Pen. Depth 24hr (μm)": round(float(pen_pred[idx_24]), 1),
                "Pen. Depth 72hr (μm)": round(float(pen_pred[idx_72]), 1),
                "Tumor Volume Reached (%)": round(float(pen_pred[idx_72]) / params["R_um"] * 100.0, 1),
                "PINN RMSE (μm)": round(rmse, 2),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"params": params}


def table_4_5_viability_summary(model, config: dict, norm_stats: dict, n_r: int = 150, n_t: int = 60) -> tuple[pd.DataFrame, dict]:
    """Table 4.5 -- Viability Analysis Summary at t=72hr."""
    params = baseline_params(config)
    r = radial_grid(params["R_um"], n_r, config["fdm"]["r_min_um"])
    oxygen = oxygen_gradient(r, config, R=params["R_um"])
    zones = assign_zones(r, oxygen, config)

    rows = []
    for d_NP in config["paper"]["sweeps"]["d_NP_nm"]:
        sim_params = {**params, "d_NP_nm": d_NP}
        eta = kill_fraction(model, params["R_um"], d_NP, params["C0_uM"], config, norm_stats, params["k_d_per_hr"], params["t_max_hr"])
        resistance = _resistance_map(model, params["R_um"], d_NP, params["C0_uM"], config, norm_stats, params["k_d_per_hr"], params["t_max_hr"])

        t = np.linspace(0.0, params["t_max_hr"], n_t)
        C = predict_concentration_field(model, r, t, sim_params, norm_stats)
        V_final = viability_map(C, t, config)[-1, :]

        def zone_mean(zone_name: str) -> float:
            mask = zones == zone_name
            return float(V_final[mask].mean()) if mask.any() else float("nan")

        rows.append(
            {
                "d_NP (nm)": d_NP,
                "Outer Rim Viability (%)": round(zone_mean("proliferating_rim"), 1),
                "Quiescent Zone Viability (%)": round(zone_mean("quiescent_zone"), 1),
                "Core Viability (%)": round(zone_mean("necrotic_core"), 1),
                "Overall Kill Fraction (%)": round(eta * 100.0, 1),
                "Resistance Risk Fraction (%)": round(resistance["resistant_volume_fraction"] * 100.0, 1),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"params": params}


def table_4_6_hetero_vs_homog(config: dict, solved: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table 4.6 -- Heterogeneous vs. Homogeneous D_eff Model Comparison."""
    solved = solved or solve_hetero_and_homogeneous(config)
    hetero, homog = solved["hetero"], solved["homog"]
    het_summary = _summarize_field(hetero["C"], hetero["r"], hetero["t"], config)
    hom_summary = _summarize_field(homog["C"], homog["r"], homog["t"], config)

    row_specs = [
        ("Max penetration depth (μm)", "max_penetration_depth_um", 1),
        ("Sub-therapeutic zone radius (μm)", "subtherapeutic_zone_radius_um", 1),
        ("Resistance risk fraction (%)", "resistance_risk_fraction", 1, 100.0),
        ("Overall kill fraction (%)", "kill_fraction", 1, 100.0),
    ]
    rows = []
    for spec in row_specs:
        label, key, ndigits = spec[0], spec[1], spec[2]
        scale = spec[3] if len(spec) > 3 else 1.0
        het_val, hom_val = het_summary[key] * scale, hom_summary[key] * scale
        diff_pct = (het_val - hom_val) / hom_val * 100.0 if hom_val else float("nan")
        rows.append(
            {
                "Parameter": label,
                "Heterogeneous D_eff Model": round(het_val, ndigits),
                "Homogeneous D_eff Model": round(hom_val, ndigits),
                "Difference (%)": round(diff_pct, 1),
            }
        )
    df = pd.DataFrame(rows)

    subzone_increase_pct = float(df.loc[df["Parameter"].str.contains("Sub-therapeutic"), "Difference (%)"].iloc[0])
    steeper_gradient = het_summary["avg_concentration_gradient_uM_per_um"] > hom_summary["avg_concentration_gradient_uM_per_um"]
    threshold_pct = config["paper"]["hypotheses"]["H3_subtherapeutic_zone_increase_pct"]
    h3_pass = bool(steeper_gradient and subzone_increase_pct >= threshold_pct)

    meta = {
        "params": solved["params"],
        "subtherapeutic_zone_increase_pct": subzone_increase_pct,
        "steeper_gradient_heterogeneous": bool(steeper_gradient),
        "H3_threshold_pct": threshold_pct,
        "H3_pass": h3_pass,
    }
    return df, meta


def table_4_7_optimization_results(model, config: dict, norm_stats: dict, radius_results: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table 4.7 -- Optimization Results: Optimal Nanoparticle Parameters for Different Tumor Sizes."""
    radius_results = radius_results or optimize_all_radii(model, config, norm_stats)
    rows = []
    for R_um, result in sorted(radius_results.items()):
        rows.append(
            {
                "Tumor Radius (μm)": R_um,
                "Optimal d_NP* (nm)": round(result["d_NP_star_nm"], 1),
                "Optimal C0* (μM)": round(result["C0_star_uM"], 2),
                "Maximum η (%)": round(result["max_eta"] * 100.0, 1),
                "Computation Time (sec)": round(result["computation_time_s"], 3),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"radii_um": list(radius_results.keys())}


def table_4_8_ablation(
    model, baseline_model, config: dict, norm_stats: dict, sims: list[dict], collocation_X: torch.Tensor
) -> tuple[pd.DataFrame, dict]:
    """Table 4.8 -- Ablation Study: PINN vs. Unconstrained NN Baseline."""
    biopinn_global = evaluate_test_set(model, sims, norm_stats, config)["global"]
    baseline_global = evaluate_test_set(baseline_model, sims, norm_stats, config)["global"]
    rc = compare_residuals(model, baseline_model, collocation_X, config, norm_stats)
    wx = wilcoxon_test(rc["residuals_biopinn"], rc["residuals_baseline"])

    def improvement(base, biopinn, lower_is_better=True):
        if biopinn == 0:
            return float("inf")
        ratio = base / biopinn if lower_is_better else biopinn / base
        return ratio

    rows = [
        {"Metric": "RMSE (μM)", "BIOPINN": round(biopinn_global["rmse"], 4), "Baseline": round(baseline_global["rmse"], 4),
         "Improvement Factor": round(improvement(baseline_global["rmse"], biopinn_global["rmse"]), 2)},
        {"Metric": "R²", "BIOPINN": round(biopinn_global["r2"], 4), "Baseline": round(baseline_global["r2"], 4),
         "Improvement Factor": round(biopinn_global["r2"] / baseline_global["r2"], 3) if baseline_global["r2"] else float("nan")},
        {"Metric": "Mean PDE Residual", "BIOPINN": rc["biopinn"]["mean_abs_residual"], "Baseline": rc["baseline"]["mean_abs_residual"],
         "Improvement Factor": round(rc["improvement_factor_mean"], 1)},
        {"Metric": "Max PDE Residual", "BIOPINN": rc["biopinn"]["max_abs_residual"], "Baseline": rc["baseline"]["max_abs_residual"],
         "Improvement Factor": round(rc["improvement_factor_max"], 1)},
        {"Metric": "Physical consistency (%)", "BIOPINN": round(rc["biopinn"]["physical_consistency_pct"], 1),
         "Baseline": round(rc["baseline"]["physical_consistency_pct"], 1),
         "Improvement Factor": round(rc["biopinn"]["physical_consistency_pct"] - rc["baseline"]["physical_consistency_pct"], 1)},
    ]
    df = pd.DataFrame(rows)

    hyp_cfg = config["evaluation"]["hypotheses"]
    h5_pass = bool(rc["improvement_factor_mean"] >= hyp_cfg["H5_residual_improvement_factor"] and wx["significant"])
    meta = {
        "wilcoxon_p_value": wx["p_value"],
        "wilcoxon_significant": wx["significant"],
        "improvement_factor_mean": rc["improvement_factor_mean"],
        "H5_target_improvement_factor": hyp_cfg["H5_residual_improvement_factor"],
        "H5_pass": h5_pass,
        "n_test_sims": len(sims),
    }
    return df, meta


def table_4_9_hypothesis_summary(hypotheses: dict) -> pd.DataFrame:
    """Table 4.9 -- Hypothesis Evaluation Summary. `hypotheses` is the dict
    assembled by scripts/generate_results.py from the H1-H6 checks computed
    across the tables above (each entry: pass, expected, evidence, actual)."""
    rows = []
    for key in ("H1", "H2", "H3", "H4", "H5", "H6"):
        h = hypotheses[key]
        rows.append(
            {
                "Hypothesis": f"{key} — {h['name']}",
                "Expected Outcome": h["expected"],
                "Evidence Basis": h["evidence"],
                "Conclusion": "Supported" if h["pass"] else "Not supported",
            }
        )
    return pd.DataFrame(rows)
