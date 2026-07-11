"""Surrogate-based optimization + efficiency study.

Uses the trained PINN as a fast, differentiable-free surrogate objective:
grid search over (d_NP, C0) at R in {200,300,400,500} um to maximize the
volume-averaged tumor kill fraction eta (SO7), the homogeneous-vs-
heterogeneous diffusion comparison (H3), and the PINN-vs-FDM speedup study
over held-out parameter combinations (H6). Local module: loads
artifacts/biopinn_model.pt, never retrains (only src/ablation.py trains
anything locally).
"""

from __future__ import annotations

import time

import numpy as np

from src.biology import cytotoxicity_map, penetration_depth, predict_concentration_field
from src.data_pipeline import latin_hypercube_sample
from src.fdm_solver import solve_fdm
from src.microenvironment import radial_grid, stokes_einstein_diffusivity

DEFAULT_K_D_PER_HR = 0.01
DEFAULT_T_MAX_HR = 72.0
SUBTHERAPEUTIC_THRESHOLD_FACTOR = 0.1  # fraction of IC50, matching the guide's resistance-zone convention


def _volume_average(values: np.ndarray, r: np.ndarray) -> float:
    """Volume-weighted average over a sphere: dV proportional to r^2 dr."""
    weights = r**2
    return float(np.sum(values * weights) / np.sum(weights))


def kill_fraction(
    model,
    R_um: float,
    d_NP_nm: float,
    C0_uM: float,
    config: dict,
    norm_stats: dict,
    k_d_per_hr: float = DEFAULT_K_D_PER_HR,
    t_max_hr: float = DEFAULT_T_MAX_HR,
    N_t: int = 100,
) -> float:
    """Volume-averaged tumor kill fraction eta at t_max_hr for one parameter
    combination, from the PINN surrogate. Uses N_t>2 time points so the
    survival-fraction hazard integral (src.biology.survival_fraction) is
    actually resolved over [0, t_max_hr] rather than collapsing to zero
    kill from a single, too-coarse step."""
    fdm_cfg = config["fdm"]
    r = radial_grid(R_um, fdm_cfg["N_r"], fdm_cfg["r_min_um"])
    t = np.linspace(0.0, t_max_hr, N_t)

    sim_params = {"R_um": R_um, "d_NP_nm": d_NP_nm, "C0_uM": C0_uM, "k_d_per_hr": k_d_per_hr, "t_max_hr": t_max_hr}
    C_rt = predict_concentration_field(model, r, t, sim_params, norm_stats)
    cytotox = cytotoxicity_map(C_rt, t, config)
    return _volume_average(cytotox[-1, :], r)


def _resistance_map(
    model,
    R_um: float,
    d_NP_nm: float,
    C0_uM: float,
    config: dict,
    norm_stats: dict,
    k_d_per_hr: float,
    t_max_hr: float,
    N_t: int = 100,
) -> dict:
    """Sub-therapeutic (resistance-risk) zone map at t_max_hr: which radii
    never reach subtherapeutic_threshold_factor*IC50, and what volume
    fraction of the tumor that represents."""
    fdm_cfg = config["fdm"]
    r = radial_grid(R_um, fdm_cfg["N_r"], fdm_cfg["r_min_um"])
    t = np.linspace(0.0, t_max_hr, N_t)

    sim_params = {"R_um": R_um, "d_NP_nm": d_NP_nm, "C0_uM": C0_uM, "k_d_per_hr": k_d_per_hr, "t_max_hr": t_max_hr}
    C_rt = predict_concentration_field(model, r, t, sim_params, norm_stats)
    C_final = C_rt[-1, :]

    threshold = SUBTHERAPEUTIC_THRESHOLD_FACTOR * config["biology"]["IC50_uM"]
    resistant_mask = C_final < threshold

    return {
        "r_um": r,
        "C_final_uM": C_final,
        "resistant_mask": resistant_mask,
        "resistant_volume_fraction": _volume_average(resistant_mask.astype(float), r),
        "threshold_uM": threshold,
    }


def grid_search_radius(
    model,
    R_um: float,
    config: dict,
    norm_stats: dict,
    k_d_per_hr: float = DEFAULT_K_D_PER_HR,
    t_max_hr: float = DEFAULT_T_MAX_HR,
) -> dict:
    """Grid search (d_NP, C0) at a fixed radius to maximize eta.

    Returns the optimal combo, max eta, the full eta surface (for later
    plotting), the resistance-zone map at the optimum, and the
    per-configuration computation time.
    """
    opt_cfg = config["optimization"]
    d_lo, d_hi = opt_cfg["d_NP_grid_nm"]
    c_lo, c_hi = opt_cfg["C0_grid_uM"]
    d_NP_grid = np.linspace(d_lo, d_hi, opt_cfg["n_d_NP_points"])
    C0_grid = np.linspace(c_lo, c_hi, opt_cfg["n_C0_points"])

    t0 = time.perf_counter()
    eta_grid = np.zeros((len(d_NP_grid), len(C0_grid)))
    for i, d_NP in enumerate(d_NP_grid):
        for j, C0 in enumerate(C0_grid):
            eta_grid[i, j] = kill_fraction(model, R_um, d_NP, C0, config, norm_stats, k_d_per_hr, t_max_hr)
    elapsed = time.perf_counter() - t0

    best_i, best_j = np.unravel_index(np.argmax(eta_grid), eta_grid.shape)
    d_NP_star, C0_star = float(d_NP_grid[best_i]), float(C0_grid[best_j])
    max_eta = float(eta_grid[best_i, best_j])

    resistance_map = _resistance_map(model, R_um, d_NP_star, C0_star, config, norm_stats, k_d_per_hr, t_max_hr)

    return {
        "R_um": R_um,
        "d_NP_grid_nm": d_NP_grid,
        "C0_grid_uM": C0_grid,
        "eta_grid": eta_grid,
        "d_NP_star_nm": d_NP_star,
        "C0_star_uM": C0_star,
        "max_eta": max_eta,
        "resistance_map": resistance_map,
        "computation_time_s": elapsed,
        "n_combinations": int(eta_grid.size),
    }


def optimize_all_radii(
    model,
    config: dict,
    norm_stats: dict,
    k_d_per_hr: float = DEFAULT_K_D_PER_HR,
    t_max_hr: float = DEFAULT_T_MAX_HR,
) -> dict:
    """Run grid_search_radius for each radius in optimization.radii_um."""
    return {
        R_um: grid_search_radius(model, R_um, config, norm_stats, k_d_per_hr, t_max_hr)
        for R_um in config["optimization"]["radii_um"]
    }


def _summarize_field(C_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict) -> dict:
    """Max penetration depth, sub-therapeutic zone radius, resistance-risk
    fraction, and overall kill fraction for one solved concentration field."""
    cytotox = cytotoxicity_map(C_rt, t, config)
    pen_depth = penetration_depth(cytotox, r, config)

    C_final = C_rt[-1, :]
    threshold = SUBTHERAPEUTIC_THRESHOLD_FACTOR * config["biology"]["IC50_uM"]
    subtherapeutic_mask = C_final < threshold
    subtherapeutic_indices = np.where(subtherapeutic_mask)[0]
    # Outer edge of the under-dosed core (0 if the whole tumor is therapeutic).
    subtherapeutic_zone_radius = float(r[subtherapeutic_indices[-1]]) if len(subtherapeutic_indices) else 0.0

    # Average concentration gradient magnitude (uM/um) from surface to
    # center at t_max, as an explicit "steepness" measure for H3.
    avg_gradient = float(abs(C_rt[-1, -1] - C_rt[-1, 0]) / r[-1])

    return {
        "max_penetration_depth_um": float(pen_depth[-1]),
        "subtherapeutic_zone_radius_um": subtherapeutic_zone_radius,
        "resistance_risk_fraction": _volume_average(subtherapeutic_mask.astype(float), r),
        "kill_fraction": _volume_average(cytotox[-1, :], r),
        "avg_concentration_gradient_uM_per_um": avg_gradient,
    }


def homogeneous_vs_heterogeneous(
    config: dict,
    d_NP_nm: float = 100.0,
    R_um: float = 400.0,
    C0_uM: float = 10.0,
    k_d_per_hr: float = DEFAULT_K_D_PER_HR,
    t_max_hr: float = DEFAULT_T_MAX_HR,
) -> dict:
    """H3: re-solve the FDM diffusion model with a homogeneous D_eff (the
    arithmetic mean of the three zone D_eff values) and compare against the
    normal three-zone heterogeneous model at a fixed case. This is a physics
    comparison (re-solving the PDE, not querying the trained PINN), matching
    the spec's literal "re-run the diffusion model"."""
    hetero_result = solve_fdm(R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr, config)

    const = config["constants"]
    D_free_m2_s = stokes_einstein_diffusivity(d_NP_nm, const["T"], const["eta"], const["k_B"])
    D_free_um2_hr = D_free_m2_s * 1e12 * 3600.0
    D_eff_homogeneous = D_free_um2_hr * float(np.mean(list(config["microenvironment"]["f_zone"].values())))

    homog_result = solve_fdm(
        R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr, config, D_eff_override=D_eff_homogeneous
    )

    hetero_summary = _summarize_field(hetero_result["C"], hetero_result["r"], hetero_result["t"], config)
    homog_summary = _summarize_field(homog_result["C"], homog_result["r"], homog_result["t"], config)

    h3_pass = (
        hetero_summary["avg_concentration_gradient_uM_per_um"]
        > homog_summary["avg_concentration_gradient_uM_per_um"]
        and hetero_summary["subtherapeutic_zone_radius_um"] > homog_summary["subtherapeutic_zone_radius_um"]
    )

    return {
        "parameters": {"d_NP_nm": d_NP_nm, "R_um": R_um, "C0_uM": C0_uM, "k_d_per_hr": k_d_per_hr, "t_max_hr": t_max_hr},
        "D_eff_homogeneous_um2_per_hr": D_eff_homogeneous,
        "heterogeneous": hetero_summary,
        "homogeneous": homog_summary,
        "H3": {
            "pass": bool(h3_pass),
            "description": "heterogeneous model shows a steeper gradient and larger sub-therapeutic zone than homogeneous",
        },
    }


def speedup_study(
    model,
    config: dict,
    norm_stats: dict,
    n_combinations: int | None = None,
    seed: int = 123,
) -> dict:
    """H6: per-combination computation time for the PINN surrogate vs. the
    FDM solver, across n_combinations LHS-sampled parameter combinations."""
    n_combinations = n_combinations or config["optimization"]["speedup_study"]["n_combinations"]
    params = latin_hypercube_sample(n_combinations, config, seed)
    fdm_cfg = config["fdm"]

    pinn_times = np.zeros(n_combinations)
    fdm_times = np.zeros(n_combinations)

    for idx in range(n_combinations):
        R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr = params[idx]

        r = radial_grid(R_um, fdm_cfg["N_r"], fdm_cfg["r_min_um"])
        t = np.linspace(0.0, t_max_hr, fdm_cfg["N_t_initial"])
        sim_params = {"R_um": R_um, "d_NP_nm": d_NP_nm, "C0_uM": C0_uM, "k_d_per_hr": k_d_per_hr, "t_max_hr": t_max_hr}

        t0 = time.perf_counter()
        predict_concentration_field(model, r, t, sim_params, norm_stats)
        pinn_times[idx] = time.perf_counter() - t0

        t0 = time.perf_counter()
        solve_fdm(R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr, config)
        fdm_times[idx] = time.perf_counter() - t0

    speedup_ratios = fdm_times / pinn_times
    hyp_cfg = config["evaluation"]["hypotheses"]
    target_speedup = 10.0 ** hyp_cfg["H6_speedup_orders_of_magnitude"]

    return {
        "n_combinations": n_combinations,
        "pinn_time_mean_s": float(pinn_times.mean()),
        "pinn_time_std_s": float(pinn_times.std()),
        "fdm_time_mean_s": float(fdm_times.mean()),
        "fdm_time_std_s": float(fdm_times.std()),
        "speedup_ratio_mean": float(speedup_ratios.mean()),
        "speedup_ratio_median": float(np.median(speedup_ratios)),
        "speedup_ratio_std": float(speedup_ratios.std()),
        "H6": {
            "pass": bool(speedup_ratios.mean() >= target_speedup),
            "target_speedup": target_speedup,
            "target_orders_of_magnitude": hyp_cfg["H6_speedup_orders_of_magnitude"],
        },
    }
