"""Pharmacodynamic / biological response module.

Converts a predicted (or FDM-reference) drug concentration field C(r,t) into
biological outcomes: the Hill-equation drug-induced death rate, survival
fraction f_alive(r,t) via the time-integrated hazard, viability V(r,t) and
cytotoxicity Cyt(r,t) maps, penetration depth, and the logistic tumor-growth
rate. Also provides the glue to go straight from a loaded BIOPINN checkpoint
to these maps (predict_concentration_field / compute_biological_response),
and an H4-hypothesis check (rim viability < 20%, core viability > 60% at
t=72hr). Local module: consumes a loaded checkpoint, never retrains.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data_pipeline import PARAM_ORDER
from src.microenvironment import assign_zones, oxygen_gradient


def hill_death_rate(C: np.ndarray, config: dict) -> np.ndarray:
    """delta(C) = delta_max * C^n / (IC50^n + C^n), delta_max converted to /hr."""
    bio = config["biology"]
    delta_max_per_hr = bio["delta_max_per_day"] / 24.0
    IC50 = bio["IC50_uM"]
    n = bio["n_hill"]

    C_safe = np.maximum(np.asarray(C, dtype=float), 0.0)
    return delta_max_per_hr * C_safe**n / (IC50**n + C_safe**n)


def survival_fraction(C_rt: np.ndarray, t: np.ndarray, config: dict) -> np.ndarray:
    """f_alive(r,t) = exp(-integral_0^t delta(C(r,t')) dt'), shape [N_t, N_r].

    Uses a left-Riemann cumulative sum of delta*dt (exact for delta constant
    within each step, and -- unlike the linearized (1-delta*dt) product form
    -- always stays in (0, 1] regardless of step size).
    """
    delta_rt = hill_death_rate(C_rt, config)
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)

    kill_integral = np.zeros_like(C_rt, dtype=float)
    if len(dt) > 0:
        kill_integral[1:, :] = np.cumsum(delta_rt[:-1, :] * dt[:, None], axis=0)
    return np.exp(-kill_integral)


def viability_map(C_rt: np.ndarray, t: np.ndarray, config: dict) -> np.ndarray:
    """V(r,t) = f_alive * 100 (percent)."""
    return survival_fraction(C_rt, t, config) * 100.0


def cytotoxicity_map(C_rt: np.ndarray, t: np.ndarray, config: dict) -> np.ndarray:
    """Cyt(r,t) = 1 - f_alive."""
    return 1.0 - survival_fraction(C_rt, t, config)


def penetration_depth(cyt_rt: np.ndarray, r: np.ndarray, config: dict) -> np.ndarray:
    """Depth (um) the cytotoxic effect has penetrated inward from the tumor
    surface at each time step: R minus the innermost radius still above the
    cytotoxicity threshold (0 if no point exceeds it)."""
    threshold = config["biology"]["penetration_cytotoxicity_threshold"]
    R = r[-1]

    N_t = cyt_rt.shape[0]
    depths = np.zeros(N_t)
    for n in range(N_t):
        above = np.where(cyt_rt[n, :] > threshold)[0]
        depths[n] = R - r[above[0]] if len(above) > 0 else 0.0
    return depths


def logistic_growth(N: np.ndarray, k_p_per_hr: float, N_max: float) -> np.ndarray:
    """growth = k_p * N * (1 - N/N_max)."""
    return k_p_per_hr * N * (1.0 - N / N_max)


def predict_concentration_field(
    model: torch.nn.Module, r: np.ndarray, t: np.ndarray, sim_params: dict, norm_stats: dict
) -> np.ndarray:
    """Query a trained BIOPINN model for C(r,t) over a full grid, for one
    physical parameter combination.

    Args:
        model: trained BIOPINN (put in eval mode internally).
        r: radial grid (um), shape [N_r].
        t: time grid (hr), shape [N_t].
        sim_params: dict with R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr.
        norm_stats: normalization stats (src.data_pipeline.PARAM_ORDER keys).

    Returns:
        C_rt: concentration field (uM), shape [N_t, N_r].
    """
    model.eval()
    R_um, C0_uM, t_max_hr = sim_params["R_um"], sim_params["C0_uM"], sim_params["t_max_hr"]

    def normalize(key: str) -> float:
        lo, hi = norm_stats[key]["min"], norm_stats[key]["max"]
        value = sim_params[key]
        return 0.0 if hi <= lo else (value - lo) / (hi - lo)

    param_row = torch.tensor([normalize(k) for k in PARAM_ORDER], dtype=torch.float32)

    N_r, N_t = len(r), len(t)
    r_norm = torch.as_tensor(np.asarray(r) / R_um, dtype=torch.float32)
    t_norm = torch.as_tensor(np.asarray(t) / t_max_hr, dtype=torch.float32)

    r_col = r_norm.reshape(1, N_r).expand(N_t, N_r).reshape(-1, 1)
    t_col = t_norm.reshape(N_t, 1).expand(N_t, N_r).reshape(-1, 1)
    param_cols = param_row.unsqueeze(0).expand(N_t * N_r, -1)
    X = torch.cat([r_col, t_col, param_cols], dim=1)

    with torch.no_grad():
        C_norm = model(X).numpy().reshape(N_t, N_r)
    return C_norm * C0_uM


def compute_biological_response(
    model: torch.nn.Module, r: np.ndarray, t: np.ndarray, sim_params: dict, norm_stats: dict, config: dict
) -> dict:
    """End-to-end: query a loaded checkpoint for C(r,t), then derive viability,
    cytotoxicity, and penetration depth."""
    C_rt = predict_concentration_field(model, r, t, sim_params, norm_stats)
    viability = viability_map(C_rt, t, config)
    cytotoxicity = cytotoxicity_map(C_rt, t, config)
    pen_depth = penetration_depth(cytotoxicity, r, config)
    return {"C": C_rt, "viability": viability, "cytotoxicity": cytotoxicity, "penetration_depth": pen_depth}


def evaluate_h4_hypothesis(
    viability_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict, R_um: float | None = None
) -> dict:
    """H4: at t~=72hr, proliferating-rim viability < rim_max and necrotic-core
    viability > core_min (configs/default_config.yaml -> biology.h4_*)."""
    bio = config["biology"]
    t_idx = int(np.argmin(np.abs(np.asarray(t) - bio["h4_time_hr"])))

    oxygen = oxygen_gradient(r, config, R=R_um)
    zones = assign_zones(r, oxygen, config)
    viability_at_t = viability_rt[t_idx, :]

    rim_mask = zones == "proliferating_rim"
    core_mask = zones == "necrotic_core"

    rim_viability = float(viability_at_t[rim_mask].mean()) if rim_mask.any() else float("nan")
    core_viability = float(viability_at_t[core_mask].mean()) if core_mask.any() else float("nan")

    rim_pass = bool(rim_mask.any() and rim_viability < bio["h4_rim_viability_max"])
    core_pass = bool(core_mask.any() and core_viability > bio["h4_core_viability_min"])

    return {
        "t_hr": float(t[t_idx]),
        "rim_viability_pct": rim_viability,
        "core_viability_pct": core_viability,
        "rim_pass": rim_pass,
        "core_pass": core_pass,
        "overall_pass": rim_pass and core_pass,
    }
