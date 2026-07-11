"""Six-metric evaluation against publication thresholds.

Loads a trained checkpoint + the test split's exact simulation parameters
(paths.processed/sim_params.json, written by src.data_pipeline.build_dataset),
deterministically re-solves each test simulation's full FDM reference field
(src.fdm_solver.solve_fdm), queries the PINN over the same (r,t) grid
(src.biology.predict_concentration_field), and computes all six metrics --
RMSE, MAE, R^2, L2 relative error, mean PDE residual, penetration RMSE --
globally and decomposed by tumor zone / nanoparticle-size range / time range,
plus the H1/H2/H4 hypothesis checks. (H3/H5/H6 are src/optimize.py's and
src/ablation.py's -- run_evaluation.py's summary composes with those.)

Local module: loads artifacts/biopinn_model.pt + paths.processed, never
retrains.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from src.biology import evaluate_h4_hypothesis, predict_concentration_field, viability_map
from src.config import resolve_path
from src.data_pipeline import PARAM_ORDER, _solve_one
from src.losses import pde_residual
from src.microenvironment import assign_zones, oxygen_gradient, radial_grid

TIME_BIN_LABELS = ("early_0_24h", "mid_24_48h", "late_48_72h")
NP_SIZE_BIN_LABELS = ("small", "medium", "large")


def compute_metrics(C_pred: np.ndarray, C_true: np.ndarray, config: dict) -> dict:
    """RMSE, MAE, R^2, L2 relative error over a flat set of predictions."""
    C_pred = np.asarray(C_pred, dtype=float).ravel()
    C_true = np.asarray(C_true, dtype=float).ravel()

    err = C_pred - C_true
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")

    ss_res = np.sum(err**2)
    ss_tot = np.sum((C_true - C_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    l2_den = np.sqrt(np.sum(C_true**2))
    l2_relative = float(np.sqrt(ss_res) / l2_den) if l2_den > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "l2_relative": l2_relative, "n_points": int(len(C_pred))}


def compute_pde_residual_stats(
    model: torch.nn.Module, collocation_X: torch.Tensor, config: dict, norm_stats: dict
) -> dict:
    """Mean/max |PDE residual| over a batch of collocation points, plus the
    raw per-point residuals for a histogram."""
    residual = pde_residual(model, collocation_X, config, norm_stats).detach().cpu().numpy().ravel()
    abs_residual = np.abs(residual)
    return {
        "mean_abs_residual": float(np.mean(abs_residual)),
        "max_abs_residual": float(np.max(abs_residual)),
        "residuals": abs_residual,
    }


def penetration_depth_from_concentration(
    C_rt: np.ndarray, r: np.ndarray, C0_uM: float, threshold_fraction: float = 0.1
) -> np.ndarray:
    """Depth (um) the drug front has penetrated inward from the tumor
    surface at each time step: R minus the innermost radius where C exceeds
    threshold_fraction*C0 (0 if no point exceeds it). Concentration-based
    (not the cytotoxicity-based src.biology.penetration_depth), matching the
    metric this module's "penetration RMSE" evaluates."""
    threshold = threshold_fraction * C0_uM
    R = r[-1]
    N_t = C_rt.shape[0]
    depths = np.zeros(N_t)
    for n in range(N_t):
        above = np.where(C_rt[n, :] > threshold)[0]
        depths[n] = R - r[above[0]] if len(above) > 0 else 0.0
    return depths


def compute_penetration_rmse(model, sims: list[dict], norm_stats: dict, config: dict) -> float:
    """RMSE between PINN-predicted and FDM-true concentration-based
    penetration depth, pooled across every time step of every test sim."""
    terms = []
    for sim in sims:
        r, t = sim["r"], sim["t"]
        C_pred = predict_concentration_field(model, r, t, sim, norm_stats)
        pen_pred = penetration_depth_from_concentration(C_pred, r, sim["C0_uM"])
        pen_true = penetration_depth_from_concentration(sim["C"], r, sim["C0_uM"])
        terms.append((pen_pred - pen_true) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(terms))))


def load_test_sim_params(config: dict) -> list[dict]:
    """Load the exact 5-tuples used for the test split, as saved by
    src.data_pipeline.build_dataset alongside the processed .npz files."""
    processed_dir = resolve_path(config, "processed")
    with open(processed_dir / "sim_params.json") as f:
        sim_params = json.load(f)
    return sim_params["test"]


def resolve_test_simulations(
    config: dict, sim_params: list[dict] | None = None, n_jobs: int = 1
) -> list[dict]:
    """Deterministically re-solve the full FDM reference field for each
    test-set simulation. Reuses src.data_pipeline's parallel-solve worker so
    the physics is identical to how the training dataset itself was built."""
    sim_params = sim_params if sim_params is not None else load_test_sim_params(config)
    args = [(p["sim_id"], np.array([p[k] for k in PARAM_ORDER]), config) for p in sim_params]

    if n_jobs == 1:
        results = [_solve_one(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_solve_one, args))
    return sorted(results, key=lambda sim: sim["sim_id"])


def _np_size_bin(d_NP_nm: float, config: dict) -> str:
    lo, hi = config["dataset"]["parameter_ranges"]["d_NP_nm"]
    edges = np.linspace(lo, hi, len(NP_SIZE_BIN_LABELS) + 1)
    idx = int(np.clip(np.digitize([d_NP_nm], edges[1:-1])[0], 0, len(NP_SIZE_BIN_LABELS) - 1))
    return NP_SIZE_BIN_LABELS[idx]


def _time_bins(t: np.ndarray) -> np.ndarray:
    return np.where(t < 24.0, TIME_BIN_LABELS[0], np.where(t < 48.0, TIME_BIN_LABELS[1], TIME_BIN_LABELS[2]))


def decompose_by(pred: np.ndarray, true: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    """Decompose compute_metrics by an arbitrary per-point label array
    (zone / NP-size bin / time bin)."""
    return {
        str(label): compute_metrics(pred[labels == label], true[labels == label], config)
        for label in np.unique(labels)
    }


def evaluate_test_set(model, sims: list[dict], norm_stats: dict, config: dict) -> dict:
    """Predict C(r,t) for every test simulation and compute global +
    zone/NP-size/time-decomposed metrics plus the penetration RMSE."""
    all_pred, all_true, all_zone, all_np_size, all_time = [], [], [], [], []

    for sim in sims:
        r, t, C_true = sim["r"], sim["t"], sim["C"]
        C_pred = predict_concentration_field(model, r, t, sim, norm_stats)

        oxygen = oxygen_gradient(r, config, R=sim["R_um"])
        zones = assign_zones(r, oxygen, config)
        np_size_label = _np_size_bin(sim["d_NP_nm"], config)
        time_per_t = _time_bins(t)

        all_pred.append(C_pred.ravel())
        all_true.append(C_true.ravel())
        all_zone.append(np.tile(zones, len(t)))
        all_np_size.append(np.full(C_pred.size, np_size_label))
        all_time.append(np.repeat(time_per_t, len(r)))

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    all_zone = np.concatenate(all_zone)
    all_np_size = np.concatenate(all_np_size)
    all_time = np.concatenate(all_time)

    global_metrics = compute_metrics(all_pred, all_true, config)
    global_metrics["penetration_rmse_um"] = compute_penetration_rmse(model, sims, norm_stats, config)

    decomposed = {
        "by_zone": decompose_by(all_pred, all_true, all_zone, config),
        "by_np_size": decompose_by(all_pred, all_true, all_np_size, config),
        "by_time": decompose_by(all_pred, all_true, all_time, config),
    }

    return {"global": global_metrics, "decomposed": decomposed}


def evaluate_h2_hypothesis(
    model,
    norm_stats: dict,
    config: dict,
    R_um: float = 400.0,
    C0_uM: float = 10.0,
    k_d_per_hr: float = 0.01,
) -> dict:
    """H2: at t=72hr and R=400um, the penetration-depth difference between a
    10nm and a 200nm nanoparticle exceeds 100um (smaller NPs diffuse faster
    and so penetrate deeper)."""
    hyp_cfg = config["evaluation"]["hypotheses"]
    fdm_cfg = config["fdm"]
    r = radial_grid(R_um, fdm_cfg["N_r"], fdm_cfg["r_min_um"])
    t = np.array([72.0])

    def depth_for(d_NP_nm: float) -> float:
        sim_params = {"R_um": R_um, "d_NP_nm": d_NP_nm, "C0_uM": C0_uM, "k_d_per_hr": k_d_per_hr, "t_max_hr": 72.0}
        C_pred = predict_concentration_field(model, r, t, sim_params, norm_stats)
        return float(penetration_depth_from_concentration(C_pred, r, C0_uM)[0])

    depth_10 = depth_for(10.0)
    depth_200 = depth_for(200.0)
    difference = depth_10 - depth_200

    return {
        "pass": bool(difference > hyp_cfg["H2_penetration_diff_um"]),
        "depth_10nm_um": depth_10,
        "depth_200nm_um": depth_200,
        "difference_um": difference,
        "target_um": hyp_cfg["H2_penetration_diff_um"],
    }


def evaluate_h4_over_test_set(model, sims: list[dict], norm_stats: dict, config: dict) -> dict:
    """H4 (rim<20%, core>60% viability at t=72hr), evaluated per test
    simulation and aggregated as a pass rate."""
    per_sim = []
    for sim in sims:
        r, t = sim["r"], sim["t"]
        C_pred = predict_concentration_field(model, r, t, sim, norm_stats)
        viability = viability_map(C_pred, t, config)
        result = evaluate_h4_hypothesis(viability, r, t, config, R_um=sim["R_um"])
        result["sim_id"] = sim["sim_id"]
        per_sim.append(result)

    pass_rate = float(np.mean([res["overall_pass"] for res in per_sim])) if per_sim else float("nan")
    return {"pass_rate": pass_rate, "n_sims": len(per_sim), "pass": bool(pass_rate >= 0.5), "per_sim": per_sim}


def evaluate_hypotheses(
    metrics_report: dict, h2_result: dict, h4_result: dict, config: dict
) -> dict:
    """Pass/fail summary for H1, H2, H4 against configs' evaluation.hypotheses
    targets. (H3/H5/H6 are computed by src/optimize.py and src/ablation.py;
    scripts/run_evaluation.py or a higher-level summary composes all six.)"""
    thresholds = config["evaluation"]["thresholds"]
    hyp_cfg = config["evaluation"]["hypotheses"]
    global_metrics = metrics_report["global"]

    h1_pass = global_metrics["rmse"] < hyp_cfg["H1_rmse_target_uM"] and global_metrics["r2"] > thresholds["r2"]

    return {
        "H1": {
            "pass": bool(h1_pass),
            "rmse_uM": global_metrics["rmse"],
            "r2": global_metrics["r2"],
            "target_rmse_uM": hyp_cfg["H1_rmse_target_uM"],
            "target_r2": thresholds["r2"],
        },
        "H2": h2_result,
        "H4": h4_result,
    }


def full_evaluation_report(
    model,
    config: dict,
    norm_stats: dict,
    sims: list[dict] | None = None,
    collocation_X: torch.Tensor | None = None,
    n_jobs: int = 1,
) -> dict:
    """Top-level entry point used by scripts/run_evaluation.py."""
    sims = sims if sims is not None else resolve_test_simulations(config, n_jobs=n_jobs)

    metrics_report = evaluate_test_set(model, sims, norm_stats, config)

    if collocation_X is None:
        processed_dir = resolve_path(config, "processed")
        test_npz = np.load(processed_dir / "test.npz")
        collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)
    residual_stats = compute_pde_residual_stats(model, collocation_X, config, norm_stats)
    metrics_report["global"]["mean_pde_residual"] = residual_stats["mean_abs_residual"]

    h2 = evaluate_h2_hypothesis(model, norm_stats, config)
    h4 = evaluate_h4_over_test_set(model, sims, norm_stats, config)
    hypotheses = evaluate_hypotheses(metrics_report, h2, h4, config)

    thresholds = config["evaluation"]["thresholds"]
    g = metrics_report["global"]
    threshold_pass_fail = {
        "rmse": g["rmse"] < thresholds["rmse_uM"],
        "mae": g["mae"] < thresholds["mae_uM"],
        "r2": g["r2"] > thresholds["r2"],
        "l2_relative_error": g["l2_relative"] < thresholds["l2_relative_error"],
        "mean_pde_residual": g["mean_pde_residual"] < thresholds["mean_pde_residual"],
        "penetration_rmse_um": g["penetration_rmse_um"] < thresholds["penetration_rmse_um"],
    }

    return {
        "metrics": metrics_report,
        "residual_histogram": residual_stats["residuals"],
        "threshold_pass_fail": threshold_pass_fail,
        "hypotheses": hypotheses,
        "n_test_sims": len(sims),
    }
