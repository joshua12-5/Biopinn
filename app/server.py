"""FastAPI backend for the BIOPINN results dashboard.

Loads the trained checkpoint once at startup and serves it live: a fast
`/api/predict` endpoint queries the PINN surrogate for concentration,
viability, cytotoxicity, penetration depth, and treatment effectiveness at
whatever (R, d_NP, C0, k_d, t_max) the user sets, plus slower endpoints
(optimization grid search, six-metric evaluation, ablation comparison) that
are computed once in a background thread and cached, since they re-solve
FDM references / re-run grid searches. Never retrains -- consumes
artifacts/biopinn_model.pt (+ optionally a w_phys=0 ablation baseline)
exactly as scripts/run_evaluation.py, run_ablation.py, and
run_optimization.py do.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ablation import compare_residuals, wilcoxon_test
from src.biology import compute_biological_response
from src.config import load_config, resolve_path
from src.evaluate import full_evaluation_report, resolve_test_simulations
from src.microenvironment import assign_zones, oxygen_gradient, radial_grid
from src.model import load_checkpoint
from src.optimize import _resistance_map, kill_fraction, optimize_all_radii

APP_DIR = Path(__file__).resolve().parent
UI_N_R = 120
UI_N_T = 80


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_startup, daemon=True).start()
    yield


app = FastAPI(title="BIOPINN results dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


class ParamsIn(BaseModel):
    R_um: float = Field(..., gt=0)
    d_NP_nm: float = Field(..., gt=0)
    C0_uM: float = Field(..., gt=0)
    k_d_per_hr: float = Field(..., ge=0)
    t_max_hr: float = Field(..., gt=0)


class State:
    """Everything loaded once at startup + lazily-computed, cached results."""

    def __init__(self) -> None:
        self.config: dict | None = None
        self.norm_stats: dict | None = None
        self.model: torch.nn.Module | None = None
        self.baseline_model: torch.nn.Module | None = None
        self.load_error: str | None = None

        self.optimization: dict = {"status": "pending"}
        self.evaluation: dict = {"status": "pending"}
        self.ablation: dict = {"status": "pending"}


state = State()


def _startup() -> None:
    try:
        config = load_config(os.environ.get("BIOPINN_EXPERIMENT") or None)
        checkpoint_path = resolve_path(config, "model_checkpoint")
        if not checkpoint_path.exists():
            state.load_error = (
                f"No checkpoint found at {checkpoint_path}. "
                "Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts into artifacts/."
            )
            return

        model = load_checkpoint(str(checkpoint_path), config)
        stats_path = resolve_path(config, "normalization_stats")
        with open(stats_path, encoding="utf-8") as f:
            norm_stats = json.load(f)

        state.config = config
        state.norm_stats = norm_stats
        state.model = model

        baseline_path = checkpoint_path.with_name(checkpoint_path.stem + "_baseline" + checkpoint_path.suffix)
        if baseline_path.exists():
            state.baseline_model = load_checkpoint(str(baseline_path), config)
    except Exception as exc:  # surfaced to the UI rather than crashing the server
        state.load_error = f"Failed to load model artifacts: {exc}"
        return

    threading.Thread(target=_compute_optimization, daemon=True).start()
    threading.Thread(target=_compute_evaluation, daemon=True).start()


def _compute_optimization() -> None:
    try:
        results = optimize_all_radii(state.model, state.config, state.norm_stats)
        state.optimization = _sanitize(
            {
                "status": "ready",
                "radii": [
                    {
                        "R_um": R_um,
                        "d_NP_star_nm": r["d_NP_star_nm"],
                        "C0_star_uM": r["C0_star_uM"],
                        "max_eta": r["max_eta"],
                        "resistant_volume_fraction": r["resistance_map"]["resistant_volume_fraction"],
                        "computation_time_s": r["computation_time_s"],
                        "eta_grid": r["eta_grid"].tolist(),
                        "d_NP_grid_nm": r["d_NP_grid_nm"].tolist(),
                        "C0_grid_uM": r["C0_grid_uM"].tolist(),
                    }
                    for R_um, r in sorted(results.items())
                ],
            }
        )
    except Exception as exc:
        state.optimization = {"status": "error", "message": str(exc)}


def _sanitize(obj):
    """Recursively replace NaN/Infinity with None (Starlette's JSONResponse
    uses allow_nan=False, so any NaN r2/l2_relative from a tiny or
    zero-variance decomposition bucket would otherwise 500 the endpoint)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _histogram(residuals: np.ndarray, bins: int = 60) -> dict:
    counts, edges = np.histogram(residuals, bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def _compute_evaluation() -> None:
    try:
        config, model, norm_stats = state.config, state.model, state.norm_stats
        sims = resolve_test_simulations(config, n_jobs=1)
        report = full_evaluation_report(model, config, norm_stats, sims=sims)

        sim = sims[0]
        from src.biology import predict_concentration_field

        C_pred = predict_concentration_field(model, sim["r"], sim["t"], sim, norm_stats)

        state.evaluation = _sanitize(
            {
                "status": "ready",
                "global": report["metrics"]["global"],
                "threshold_pass_fail": report["threshold_pass_fail"],
                "hypotheses": {
                    "H1": report["hypotheses"]["H1"],
                    "H2": report["hypotheses"]["H2"],
                    "H4": report["hypotheses"]["H4"],
                },
                "n_test_sims": report["n_test_sims"],
                "residual_histogram": _histogram(report["residual_histogram"]),
                "overlay": {
                    "sim_id": sim["sim_id"],
                    "R_um": sim["R_um"],
                    "d_NP_nm": sim["d_NP_nm"],
                    "C0_uM": sim["C0_uM"],
                    "r": sim["r"].tolist(),
                    "t": sim["t"].tolist(),
                    "C_true": sim["C"].tolist(),
                    "C_pred": C_pred.tolist(),
                },
            }
        )

        threading.Thread(target=_compute_ablation, args=(sims,), daemon=True).start()
    except Exception as exc:
        state.evaluation = {"status": "error", "message": str(exc)}
        state.ablation = {"status": "error", "message": "evaluation failed, ablation skipped"}


def _compute_ablation(sims: list[dict]) -> None:
    if state.baseline_model is None:
        state.ablation = {
            "status": "unavailable",
            "message": "No ablation baseline checkpoint found. Run scripts/run_ablation.py to enable this panel.",
        }
        return
    try:
        config, norm_stats = state.config, state.norm_stats
        processed_dir = resolve_path(config, "processed")
        with np.load(processed_dir / "test.npz") as test_npz:
            collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)

        rc = compare_residuals(state.model, state.baseline_model, collocation_X, config, norm_stats)
        wx = wilcoxon_test(rc["residuals_biopinn"], rc["residuals_baseline"])

        hyp_cfg = config["evaluation"]["hypotheses"]
        h5_pass = rc["improvement_factor_mean"] >= hyp_cfg["H5_residual_improvement_factor"] and wx["significant"]

        state.ablation = _sanitize(
            {
                "status": "ready",
                "biopinn": rc["biopinn"],
                "baseline": rc["baseline"],
                "improvement_factor_mean": rc["improvement_factor_mean"],
                "improvement_factor_max": rc["improvement_factor_max"],
                "wilcoxon": wx,
                "H5": {
                    "pass": bool(h5_pass),
                    "target_improvement_factor": hyp_cfg["H5_residual_improvement_factor"],
                },
                "residuals_biopinn_hist": _histogram(rc["residuals_biopinn"]),
                "residuals_baseline_hist": _histogram(rc["residuals_baseline"]),
            }
        )
    except Exception as exc:
        state.ablation = {"status": "error", "message": str(exc)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(APP_DIR / "templates" / "index.html"))


@app.get("/api/meta")
def get_meta() -> JSONResponse:
    if state.load_error:
        return JSONResponse({"status": "error", "message": state.load_error}, status_code=503)
    if state.config is None:
        return JSONResponse({"status": "loading"}, status_code=503)

    ranges = state.config["dataset"]["parameter_ranges"]

    def midpoint(key: str) -> float:
        lo, hi = ranges[key]
        return round((lo + hi) / 2, 4)

    return JSONResponse(
        {
            "status": "ready",
            "parameters": {
                "R_um": {"min": ranges["R_um"][0], "max": ranges["R_um"][1], "default": midpoint("R_um"), "unit": "μm", "label": "Tumor radius"},
                "d_NP_nm": {"min": ranges["d_NP_nm"][0], "max": ranges["d_NP_nm"][1], "default": midpoint("d_NP_nm"), "unit": "nm", "label": "Nanoparticle diameter"},
                "C0_uM": {"min": ranges["C0_uM"][0], "max": ranges["C0_uM"][1], "default": midpoint("C0_uM"), "unit": "μM", "label": "Surface concentration"},
                "k_d_per_hr": {"min": ranges["k_d_per_hr"][0], "max": ranges["k_d_per_hr"][1], "default": midpoint("k_d_per_hr"), "unit": "hr⁻¹", "label": "Decay rate"},
                "t_max_hr": {"min": ranges["t_max_hr"][0], "max": ranges["t_max_hr"][1], "default": midpoint("t_max_hr"), "unit": "hr", "label": "Simulation duration"},
            },
            "thresholds": state.config["evaluation"]["thresholds"],
            "zones": state.config["microenvironment"]["zones"],
            "radii_um": state.config["optimization"]["radii_um"],
            "has_ablation_baseline": state.baseline_model is not None,
        }
    )


@app.post("/api/predict")
def predict(params: ParamsIn) -> JSONResponse:
    if state.model is None:
        raise HTTPException(503, state.load_error or "Model not loaded yet.")

    config, norm_stats = state.config, state.norm_stats
    sim_params = params.model_dump()

    r = radial_grid(sim_params["R_um"], UI_N_R, config["fdm"]["r_min_um"])
    t = np.linspace(0.0, sim_params["t_max_hr"], UI_N_T)

    response = compute_biological_response(state.model, r, t, sim_params, norm_stats, config)

    oxygen = oxygen_gradient(r, config, R=sim_params["R_um"])
    zones = assign_zones(r, oxygen, config)

    eta = kill_fraction(
        state.model, sim_params["R_um"], sim_params["d_NP_nm"], sim_params["C0_uM"],
        config, norm_stats, sim_params["k_d_per_hr"], sim_params["t_max_hr"],
    )
    resistance = _resistance_map(
        state.model, sim_params["R_um"], sim_params["d_NP_nm"], sim_params["C0_uM"],
        config, norm_stats, sim_params["k_d_per_hr"], sim_params["t_max_hr"],
    )

    return JSONResponse(
        _sanitize(
            {
                "r": r.tolist(),
                "t": t.tolist(),
                "C": response["C"].tolist(),
                "viability": response["viability"].tolist(),
                "cytotoxicity": response["cytotoxicity"].tolist(),
                "penetration_depth": response["penetration_depth"].tolist(),
                "zones": zones.tolist(),
                "kill_fraction": eta,
                "resistance": {
                    "C_final_uM": resistance["C_final_uM"].tolist(),
                    "resistant_mask": resistance["resistant_mask"].tolist(),
                    "resistant_volume_fraction": resistance["resistant_volume_fraction"],
                    "threshold_uM": resistance["threshold_uM"],
                },
            }
        )
    )


@app.get("/api/optimization")
def get_optimization() -> JSONResponse:
    status = 200 if state.optimization.get("status") in ("ready", "unavailable") else 202
    return JSONResponse(state.optimization, status_code=status)


@app.get("/api/evaluation")
def get_evaluation() -> JSONResponse:
    status = 200 if state.evaluation.get("status") in ("ready", "unavailable") else 202
    return JSONResponse(state.evaluation, status_code=status)


@app.get("/api/ablation")
def get_ablation() -> JSONResponse:
    status = 200 if state.ablation.get("status") in ("ready", "unavailable") else 202
    return JSONResponse(state.ablation, status_code=status)


@app.get("/api/export/metrics.json")
def export_metrics_json() -> JSONResponse:
    if state.evaluation.get("status") != "ready":
        raise HTTPException(409, "Evaluation not ready yet.")
    payload = {k: v for k, v in state.evaluation.items() if k not in ("residual_histogram", "overlay")}
    return JSONResponse(payload, headers={"Content-Disposition": "attachment; filename=biopinn_metrics.json"})


@app.get("/api/export/metrics.csv")
def export_metrics_csv() -> PlainTextResponse:
    if state.evaluation.get("status") != "ready":
        raise HTTPException(409, "Evaluation not ready yet.")
    g = state.evaluation["global"]
    pf = state.evaluation["threshold_pass_fail"]
    lines = ["metric,value,pass"]
    for key in ("rmse", "mae", "r2", "l2_relative", "mean_pde_residual", "penetration_rmse_um"):
        pf_key = key if key != "l2_relative" else "l2_relative_error"
        lines.append(f"{key},{g[key]},{pf.get(pf_key, '')}")
    return PlainTextResponse(
        "\n".join(lines), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=biopinn_metrics.csv"},
    )
