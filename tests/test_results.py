"""Phase 14 tests: manuscript Results & Discussion asset generation (src/results.py).

Uses a tiny FDM grid and an untrained BIOPINN model (same FAST_CONFIG pattern
as tests/test_optimize.py) so every figure/table function can be exercised
end-to-end without a real trained checkpoint or the full 300-sim test set --
this checks the glue code (shapes, column names, save mechanics, pass/fail
logic), not scientific correctness of the underlying physics (already
covered by test_fdm_solver.py, test_biology.py, test_optimize.py, etc.).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest
import torch

from src.ablation import train_baseline
from src.config import load_config
from src.data_pipeline import _solve_one, build_dataset
from src.model import BIOPINN
from src import results as R

CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 20
FAST_CONFIG["fdm"]["N_t_initial"] = 12
FAST_CONFIG["dataset"]["split"] = {"train": 4, "val": 2, "test": 3}
FAST_CONFIG["dataset"]["points_per_sim"] = {"data": 40, "collocation": 60, "bc_surface": 10, "bc_center": 10, "ic": 10}
FAST_CONFIG["training"]["adam"]["iters"] = 15
FAST_CONFIG["training"]["lbfgs"]["iters"] = 5
FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"] = [50, 100, 150]  # must include baseline.d_NP_nm (100) -- Table 4.1/Fig 4.3 reuse the swept solve for the baseline case
FAST_CONFIG["paper"]["sweeps"]["time_points_hr"] = [0, 24, 72]
FAST_CONFIG["paper"]["sweeps"]["radii_um"] = [300, 400]
FAST_CONFIG["paper"]["hypotheses"]["H6_speedup_n_combinations"] = 3
FAST_CONFIG["optimization"]["n_d_NP_points"] = 3
FAST_CONFIG["optimization"]["n_C0_points"] = 2
FAST_CONFIG["optimization"]["radii_um"] = [300.0, 400.0]
FAST_CONFIG["optimization"]["speedup_study"]["n_combinations"] = 3

NORM_STATS = {
    "R_um": {"min": 100.0, "max": 500.0},
    "d_NP_nm": {"min": 10.0, "max": 200.0},
    "C0_uM": {"min": 0.1, "max": 20.0},
    "k_d_per_hr": {"min": 0.005, "max": 0.05},
    "t_max_hr": {"min": 24.0, "max": 72.0},
}


def _make_sim(sim_id, R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr):
    return _solve_one((sim_id, np.array([R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr]), FAST_CONFIG))


@pytest.fixture(scope="module")
def model():
    return BIOPINN(FAST_CONFIG)


@pytest.fixture(scope="module")
def sims():
    return [
        _make_sim(0, 400.0, 100.0, 10.0, 0.01, 72.0),
        _make_sim(1, 300.0, 50.0, 5.0, 0.02, 48.0),
    ]


@pytest.fixture(scope="module")
def collocation_X():
    rng = np.random.default_rng(0)
    return torch.tensor(rng.uniform(0.0, 1.0, size=(200, 7)), dtype=torch.float32)


@pytest.fixture(scope="module")
def history():
    from src.train import HISTORY_KEYS

    rng = np.random.default_rng(1)
    return {key: list(np.abs(rng.normal(1e-2, 1e-3, size=30))) for key in HISTORY_KEYS} | {
        "val_data": list(np.abs(rng.normal(1e-2, 1e-3, size=30))),
        "val_phys": list(np.abs(rng.normal(1e-3, 1e-4, size=30))),
    }


def test_baseline_params_matches_config():
    params = R.baseline_params(FAST_CONFIG)
    assert params == FAST_CONFIG["paper"]["baseline"]


def test_load_training_history_prefers_dedicated_file(tmp_path):
    import src.config as cfg_module

    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])
    config["paths"]["training_history"] = "artifacts/training_history.json"

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        (tmp_path / "artifacts").mkdir(parents=True)
        history_data = {"total": [1.0, 0.5]}
        with open(tmp_path / "artifacts" / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f)

        loaded = R.load_training_history(config)
        assert loaded == history_data
    finally:
        cfg_module.REPO_ROOT = original_root


def test_load_training_history_falls_back_to_training_run_json(tmp_path):
    import src.config as cfg_module

    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])
    config["paths"]["training_history"] = "artifacts/training_history.json"
    config["paths"]["artifacts"] = "artifacts"

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        (tmp_path / "artifacts").mkdir(parents=True)
        with open(tmp_path / "artifacts" / "training_run.json", "w", encoding="utf-8") as f:
            json.dump({"history": {"total": [2.0, 1.0]}, "config": {}}, f)

        loaded = R.load_training_history(config)
        assert loaded == {"total": [2.0, 1.0]}
    finally:
        cfg_module.REPO_ROOT = original_root


def test_load_training_history_raises_clear_error_when_missing(tmp_path):
    import src.config as cfg_module

    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])
    config["paths"]["training_history"] = "artifacts/training_history.json"
    config["paths"]["artifacts"] = "artifacts"

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        with pytest.raises(FileNotFoundError):
            R.load_training_history(config)
    finally:
        cfg_module.REPO_ROOT = original_root


def test_solve_diameter_sweep_covers_configured_diameters():
    sweep = R.solve_diameter_sweep(FAST_CONFIG)
    assert set(sweep.keys()) == set(FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"])
    for fdm_result in sweep.values():
        assert fdm_result["C"].shape[1] == FAST_CONFIG["fdm"]["N_r"]


def test_solve_hetero_and_homogeneous_shapes_match():
    solved = R.solve_hetero_and_homogeneous(FAST_CONFIG)
    assert solved["hetero"]["C"].shape == solved["homog"]["C"].shape
    assert solved["D_eff_homogeneous_um2_per_hr"] > 0


def test_save_figure_writes_png_and_pdf(tmp_path):
    fig, ax = R.plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = R._save_figure(fig, tmp_path, "test_fig")
    assert Path(paths["png"]).exists() and Path(paths["png"]).stat().st_size > 0
    assert Path(paths["pdf"]).exists() and Path(paths["pdf"]).stat().st_size > 0


# --------------------------------------------------------------------------- #
# Figures: each returns (Figure, meta dict)
# --------------------------------------------------------------------------- #


def test_fig_4_1_concentration_heatmap(model):
    fig, meta = R.fig_4_1_concentration_heatmap(model, FAST_CONFIG, NORM_STATS, n_r=15, n_t=10)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["max_concentration_uM"] >= 0
    R.plt.close(fig)


def test_fig_4_2_radial_profiles(model):
    fig, meta = R.fig_4_2_radial_profiles(model, FAST_CONFIG, NORM_STATS, n_r=15)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["time_points_hr"] == FAST_CONFIG["paper"]["sweeps"]["time_points_hr"]
    R.plt.close(fig)


def test_fig_4_3_pinn_vs_fdm_t24(model):
    sweep = R.solve_diameter_sweep(FAST_CONFIG, params={"d_NP_nm": FAST_CONFIG["paper"]["baseline"]["d_NP_nm"]})
    fig, meta = R.fig_4_3_pinn_vs_fdm_t24(model, FAST_CONFIG, NORM_STATS, diameter_sweep=sweep)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["mean_abs_residual_uM"] >= 0
    R.plt.close(fig)


def test_fig_4_4_scatter_pred_vs_ref(model, sims):
    fig, meta = R.fig_4_4_scatter_pred_vs_ref(model, FAST_CONFIG, NORM_STATS, sims, max_points=50)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["n_test_sims"] == len(sims)
    assert -np.inf < meta["r2"] <= 1.0
    R.plt.close(fig)


def test_fig_4_5_training_loss(history):
    fig, meta = R.fig_4_5_training_loss(history, FAST_CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["n_logged_steps"] == 30
    R.plt.close(fig)


def test_fig_4_6_penetration_vs_time(model):
    fig, meta = R.fig_4_6_penetration_vs_time(model, FAST_CONFIG, NORM_STATS, n_r=15, n_t=8)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert set(meta["final_penetration_depth_um"].keys()) == set(FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"])
    R.plt.close(fig)


def test_fig_4_7_viability_t72(model):
    fig, meta = R.fig_4_7_viability_t72(model, FAST_CONFIG, NORM_STATS, n_r=20, n_t=8)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert set(meta["zone_mean_viability_pct"].keys()) == set(R.ZONE_COLORS.keys())
    R.plt.close(fig)


def test_fig_4_8_cytotoxicity_evolution(model):
    fig, meta = R.fig_4_8_cytotoxicity_evolution(model, FAST_CONFIG, NORM_STATS, n_r=15)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(meta["time_points_hr"]) <= 5
    R.plt.close(fig)


def test_fig_4_9_hetero_vs_homog():
    fig, meta = R.fig_4_9_hetero_vs_homog(FAST_CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["D_eff_homogeneous_um2_per_hr"] > 0
    R.plt.close(fig)


def test_fig_4_10_effectiveness_surface(model):
    fig, meta = R.fig_4_10_effectiveness_surface(model, FAST_CONFIG, NORM_STATS)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert meta["R_um"] == FAST_CONFIG["paper"]["baseline"]["R_um"]
    R.plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables: each returns (DataFrame, meta dict)
# --------------------------------------------------------------------------- #


def test_table_4_1_fdm_summary():
    df, meta = R.table_4_1_fdm_summary(FAST_CONFIG)
    assert list(df.columns) == ["Time (hr)", "Max Penetration Depth (μm)", "Center Conc. (μM)", "Mean Tumor Conc. (μM)", "CFL Number"]
    assert len(df) == len(FAST_CONFIG["paper"]["sweeps"]["time_points_hr"])
    assert meta["cfl_number"] > 0


def test_table_4_2_penetration_vs_diameter():
    df, _ = R.table_4_2_penetration_vs_diameter(FAST_CONFIG)
    assert len(df) == len(FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"])
    assert (df["D_free (μm²/hr)"] > 0).all()
    # Stokes-Einstein: smaller nanoparticles diffuse faster.
    sorted_by_size = df.sort_values("d_NP (nm)")
    assert sorted_by_size["D_free (μm²/hr)"].is_monotonic_decreasing


def test_table_4_3_pinn_metrics(model, sims, collocation_X):
    df, meta = R.table_4_3_pinn_metrics(model, FAST_CONFIG, NORM_STATS, sims, collocation_X)
    assert list(df["Metric"]) == ["RMSE (μM)", "MAE (μM)", "R²", "L2 Relative Error", "Mean PDE Residual", "Penetration RMSE (μm)"]
    assert set(df["Hypothesis Outcome"]) <= {"Pass", "Fail"}
    assert meta["n_test_sims"] == len(sims)
    assert meta["n_collocation_points"] == collocation_X.shape[0]


def test_table_4_4_penetration_analysis(model):
    df, _ = R.table_4_4_penetration_analysis(model, FAST_CONFIG, NORM_STATS)
    assert len(df) == len(FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"])
    assert (df["PINN RMSE (μm)"] >= 0).all()


def test_table_4_5_viability_summary(model):
    df, _ = R.table_4_5_viability_summary(model, FAST_CONFIG, NORM_STATS, n_r=20, n_t=8)
    assert len(df) == len(FAST_CONFIG["paper"]["sweeps"]["d_NP_nm"])
    for col in ("Outer Rim Viability (%)", "Quiescent Zone Viability (%)", "Core Viability (%)", "Overall Kill Fraction (%)", "Resistance Risk Fraction (%)"):
        assert col in df.columns


def test_table_4_6_hetero_vs_homog():
    df, meta = R.table_4_6_hetero_vs_homog(FAST_CONFIG)
    assert list(df["Parameter"]) == [
        "Max penetration depth (μm)",
        "Sub-therapeutic zone radius (μm)",
        "Resistance risk fraction (%)",
        "Overall kill fraction (%)",
    ]
    assert isinstance(meta["H3_pass"], bool)


def test_table_4_7_optimization_results(model):
    df, meta = R.table_4_7_optimization_results(model, FAST_CONFIG, NORM_STATS)
    assert set(df["Tumor Radius (μm)"]) == set(FAST_CONFIG["optimization"]["radii_um"])
    assert (df["Computation Time (sec)"] >= 0).all()


def test_table_4_8_ablation(model, sims, collocation_X, tmp_path):
    dataset = build_dataset(FAST_CONFIG, seed=7, save=False)
    baseline_result = train_baseline(FAST_CONFIG, dataset, save=False)
    baseline_model = baseline_result["model"]

    df, meta = R.table_4_8_ablation(model, baseline_model, FAST_CONFIG, NORM_STATS, sims, collocation_X)
    assert list(df["Metric"]) == ["RMSE (μM)", "R²", "Mean PDE Residual", "Max PDE Residual", "Physical consistency (%)"]
    assert isinstance(meta["H5_pass"], bool)
    assert 0.0 <= meta["wilcoxon_p_value"] <= 1.0


def test_table_4_9_hypothesis_summary():
    hypotheses = {
        key: {"name": f"Hypothesis {key}", "expected": "expected outcome text", "evidence": f"Table 4.{i+1}", "pass": i % 2 == 0}
        for i, key in enumerate(("H1", "H2", "H3", "H4", "H5", "H6"))
    }
    df = R.table_4_9_hypothesis_summary(hypotheses)
    assert list(df.columns) == ["Hypothesis", "Expected Outcome", "Evidence Basis", "Conclusion"]
    assert len(df) == 6
    assert df.iloc[0]["Conclusion"] == "Supported"
    assert df.iloc[1]["Conclusion"] == "Not supported"
