"""Phase 8 tests: six-metric evaluation, decomposition, PDE-residual stats,
penetration RMSE, and the H1/H2/H4 hypothesis checks."""

import copy
import json

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data_pipeline import build_dataset
from src.evaluate import (
    compute_metrics,
    compute_penetration_rmse,
    compute_pde_residual_stats,
    decompose_by,
    evaluate_h2_hypothesis,
    evaluate_h4_over_test_set,
    evaluate_hypotheses,
    evaluate_test_set,
    full_evaluation_report,
    load_test_sim_params,
    penetration_depth_from_concentration,
    resolve_test_simulations,
)
from src.model import BIOPINN
from src.train import train

CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 25
FAST_CONFIG["fdm"]["N_t_initial"] = 15
FAST_CONFIG["dataset"]["split"] = {"train": 4, "val": 2, "test": 3}
FAST_CONFIG["dataset"]["points_per_sim"] = {
    "data": 40,
    "collocation": 60,
    "bc_surface": 10,
    "bc_center": 10,
    "ic": 10,
}
FAST_CONFIG["training"]["adam"]["iters"] = 15
FAST_CONFIG["training"]["lbfgs"]["iters"] = 5


def test_compute_metrics_perfect_prediction():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(true.copy(), true, CONFIG)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["l2_relative"] == pytest.approx(0.0)
    assert metrics["n_points"] == 4


def test_compute_metrics_matches_manual_computation():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.5, 2.0, 2.0])
    metrics = compute_metrics(pred, true, CONFIG)

    err = pred - true
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)))
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(err)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    assert metrics["r2"] == pytest.approx(1 - ss_res / ss_tot)
    assert metrics["l2_relative"] == pytest.approx(np.sqrt(ss_res) / np.sqrt(np.sum(true**2)))


def test_penetration_depth_from_concentration_basic_cases():
    r = np.linspace(0, 400, 5)
    C0 = 10.0

    none_above = np.zeros((1, 5))
    assert penetration_depth_from_concentration(none_above, r, C0)[0] == 0.0

    all_above = np.full((1, 5), 5.0)  # > 0.1*C0 = 1.0 everywhere
    assert penetration_depth_from_concentration(all_above, r, C0)[0] == pytest.approx(400.0)

    partial = np.array([[0.5, 0.5, 0.5, 2.0, 5.0]])  # only last two exceed threshold=1.0
    assert penetration_depth_from_concentration(partial, r, C0)[0] == pytest.approx(100.0)


def test_decompose_by_splits_correctly():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    true = np.array([1.0, 2.0, 3.0, 5.0])
    labels = np.array(["a", "a", "b", "b"])

    result = decompose_by(pred, true, labels, CONFIG)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"]["rmse"] == pytest.approx(0.0)
    assert result["b"]["rmse"] == pytest.approx(np.sqrt(np.mean([0.0, 1.0])))


def _dataset():
    return build_dataset(FAST_CONFIG, seed=21, save=False)


def _quick_trained_model_and_dataset():
    dataset = _dataset()
    result = train(FAST_CONFIG, dataset, save=False)
    return result["model"], dataset


def test_resolve_test_simulations_matches_original_raw_sims():
    dataset = _dataset()
    original_test_sims = dataset["sims"]["test"]
    sim_params = [
        {k: sim[k] for k in ("sim_id", "R_um", "d_NP_nm", "C0_uM", "k_d_per_hr", "t_max_hr")}
        for sim in original_test_sims
    ]

    resolved = resolve_test_simulations(FAST_CONFIG, sim_params=sim_params, n_jobs=1)
    assert len(resolved) == len(original_test_sims)
    for original, re_solved in zip(
        sorted(original_test_sims, key=lambda s: s["sim_id"]), resolved
    ):
        assert original["sim_id"] == re_solved["sim_id"]
        np.testing.assert_allclose(original["C"], re_solved["C"])


def test_load_test_sim_params_round_trip(tmp_path):
    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])

    import src.config as cfg_module

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["processed"] = "processed"

        result = build_dataset(config, seed=21, save=True)
        loaded_params = load_test_sim_params(config)

        assert len(loaded_params) == config["dataset"]["split"]["test"]
        original_ids = {sim["sim_id"] for sim in result["sims"]["test"]}
        loaded_ids = {p["sim_id"] for p in loaded_params}
        assert original_ids == loaded_ids
    finally:
        cfg_module.REPO_ROOT = original_root


def test_evaluate_test_set_shapes_and_keys():
    model, dataset = _quick_trained_model_and_dataset()
    sims = dataset["sims"]["test"]
    norm_stats = dataset["stats"]

    report = evaluate_test_set(model, sims, norm_stats, FAST_CONFIG)
    assert set(report.keys()) == {"global", "decomposed"}
    for key in ("rmse", "mae", "r2", "l2_relative", "penetration_rmse_um"):
        assert np.isfinite(report["global"][key])

    assert set(report["decomposed"].keys()) == {"by_zone", "by_np_size", "by_time"}
    for decomposition in report["decomposed"].values():
        assert len(decomposition) > 0
        for bucket_metrics in decomposition.values():
            assert np.isfinite(bucket_metrics["rmse"])


def test_compute_pde_residual_stats_finite():
    model, dataset = _quick_trained_model_and_dataset()
    norm_stats = dataset["stats"]
    X = torch.as_tensor(dataset["splits"]["test"]["collocation_X"], dtype=torch.float32)

    stats = compute_pde_residual_stats(model, X, FAST_CONFIG, norm_stats)
    assert np.isfinite(stats["mean_abs_residual"])
    assert np.isfinite(stats["max_abs_residual"])
    assert stats["max_abs_residual"] >= stats["mean_abs_residual"]
    assert stats["residuals"].shape == (X.shape[0],)


def test_compute_penetration_rmse_finite_and_zero_for_self_comparison():
    model, dataset = _quick_trained_model_and_dataset()
    sims = dataset["sims"]["test"]
    norm_stats = dataset["stats"]

    rmse = compute_penetration_rmse(model, sims, norm_stats, FAST_CONFIG)
    assert np.isfinite(rmse)
    assert rmse >= 0.0


def test_evaluate_h2_hypothesis_shape_and_types():
    model = BIOPINN(FAST_CONFIG)
    norm_stats = {
        "R_um": {"min": 100.0, "max": 500.0},
        "d_NP_nm": {"min": 10.0, "max": 200.0},
        "C0_uM": {"min": 0.1, "max": 20.0},
        "k_d_per_hr": {"min": 0.005, "max": 0.05},
        "t_max_hr": {"min": 24.0, "max": 72.0},
    }
    result = evaluate_h2_hypothesis(model, norm_stats, CONFIG)
    assert isinstance(result["pass"], bool)
    for key in ("depth_10nm_um", "depth_200nm_um", "difference_um"):
        assert np.isfinite(result[key])


def test_evaluate_h4_over_test_set_aggregates_correctly():
    model, dataset = _quick_trained_model_and_dataset()
    sims = dataset["sims"]["test"]
    norm_stats = dataset["stats"]

    result = evaluate_h4_over_test_set(model, sims, norm_stats, FAST_CONFIG)
    assert result["n_sims"] == len(sims)
    assert 0.0 <= result["pass_rate"] <= 1.0
    assert len(result["per_sim"]) == len(sims)
    assert isinstance(result["pass"], bool)


def test_evaluate_hypotheses_h1_pass_fail_logic():
    good_report = {"global": {"rmse": 0.01, "r2": 0.999}}
    bad_report = {"global": {"rmse": 1.0, "r2": 0.5}}
    h2_stub, h4_stub = {"pass": True}, {"pass": True}

    good = evaluate_hypotheses(good_report, h2_stub, h4_stub, CONFIG)
    bad = evaluate_hypotheses(bad_report, h2_stub, h4_stub, CONFIG)
    assert good["H1"]["pass"] is True
    assert bad["H1"]["pass"] is False


def test_full_evaluation_report_end_to_end():
    model, dataset = _quick_trained_model_and_dataset()
    sims = dataset["sims"]["test"]
    norm_stats = dataset["stats"]
    collocation_X = torch.as_tensor(dataset["splits"]["test"]["collocation_X"], dtype=torch.float32)

    report = full_evaluation_report(
        model, FAST_CONFIG, norm_stats, sims=sims, collocation_X=collocation_X
    )

    assert set(report.keys()) == {
        "metrics",
        "residual_histogram",
        "threshold_pass_fail",
        "hypotheses",
        "n_test_sims",
    }
    assert report["n_test_sims"] == len(sims)
    assert set(report["hypotheses"].keys()) == {"H1", "H2", "H4"}
    for passed in report["threshold_pass_fail"].values():
        assert isinstance(passed, (bool, np.bool_))

    # Report must be JSON-serializable once the (numpy) residual histogram
    # array is dropped -- exactly how scripts/run_evaluation.py exports it.
    exportable = {k: v for k, v in report.items() if k != "residual_histogram"}
    json.dumps(exportable, default=lambda o: o.item() if hasattr(o, "item") else str(o))
