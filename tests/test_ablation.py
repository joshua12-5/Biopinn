"""Phase 9 tests: w_phys=0 baseline training, residual comparison, Wilcoxon
signed-rank test, and the H5 hypothesis check."""

import copy

import numpy as np
import pytest
import torch

from src.ablation import ablation_report, compare_residuals, train_baseline, wilcoxon_test
from src.config import load_config
from src.data_pipeline import build_dataset
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


def _dataset():
    return build_dataset(FAST_CONFIG, seed=31, save=False)


def test_train_baseline_uses_w_phys_zero():
    dataset = _dataset()
    result = train_baseline(FAST_CONFIG, dataset, save=False)

    history = result["history"]
    loss_cfg = FAST_CONFIG["loss"]
    # With w_phys=0, total must reconstruct exactly from the other four
    # weighted components (phys is excluded from the sum entirely).
    for i in range(len(history["total"])):
        expected = (
            loss_cfg["w_data"] * history["data"][i]
            + loss_cfg["w_bc"] * history["bc"][i]
            + loss_cfg["w_neu"] * history["neu"][i]
            + loss_cfg["w_ic"] * history["ic"][i]
        )
        assert history["total"][i] == pytest.approx(expected, rel=1e-4)


def test_train_baseline_does_not_touch_primary_config():
    dataset = _dataset()
    original = copy.deepcopy(FAST_CONFIG)
    train_baseline(FAST_CONFIG, dataset, save=False)
    assert FAST_CONFIG["loss"]["w_phys"] == original["loss"]["w_phys"]
    assert FAST_CONFIG["paths"]["model_checkpoint"] == original["paths"]["model_checkpoint"]


def test_train_baseline_checkpoint_path_differs_from_primary(tmp_path):
    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])

    import src.config as cfg_module

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["model_checkpoint"] = "artifacts/biopinn_model.pt"
        config["paths"]["normalization_stats"] = "artifacts/normalization_stats.json"

        dataset = build_dataset(config, seed=31, save=False)
        result = train_baseline(config, dataset, save=True)

        baseline_ckpt = result["artifacts"]["checkpoint_path"]
        assert baseline_ckpt != tmp_path / "artifacts/biopinn_model.pt"
        assert baseline_ckpt.exists()
        assert "_baseline" in baseline_ckpt.name
    finally:
        cfg_module.REPO_ROOT = original_root


def test_compare_residuals_shapes_and_finiteness():
    dataset = _dataset()
    biopinn_model = train(FAST_CONFIG, dataset, save=False)["model"]
    baseline_model = train_baseline(FAST_CONFIG, dataset, save=False)["model"]
    norm_stats = dataset["stats"]

    collocation_X = torch.as_tensor(dataset["splits"]["test"]["collocation_X"], dtype=torch.float32)
    comparison = compare_residuals(biopinn_model, baseline_model, collocation_X, FAST_CONFIG, norm_stats)

    for key in ("biopinn", "baseline"):
        block = comparison[key]
        assert np.isfinite(block["mean_abs_residual"])
        assert np.isfinite(block["max_abs_residual"])
        assert 0.0 <= block["physical_consistency_pct"] <= 100.0

    assert comparison["improvement_factor_mean"] > 0
    assert comparison["residuals_biopinn"].shape == (collocation_X.shape[0],)
    assert comparison["residuals_baseline"].shape == (collocation_X.shape[0],)


def test_wilcoxon_detects_clear_separation():
    rng = np.random.default_rng(0)
    residuals_biopinn = np.abs(rng.normal(loc=0.001, scale=0.0005, size=200))
    residuals_baseline = residuals_biopinn + np.abs(rng.normal(loc=0.5, scale=0.05, size=200))

    result = wilcoxon_test(residuals_biopinn, residuals_baseline)
    assert result["significant"] is True
    assert result["p_value"] < 0.05
    assert result["n_pairs"] == 200


def test_wilcoxon_no_separation_when_distributions_match():
    rng = np.random.default_rng(1)
    residuals_biopinn = np.abs(rng.normal(loc=0.1, scale=0.02, size=200))
    residuals_baseline = np.abs(rng.normal(loc=0.1, scale=0.02, size=200))

    result = wilcoxon_test(residuals_biopinn, residuals_baseline)
    assert isinstance(result["significant"], bool)
    assert 0.0 <= result["p_value"] <= 1.0


def test_ablation_report_end_to_end():
    dataset = _dataset()
    biopinn_model = train(FAST_CONFIG, dataset, save=False)["model"]
    baseline_model = train_baseline(FAST_CONFIG, dataset, save=False)["model"]
    norm_stats = dataset["stats"]
    sims = dataset["sims"]["test"]
    collocation_X = torch.as_tensor(dataset["splits"]["test"]["collocation_X"], dtype=torch.float32)

    report = ablation_report(
        biopinn_model, baseline_model, FAST_CONFIG, norm_stats, sims=sims, collocation_X=collocation_X
    )

    assert set(report.keys()) == {
        "biopinn_data_metrics",
        "baseline_data_metrics",
        "residual_comparison",
        "wilcoxon",
        "H5",
        "n_test_sims",
    }
    assert report["n_test_sims"] == len(sims)
    assert isinstance(report["H5"]["pass"], bool)
    assert np.isfinite(report["H5"]["improvement_factor_mean"])
    assert "residuals_biopinn" not in report["residual_comparison"]  # raw arrays stripped for export
