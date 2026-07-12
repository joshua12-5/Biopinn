"""Physics-informed vs. unconstrained ablation study (H5).

Trains an identical-architecture baseline network (5x64 tanh, same
train/val/test split and normalization stats -- see
src.data_pipeline.load_processed_dataset) but with the physics loss switched
off (w_phys=0: data + Dirichlet + Neumann + IC only), then compares it
against the primary BIOPINN checkpoint on the held-out test set: RMSE/R^2
(expected comparable -- physics loss doesn't improve in-distribution
interpolation), mean/max PDE residual and physical-consistency % (expected
1-2 orders of magnitude better for BIOPINN), and a Wilcoxon signed-rank test
on the paired per-collocation-point residual magnitudes.

Local module: the primary BIOPINN model is loaded from a pre-trained
checkpoint and never retrained here; only the baseline is trained locally,
reusing src/train.py unchanged.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.config import resolve_path
from src.evaluate import compute_pde_residual_stats, evaluate_test_set, resolve_test_simulations
from src.train import train


def train_baseline(config: dict, dataset: dict, device: str = "cpu", save: bool = True) -> dict:
    """Train the w_phys=0 baseline (reuses src/train.py unchanged).

    `dataset` should be the exact same data the primary model was trained
    on (e.g. via src.data_pipeline.load_processed_dataset(config)), so the
    only difference between the two models is the physics loss weight.
    """
    baseline_config = copy.deepcopy(config)
    baseline_config["experiment_name"] = config["experiment_name"] + "_ablation_baseline"
    baseline_config["loss"]["w_phys"] = 0.0
    baseline_config["loss"]["w_phys_ramp"] = {**config["loss"].get("w_phys_ramp", {}), "enabled": False}

    # Never overwrite the primary model's artifacts, regardless of what
    # paths the caller's config points at.
    for key in ("model_checkpoint", "normalization_stats"):
        path = resolve_path(config, key)
        baseline_config["paths"][key] = str(path.with_name(path.stem + "_baseline" + path.suffix))

    return train(baseline_config, dataset, device=device, save=save)


def compare_residuals(
    biopinn_model, baseline_model, collocation_X: torch.Tensor, config: dict, norm_stats: dict
) -> dict:
    """Mean/max PDE residual + physical-consistency % for both models,
    evaluated on the same batch of collocation points."""
    threshold = config["evaluation"]["thresholds"]["mean_pde_residual"]

    biopinn_stats = compute_pde_residual_stats(biopinn_model, collocation_X, config, norm_stats)
    baseline_stats = compute_pde_residual_stats(baseline_model, collocation_X, config, norm_stats)

    def consistency_pct(residuals: np.ndarray) -> float:
        return float(np.mean(residuals < threshold) * 100.0)

    def improvement_factor(baseline_value: float, biopinn_value: float) -> float:
        return float(baseline_value / biopinn_value) if biopinn_value > 0 else float("inf")

    return {
        "biopinn": {
            "mean_abs_residual": biopinn_stats["mean_abs_residual"],
            "max_abs_residual": biopinn_stats["max_abs_residual"],
            "physical_consistency_pct": consistency_pct(biopinn_stats["residuals"]),
        },
        "baseline": {
            "mean_abs_residual": baseline_stats["mean_abs_residual"],
            "max_abs_residual": baseline_stats["max_abs_residual"],
            "physical_consistency_pct": consistency_pct(baseline_stats["residuals"]),
        },
        "improvement_factor_mean": improvement_factor(
            baseline_stats["mean_abs_residual"], biopinn_stats["mean_abs_residual"]
        ),
        "improvement_factor_max": improvement_factor(
            baseline_stats["max_abs_residual"], biopinn_stats["max_abs_residual"]
        ),
        "residuals_biopinn": biopinn_stats["residuals"],
        "residuals_baseline": baseline_stats["residuals"],
    }


def wilcoxon_test(residuals_biopinn: np.ndarray, residuals_baseline: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired PDE-residual magnitudes: is the
    baseline's residual significantly larger than BIOPINN's (one-sided)?"""
    statistic, p_value = wilcoxon(residuals_baseline, residuals_biopinn, alternative="greater")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "alternative": "baseline residuals > biopinn residuals",
        "n_pairs": int(len(residuals_biopinn)),
    }


def ablation_report(
    biopinn_model,
    baseline_model,
    config: dict,
    norm_stats: dict,
    sims: list[dict] | None = None,
    collocation_X: torch.Tensor | None = None,
    n_jobs: int = 1,
) -> dict:
    """Top-level entry point used by scripts/run_ablation.py."""
    sims = sims if sims is not None else resolve_test_simulations(config, n_jobs=n_jobs)

    biopinn_metrics = evaluate_test_set(biopinn_model, sims, norm_stats, config)
    baseline_metrics = evaluate_test_set(baseline_model, sims, norm_stats, config)

    if collocation_X is None:
        processed_dir = resolve_path(config, "processed")
        with np.load(processed_dir / "test.npz") as test_npz:
            collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)

    residual_comparison = compare_residuals(biopinn_model, baseline_model, collocation_X, config, norm_stats)
    wilcoxon_result = wilcoxon_test(
        residual_comparison["residuals_biopinn"], residual_comparison["residuals_baseline"]
    )

    hyp_cfg = config["evaluation"]["hypotheses"]
    h5_pass = (
        residual_comparison["improvement_factor_mean"] >= hyp_cfg["H5_residual_improvement_factor"]
        and wilcoxon_result["significant"]
    )

    return {
        "biopinn_data_metrics": biopinn_metrics["global"],
        "baseline_data_metrics": baseline_metrics["global"],
        "residual_comparison": {k: v for k, v in residual_comparison.items() if not k.startswith("residuals_")},
        "wilcoxon": wilcoxon_result,
        "H5": {
            "pass": bool(h5_pass),
            "improvement_factor_mean": residual_comparison["improvement_factor_mean"],
            "target_improvement_factor": hyp_cfg["H5_residual_improvement_factor"],
            "wilcoxon_p_value": wilcoxon_result["p_value"],
        },
        "n_test_sims": len(sims),
    }
