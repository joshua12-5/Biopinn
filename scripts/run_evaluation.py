#!/usr/bin/env python3
"""CLI: evaluate a trained BIOPINN checkpoint against the held-out test set.

Loads artifacts/biopinn_model.pt + normalization_stats.json, re-solves the
FDM reference for every test-set simulation (data/processed/sim_params.json,
written by src.data_pipeline.build_dataset), computes all six metrics
globally and decomposed by tumor zone / nanoparticle-size range / time
range, checks H1/H2/H4, prints a full report, and saves metrics (JSON/CSV)
plus a PDE-residual histogram and a PINN-vs-FDM overlay figure to
paths.results/evaluation/.

Usage:
    python scripts/run_evaluation.py [--experiment NAME] [--n-jobs N]

Never retrains -- if no checkpoint is found, run notebooks/biopinn_train.ipynb
on Colab first and drop its artifacts into artifacts/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.biology import predict_concentration_field
from src.config import load_config, resolve_path
from src.evaluate import full_evaluation_report, resolve_test_simulations
from src.model import load_checkpoint
from src.visualize import plot_pde_residual_histogram, plot_pinn_vs_fdm_overlay

THRESHOLD_UNITS = {
    "rmse": "uM",
    "mae": "uM",
    "r2": "",
    "l2_relative_error": "",
    "mean_pde_residual": "",
    "penetration_rmse_um": "um",
}


def _load_normalization_stats(config: dict) -> dict:
    stats_path = resolve_path(config, "normalization_stats")
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def _mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_report(report: dict) -> None:
    g = report["metrics"]["global"]
    pf = report["threshold_pass_fail"]

    print("\n" + "=" * 64)
    print(f"BIOPINN EVALUATION -- {report['n_test_sims']} test simulations")
    print("=" * 64)
    print(f"{'Metric':<24}{'Value':<16}{'Result'}")
    print(f"{'RMSE (uM)':<24}{g['rmse']:<16.4e}{_mark(pf['rmse'])}")
    print(f"{'MAE (uM)':<24}{g['mae']:<16.4e}{_mark(pf['mae'])}")
    print(f"{'R2':<24}{g['r2']:<16.6f}{_mark(pf['r2'])}")
    print(f"{'L2 relative error':<24}{g['l2_relative']:<16.4e}{_mark(pf['l2_relative_error'])}")
    print(f"{'Mean PDE residual':<24}{g['mean_pde_residual']:<16.4e}{_mark(pf['mean_pde_residual'])}")
    print(f"{'Penetration RMSE (um)':<24}{g['penetration_rmse_um']:<16.4f}{_mark(pf['penetration_rmse_um'])}")

    for decomposition_name, decomposition in report["metrics"]["decomposed"].items():
        print(f"\n--- decomposed {decomposition_name} ---")
        for bucket, m in decomposition.items():
            print(f"  {bucket:<16} n={m['n_points']:<8} rmse={m['rmse']:.4e}  r2={m['r2']:.4f}")

    print("\n--- hypotheses ---")
    h1, h2, h4 = report["hypotheses"]["H1"], report["hypotheses"]["H2"], report["hypotheses"]["H4"]
    print(
        f"  H1 (rmse<{h1['target_rmse_uM']}uM, r2>{h1['target_r2']}): "
        f"rmse={h1['rmse_uM']:.4e} r2={h1['r2']:.4f} -> {_mark(h1['pass'])}"
    )
    print(
        f"  H2 (10nm vs 200nm penetration diff > {h2['target_um']}um @ R=400um, t=72hr): "
        f"10nm={h2['depth_10nm_um']:.1f}um 200nm={h2['depth_200nm_um']:.1f}um "
        f"diff={h2['difference_um']:.1f}um -> {_mark(h2['pass'])}"
    )
    print(
        f"  H4 (rim<20% & core>60% viability @ t=72hr): "
        f"pass_rate={h4['pass_rate']*100:.1f}% over {h4['n_sims']} sims -> {_mark(h4['pass'])}"
    )
    print(
        "  H3, H5, H6 are reported by scripts/run_optimization.py and "
        "scripts/run_ablation.py."
    )
    print("=" * 64)


def save_metrics(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    exportable = {k: v for k, v in report.items() if k != "residual_histogram"}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    g = report["metrics"]["global"]
    pf = report["threshold_pass_fail"]
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit", "pass"])
        for key in ("rmse", "mae", "r2", "l2_relative", "mean_pde_residual", "penetration_rmse_um"):
            pf_key = key if key != "l2_relative" else "l2_relative_error"
            unit = THRESHOLD_UNITS.get(pf_key, "")
            writer.writerow([key, g[key], unit, pf[pf_key]])

    print(f"\nSaved metrics.json / metrics.csv to {output_dir}")


def make_residual_histogram_figure(report: dict, config: dict, output_dir: Path) -> None:
    residuals = report["residual_histogram"]
    plot_pde_residual_histogram(residuals, config, save_path=str(output_dir / "pde_residual_histogram.png"))


def make_pinn_vs_fdm_overlay_figure(model, sims: list[dict], norm_stats: dict, config: dict, output_dir: Path) -> None:
    sim = sims[0]
    r, t = sim["r"], sim["t"]
    C_pred = predict_concentration_field(model, r, t, sim, norm_stats)
    plot_pinn_vs_fdm_overlay(sim, C_pred, config, save_path=str(output_dir / "pinn_vs_fdm_overlay.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for re-solving the test set's FDM reference")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)
    norm_stats = _load_normalization_stats(config)

    sim_params_path = resolve_path(config, "processed") / "sim_params.json"
    if not sim_params_path.exists():
        print(f"No test-set simulation parameters found at {sim_params_path}.")
        print("Run notebooks/biopinn_train.ipynb (or src.data_pipeline.build_dataset) first.")
        sys.exit(1)

    print(f"Re-solving FDM reference for the test set (n_jobs={args.n_jobs})...")
    sims = resolve_test_simulations(config, n_jobs=args.n_jobs)
    print(f"  {len(sims)} test simulations resolved.")

    report = full_evaluation_report(model, config, norm_stats, sims=sims)
    print_report(report)

    output_dir = resolve_path(config, "results") / "evaluation"
    save_metrics(report, output_dir)
    make_residual_histogram_figure(report, config, output_dir)
    make_pinn_vs_fdm_overlay_figure(model, sims, norm_stats, config, output_dir)
    print(f"Saved pde_residual_histogram.png / pinn_vs_fdm_overlay.png to {output_dir}")


if __name__ == "__main__":
    main()
