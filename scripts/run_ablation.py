#!/usr/bin/env python3
"""CLI: physics-informed vs. unconstrained ablation study (H5).

Loads the primary BIOPINN checkpoint (never retrained here), trains a fresh
w_phys=0 baseline of identical architecture on the exact same processed
dataset, compares PDE-residual statistics on the held-out test set, runs a
Wilcoxon signed-rank test on the paired residual magnitudes, and reports
H5 pass/fail.

Usage:
    python scripts/run_ablation.py [--experiment NAME] [--n-jobs N] [--device cpu|cuda]

Requires paths.model_checkpoint and a processed dataset (paths.processed) to
already exist -- run notebooks/biopinn_train.ipynb first if they don't.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import ablation_report, train_baseline
from src.config import load_config, resolve_path
from src.data_pipeline import load_processed_dataset
from src.evaluate import resolve_test_simulations
from src.model import load_checkpoint


def _mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_report(report: dict) -> None:
    bio, base = report["biopinn_data_metrics"], report["baseline_data_metrics"]
    rc = report["residual_comparison"]
    wx = report["wilcoxon"]
    h5 = report["H5"]

    print("\n" + "=" * 70)
    print(f"ABLATION STUDY: BIOPINN vs. unconstrained baseline (w_phys=0) -- {report['n_test_sims']} test sims")
    print("=" * 70)
    print(f"{'Metric':<28}{'BIOPINN':<16}{'Baseline':<16}{'Comment'}")
    print(f"{'RMSE (uM)':<28}{bio['rmse']:<16.4e}{base['rmse']:<16.4e}{'expected comparable'}")
    print(f"{'R2':<28}{bio['r2']:<16.6f}{base['r2']:<16.6f}{'expected comparable'}")
    print(f"{'Mean |PDE residual|':<28}{rc['biopinn']['mean_abs_residual']:<16.4e}"
          f"{rc['baseline']['mean_abs_residual']:<16.4e}{'lower is better'}")
    print(f"{'Max |PDE residual|':<28}{rc['biopinn']['max_abs_residual']:<16.4e}"
          f"{rc['baseline']['max_abs_residual']:<16.4e}{'lower is better'}")
    print(f"{'Physical consistency %':<28}{rc['biopinn']['physical_consistency_pct']:<16.2f}"
          f"{rc['baseline']['physical_consistency_pct']:<16.2f}{'higher is better'}")
    print()
    print(f"Improvement factor (mean residual): {rc['improvement_factor_mean']:.2f}x")
    print(f"Improvement factor (max residual):  {rc['improvement_factor_max']:.2f}x")
    print(f"Wilcoxon signed-rank test (baseline residual > biopinn residual):")
    print(f"  statistic={wx['statistic']:.4e}  p-value={wx['p_value']:.4e}  n_pairs={wx['n_pairs']}"
          f"  -> {'significant' if wx['significant'] else 'not significant'}")
    print()
    print(
        f"H5 (improvement factor >= {h5['target_improvement_factor']}x AND Wilcoxon significant): "
        f"{_mark(h5['pass'])}"
    )
    print("=" * 70)


def save_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nSaved ablation_report.json to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for re-solving the test set's FDM reference")
    parser.add_argument("--device", default="cpu", help="Device for training the baseline (cpu or cuda)")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No primary checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    processed_dir = resolve_path(config, "processed")
    if not (processed_dir / "train.npz").exists():
        print(f"No processed dataset found at {processed_dir}.")
        print("Run notebooks/biopinn_train.ipynb (or src.data_pipeline.build_dataset) first.")
        sys.exit(1)

    print(f"Loading primary checkpoint: {checkpoint_path}")
    biopinn_model = load_checkpoint(str(checkpoint_path), config)

    print(f"Loading processed dataset from: {processed_dir}")
    dataset = load_processed_dataset(config)
    norm_stats = dataset["stats"]

    print("Training w_phys=0 baseline (identical architecture, same data)...")
    baseline_result = train_baseline(config, dataset, device=args.device, save=True)
    baseline_model = baseline_result["model"]
    print(
        f"Baseline training complete "
        f"({baseline_result['adam_epochs_run']} Adam epochs, "
        f"{baseline_result['lbfgs_closure_evaluations']} L-BFGS closure evaluations)."
    )

    print(f"Re-solving FDM reference for the test set (n_jobs={args.n_jobs})...")
    sims = resolve_test_simulations(config, n_jobs=args.n_jobs)
    print(f"  {len(sims)} test simulations resolved.")

    report = ablation_report(biopinn_model, baseline_model, config, norm_stats, sims=sims)
    print_report(report)

    output_dir = resolve_path(config, "results") / "ablation"
    save_report(report, output_dir)


if __name__ == "__main__":
    main()
