#!/usr/bin/env python3
"""CLI: generate only the tables the project's Statement of the Problem (SOP)
actually needs -- no figures, no docx, no ablation study, no hypothesis
summary, no PINN-vs-FDM speedup study. A trimmed-down alternative to
scripts/generate_results.py for when you just want the four research
questions' evidence, faster.

Writes CSVs to results/sop/:
  RQ1a  rq1a_penetration_vs_diameter.csv       (Table 4.2)
        rq1a_penetration_analysis.csv          (Table 4.4)
  RQ1b  rq1b_concentration_by_zone.csv
  RQ1c  rq1c_subtherapeutic_by_diameter.csv
  RQ2   rq2_accuracy_metrics.csv               (Table 4.3: RMSE/MAE/R2/L2)
  RQ3   rq3_viability_summary.csv              (Table 4.5)
  RQ4   rq4_optimization_results.csv           (Table 4.7)

Never retrains -- consumes artifacts/ exactly like scripts/run_evaluation.py.

Usage:
    python scripts/generate_sop_tables.py [--experiment NAME] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import results as R
from src.config import load_config, resolve_path
from src.evaluate import resolve_test_simulations
from src.model import load_checkpoint
from src.optimize import optimize_all_radii


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for re-solving the test set's FDM reference")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    processed_dir = resolve_path(config, "processed")
    if not (processed_dir / "test.npz").exists():
        print(f"No processed test set found at {processed_dir}.")
        print("Run notebooks/biopinn_train.ipynb (or src.data_pipeline.build_dataset) first.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)
    with open(resolve_path(config, "normalization_stats"), encoding="utf-8") as f:
        norm_stats = json.load(f)

    print(f"Re-solving FDM reference for the test set (n_jobs={args.n_jobs})...")
    sims = resolve_test_simulations(config, n_jobs=args.n_jobs)
    print(f"  {len(sims)} test simulations resolved.")

    with np.load(processed_dir / "test.npz") as test_npz:
        collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)

    output_dir = resolve_path(config, "results") / "sop"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSolving the nanoparticle-diameter FDM sweep (shared by every RQ1 table)...")
    diameter_sweep = R.solve_diameter_sweep(config)

    tables: list[tuple[str, str, "tuple"]] = []

    print("RQ1a -- penetration depth vs. diameter")
    tables.append(("rq1a_penetration_vs_diameter", "RQ1a", R.table_4_2_penetration_vs_diameter(config, diameter_sweep=diameter_sweep)))
    tables.append(("rq1a_penetration_analysis", "RQ1a", R.table_4_4_penetration_analysis(model, config, norm_stats, diameter_sweep=diameter_sweep)))

    print("RQ1b -- concentration distribution across tumor zones")
    tables.append(("rq1b_concentration_by_zone", "RQ1b", R.table_rq1b_concentration_by_zone(config, diameter_sweep=diameter_sweep)))

    print("RQ1c -- sub-therapeutic regions vs. diameter")
    tables.append(("rq1c_subtherapeutic_by_diameter", "RQ1c", R.table_rq1c_subtherapeutic_by_diameter(config, diameter_sweep=diameter_sweep)))

    print("RQ2 -- BIOPINN accuracy vs. the FDM reference (RMSE, MAE, R2, L2 relative error)")
    tables.append(("rq2_accuracy_metrics", "RQ2", R.table_4_3_pinn_metrics(model, config, norm_stats, sims, collocation_X)))

    print("RQ3 -- predicted cell survivability and cytotoxicity")
    tables.append(("rq3_viability_summary", "RQ3", R.table_4_5_viability_summary(model, config, norm_stats)))

    print("RQ4 -- optimal nanoparticle diameter and dose")
    radius_results = optimize_all_radii(model, config, norm_stats)
    tables.append(("rq4_optimization_results", "RQ4", R.table_4_7_optimization_results(model, config, norm_stats, radius_results=radius_results)))

    print(f"\nSaving {len(tables)} tables to {output_dir}...")
    for stem, rq, (df, _meta) in tables:
        csv_path = output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  [{rq}] {csv_path.name}  ({len(df)} rows)")

    print(f"\nDone -- {len(tables)} tables written to {output_dir}")


if __name__ == "__main__":
    main()
