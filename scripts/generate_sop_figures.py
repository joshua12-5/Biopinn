#!/usr/bin/env python3
"""CLI: generate only the figures the project's Statement of the Problem (SOP)
actually needs -- no tables, no docx, no ablation study, no hypothesis
summary, no PINN-vs-FDM speedup study. Figure counterpart to
scripts/generate_sop_tables.py; a trimmed-down alternative to
scripts/generate_results.py for when you just want the four research
questions' figures, faster.

Writes 300 DPI PNG+PDF pairs to results/sop/:
  RQ1a  rq1a_penetration_depth_vs_time.png/.pdf
  RQ1b  rq1b_concentration_by_zone.png/.pdf
  RQ1c  rq1c_subtherapeutic_vs_diameter.png/.pdf
  RQ2   (no figure -- RQ2's accuracy metrics are reported as a table only,
        see scripts/generate_sop_tables.py; no FDM-comparison plot)
  RQ3   rq3_spatial_viability.png/.pdf
        rq3_cytotoxicity_evolution.png/.pdf
  RQ4   rq4_effectiveness_surface.png/.pdf

Never retrains -- consumes artifacts/ exactly like scripts/run_evaluation.py.

Usage:
    python scripts/generate_sop_figures.py [--experiment NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import results as R
from src.config import load_config, resolve_path
from src.model import load_checkpoint
from src.optimize import optimize_all_radii


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)
    with open(resolve_path(config, "normalization_stats"), encoding="utf-8") as f:
        norm_stats = json.load(f)

    output_dir = resolve_path(config, "results") / "sop"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSolving the nanoparticle-diameter FDM sweep (shared by every RQ1 figure)...")
    diameter_sweep = R.solve_diameter_sweep(config)

    figures: list[tuple[str, str, tuple]] = []

    print("RQ1a -- penetration depth vs. time, across diameters")
    figures.append(("rq1a_penetration_depth_vs_time", "RQ1a", R.fig_4_6_penetration_vs_time(model, config, norm_stats)))

    print("RQ1b -- concentration by tumor zone across diameters")
    figures.append(("rq1b_concentration_by_zone", "RQ1b", R.fig_rq1b_concentration_by_zone(config, diameter_sweep=diameter_sweep)))

    print("RQ1c -- sub-therapeutic region vs. diameter")
    figures.append(("rq1c_subtherapeutic_vs_diameter", "RQ1c", R.fig_rq1c_subtherapeutic_vs_diameter(config, diameter_sweep=diameter_sweep)))

    print("RQ3 -- spatial viability at t=t_max")
    figures.append(("rq3_spatial_viability", "RQ3", R.fig_4_7_viability_t72(model, config, norm_stats)))
    print("RQ3 -- cytotoxicity evolution over time")
    figures.append(("rq3_cytotoxicity_evolution", "RQ3", R.fig_4_8_cytotoxicity_evolution(model, config, norm_stats)))

    print("RQ4 -- treatment effectiveness surface")
    radius_results = optimize_all_radii(model, config, norm_stats)
    baseline_R = R.baseline_params(config)["R_um"]
    figures.append(("rq4_effectiveness_surface", "RQ4", R.fig_4_10_effectiveness_surface(model, config, norm_stats, grid_result=radius_results[baseline_R])))

    print(f"\nSaving {len(figures)} figures to {output_dir}...")
    for stem, rq, (fig, meta) in figures:
        paths = R._save_figure(fig, output_dir, stem)
        print(f"  [{rq}] {Path(paths['png']).name}  ({meta.get('caption', '')})")

    print(f"\nDone -- {len(figures)} figures written to {output_dir}")


if __name__ == "__main__":
    main()
