#!/usr/bin/env python3
"""CLI: surrogate-based optimization + efficiency study.

Loads the trained checkpoint (never retrained here) and:
  1. Grid-searches (d_NP, C0) at R in {200,300,400,500}um to maximize the
     volume-averaged kill fraction eta, reporting per-radius optimal
     (d_NP*, C0*), max eta, the resistance-zone map, and computation time.
  2. Runs the homogeneous-vs-heterogeneous diffusion comparison (H3).
  3. Runs the PINN-vs-FDM speedup study (H6).

Prints a full report and saves it (JSON) plus the eta(d_NP,C0) surface per
radius to results/optimization/.

Usage:
    python scripts/run_optimization.py [--experiment NAME] [--n-speedup N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.model import load_checkpoint
from src.optimize import homogeneous_vs_heterogeneous, optimize_all_radii, speedup_study


def _mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_radius_results(radius_results: dict) -> None:
    print("\n" + "=" * 78)
    print("OPTIMIZATION: (d_NP*, C0*) maximizing volume-averaged kill fraction, per tumor radius")
    print("=" * 78)
    print(f"{'R (um)':<10}{'d_NP* (nm)':<14}{'C0* (uM)':<12}{'max eta':<12}{'resistant %':<14}{'time (s)'}")
    for R_um, result in sorted(radius_results.items()):
        rm = result["resistance_map"]
        print(
            f"{R_um:<10.0f}{result['d_NP_star_nm']:<14.1f}{result['C0_star_uM']:<12.2f}"
            f"{result['max_eta']:<12.4f}{rm['resistant_volume_fraction']*100:<14.2f}"
            f"{result['computation_time_s']:.2f} ({result['n_combinations']} combos)"
        )
    print("=" * 78)


def print_h3_result(h3_result: dict) -> None:
    h, g = h3_result["heterogeneous"], h3_result["homogeneous"]
    p = h3_result["parameters"]
    print("\n" + "=" * 78)
    print(
        f"H3: HOMOGENEOUS vs. HETEROGENEOUS DIFFUSION "
        f"(d_NP={p['d_NP_nm']}nm, R={p['R_um']}um, C0={p['C0_uM']}uM, t={p['t_max_hr']}hr)"
    )
    print("=" * 78)
    print(f"D_eff homogeneous (mean of 3 zones): {h3_result['D_eff_homogeneous_um2_per_hr']:.1f} um^2/hr")
    print(f"{'Metric':<32}{'Heterogeneous':<18}{'Homogeneous'}")
    print(f"{'Max penetration depth (um)':<32}{h['max_penetration_depth_um']:<18.1f}{g['max_penetration_depth_um']:.1f}")
    print(f"{'Sub-therapeutic zone radius (um)':<32}{h['subtherapeutic_zone_radius_um']:<18.1f}{g['subtherapeutic_zone_radius_um']:.1f}")
    print(f"{'Resistance-risk fraction':<32}{h['resistance_risk_fraction']:<18.4f}{g['resistance_risk_fraction']:.4f}")
    print(f"{'Kill fraction':<32}{h['kill_fraction']:<18.4f}{g['kill_fraction']:.4f}")
    print(f"{'Avg concentration gradient':<32}{h['avg_concentration_gradient_uM_per_um']:<18.6f}{g['avg_concentration_gradient_uM_per_um']:.6f}")
    print(f"\nH3 (heterogeneous shows steeper gradient AND larger sub-therapeutic zone): {_mark(h3_result['H3']['pass'])}")
    print("=" * 78)


def print_speedup_result(speedup_result: dict) -> None:
    h6 = speedup_result["H6"]
    print("\n" + "=" * 78)
    print(f"H6: PINN-vs-FDM SPEEDUP STUDY ({speedup_result['n_combinations']} parameter combinations)")
    print("=" * 78)
    print(f"PINN surrogate: {speedup_result['pinn_time_mean_s']*1000:.3f} +/- {speedup_result['pinn_time_std_s']*1000:.3f} ms")
    print(f"FDM solver:     {speedup_result['fdm_time_mean_s']:.4f} +/- {speedup_result['fdm_time_std_s']:.4f} s")
    print(
        f"Speedup ratio:  {speedup_result['speedup_ratio_mean']:.1f}x mean "
        f"(median {speedup_result['speedup_ratio_median']:.1f}x, std {speedup_result['speedup_ratio_std']:.1f}x)"
    )
    print(f"\nH6 (speedup >= {h6['target_speedup']:.0f}x = 10^{h6['target_orders_of_magnitude']}): {_mark(h6['pass'])}")
    print("=" * 78)


def save_report(radius_results: dict, h3_result: dict, speedup_result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    report = {
        "radius_results": {
            str(R_um): {k: v for k, v in result.items() if k != "resistance_map"} | {
                "resistant_volume_fraction": result["resistance_map"]["resistant_volume_fraction"]
            }
            for R_um, result in radius_results.items()
        },
        "H3": h3_result,
        "H6": speedup_result,
    }
    with open(output_dir / "optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=default)
    print(f"\nSaved optimization_report.json to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--n-speedup", type=int, default=None, help="Override optimization.speedup_study.n_combinations")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)

    stats_path = resolve_path(config, "normalization_stats")
    with open(stats_path, encoding="utf-8") as f:
        norm_stats = json.load(f)

    print("Running grid search across tumor radii (200/300/400/500 um)...")
    radius_results = optimize_all_radii(model, config, norm_stats)
    print_radius_results(radius_results)

    print("\nRunning homogeneous-vs-heterogeneous diffusion comparison (H3)...")
    h3_result = homogeneous_vs_heterogeneous(config)
    print_h3_result(h3_result)

    print("\nRunning PINN-vs-FDM speedup study (H6)...")
    speedup_result = speedup_study(model, config, norm_stats, n_combinations=args.n_speedup)
    print_speedup_result(speedup_result)

    output_dir = resolve_path(config, "results") / "optimization"
    save_report(radius_results, h3_result, speedup_result, output_dir)


if __name__ == "__main__":
    main()
